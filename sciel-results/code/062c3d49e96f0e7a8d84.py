"""Covering design C(19,8,5) - incremental-gain randomized greedy with many restarts."""
import itertools
import time
import random
import sys


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s limit

    # Precompute all t-subsets and encode as bitmasks
    all_t_subs = list(itertools.combinations(range(v), t))
    n_subs = len(all_t_subs)

    # Map each t-subset to a bitmask of its elements
    sub_bitmasks = [0] * n_subs
    for i, sub in enumerate(all_t_subs):
        mask = 0
        for x in sub:
            mask |= (1 << x)
        sub_bitmasks[i] = mask

    # For each point, precompute which t-subsets contain it
    point_to_subs = [[] for _ in range(v)]
    for i, mask in enumerate(sub_bitmasks):
        for x in range(v):
            if mask & (1 << x):
                point_to_subs[x].append(i)

    best_blocks = None
    best_count = n_subs + 1  # worst case

    # Multi-restart randomized greedy with incremental gain tracking
    rng = random.Random(42)

    # We'll run restarts until deadline
    restart = 0
    while time.time() < deadline:
        restart += 1
        if time.time() > deadline:
            break

        # Track uncovered t-subsets using a boolean array
        uncovered = bytearray([1] * n_subs)
        uncovered_count = n_subs

        blocks = []
        block_set_list = []  # list of sets for each block

        # For incremental gain: for each point x, track how many uncovered subsets
        # are "almost covered" by the current partial block (i.e., all elements except x are in block)
        # gain[x] = number of uncovered t-subsets S such that x in S and (S \ {x}) ⊆ current_block

        # We maintain this incrementally as we add points to the current block
        # When we add point p to the block, for each other point x:
        #   gain[x] changes: subsets that were "almost covered" by x become covered (if they contain p)
        #   Also new subsets become "almost covered" by x (those containing both x and p, with rest in block)

        # Actually, let's use a simpler but still efficient approach:
        # For the current block being built, maintain:
        #   block_mask: bitmask of points in the current block
        #   For each candidate point x, gain[x] = count of uncovered subsets S where x in S and (S \ {x}) ⊆ block

        # To compute gain efficiently:
        # When block has m points, a subset S is "almost covered" by x if:
        #   x in S, |S \ (block ∪ {x})| = 0, and S \ {x} ⊆ block
        # This means S ⊆ block ∪ {x} and x in S.
        # So gain[x] = number of uncovered t-subsets S such that S ⊆ block ∪ {x} and x in S.
        # Equivalently: S \ {x} ⊆ block and x in S, and S is uncovered.

        # Let's precompute for each pair (x, y) the list of t-subsets containing both x and y.
        # And for each triple, etc. But that's expensive.

        # Alternative: maintain a counter array. For the current block B (as a set),
        # for each x not in B, gain[x] = |{S : S uncovered, x in S, S \ {x} ⊆ B}|
        # = |{S : S uncovered, x in S, S ⊆ B ∪ {x}}|

        # We can compute this by iterating over point_to_subs[x] and checking if S \ {x} ⊆ B.
        # That's O(|point_to_subs[x]|) per x, which is O(C(v-1, t-1)) per x, O(v * C(v-1,t-1)) per step.
        # For v=19, t=5: C(18,4) = 3060 per x, 19*3060 = 58140 per step. With ~335 blocks * 3 points each = ~1000 steps.
        # Total: 58140 * 1000 = 58M operations per restart. That might be too slow.

        # Let's use a smarter incremental approach:
        # Maintain an array "almost_count" where almost_count[S] = number of elements of S that are NOT in the current block.
        # When we add point p to the block, for each S containing p, almost_count[S] -= 1.
        # Then gain[x] for x not in block = number of uncovered S such that x in S and almost_count[S] == 1 and the missing element is x.
        # Wait, almost_count[S] == 1 means exactly one element of S is not in the block. If that element is x, then S \ {x} ⊆ block.
        # So gain[x] = number of uncovered S where x in S and almost_count[S] == 1.

        # But we need to be careful: if almost_count[S] == 1, the missing element is unique. So we can track:
        # missing_elem[S] = the element of S not in the block (if almost_count[S] == 1), else -1.

        # This requires updating missing_elem when we add a point, which is complex.

        # Let's go with a hybrid: use the direct computation but optimize with numpy or just accept the cost.
        # Given v=19, t=5, the per-step cost is manageable if we have enough time for restarts.

        # Actually, let's try a different strategy: instead of building one block at a time with greedy,
        # use a "global greedy" where at each step we pick the block (8-subset) that covers the most uncovered subsets.
        # But enumerating all C(19,8) = 75582 blocks and computing coverage for each is expensive.

        # Let's stick with the incremental approach but optimize the inner loop.
        # For each restart, we'll build blocks one by one. For each block, we greedily add points.
        # To speed up gain computation, we'll precompute for each point x and each subset of points (as bitmask),
        # but that's too large.

        # Practical approach: just do the direct computation and see how fast it is.
        # For v=19, t=5: C(18,4)=3060 subsets per point. 19 points * 3060 = 58140 checks per gain computation.
        # Each check is a bitmask AND operation. Let's say ~100ns each, so ~5.8ms per gain computation.
        # For a block of 8 points, we do 8-5=3 gain computations (start with 5 points, add 3 more).
        # So ~17ms per block. For 335 blocks, ~5.7s per restart. With 115s budget, ~20 restarts.

        # That's similar to the current approach. Let's try to optimize further.

        # Key optimization: instead of checking all point_to_subs[x] for each x, only check those that are uncovered.
        # And use bitmask operations for the subset check.

        # Let's implement this:

        while uncovered_count > 0:
            if time.time() > deadline:
                break

            # Find an uncovered subset
            # Use random sampling with fallback
            found = False
            for _ in range(50):
                idx = rng.randint(0, n_subs - 1)
                if uncovered[idx]:
                    found = True
                    break
            if not found:
                for idx in range(n_subs):
                    if uncovered[idx]:
                        found = True
                        break
            if not found:
                break

            # Start block with elements of this t-subset
            start_sub = all_t_subs[idx]
            block_set = set(start_sub)
            block_mask = sub_bitmasks[idx]

            # Greedily add points to maximize coverage of uncovered t-subsets
            while len(block_set) < k:
                if time.time() > deadline:
                    break

                best_gain = -1
                best_point = None

                for x in range(v):
                    if x in block_set:
                        continue
                    # Compute gain: number of uncovered t-subsets S where x in S and S \ {x} ⊆ block
                    gain = 0
                    for s_idx in point_to_subs[x]:
                        if not uncovered[s_idx]:
                            continue
                        sm = sub_bitmasks[s_idx]
                        # Check if sm contains x and sm \ {x} ⊆ block_mask
                        if sm & (1 << x):
                            if sm & ~(block_mask | (1 << x)) == 0:
                                gain += 1

                    if gain > best_gain:
                        best_gain = gain
                        best_point = x

                if best_point is None or best_gain <= 0:
                    # No point adds coverage; pick a random one to complete the block
                    remaining = [x for x in range(v) if x not in block_set]
                    if remaining:
                        best_point = rng.choice(remaining)
                    else:
                        break

                block_set.add(best_point)
                block_mask |= (1 << best_point)

            # Convert block to sorted list
            block = sorted(block_set)
            block_tuple = tuple(block)

            # Check for duplicate
            is_dup = False
            for b in blocks:
                if tuple(b) == block_tuple:
                    is_dup = True
                    break
            if is_dup:
                continue

            blocks.append(block)
            block_set_list.append(block_set)

            # Update uncovered: mark all t-subsets contained in this block as covered
            for i in range(n_subs):
                if uncovered[i]:
                    if sub_bitmasks[i] & ~block_mask == 0:
                        uncovered[i] = 0
                        uncovered_count -= 1

            if uncovered_count == 0:
                break

        # Check if we have a complete covering
        if uncovered_count == 0 and len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = [list(b) for b in blocks]

            if best_count <= 335:
                break

    if best_blocks is None:
        # Fallback: deterministic greedy
        uncovered_set = set(range(n_subs))
        blocks = []
        for idx in sorted(uncovered_set):
            if time.time() > deadline:
                break
            block = list(all_t_subs[idx])
            blocks.append(block)
        best_blocks = blocks

    return best_blocks
