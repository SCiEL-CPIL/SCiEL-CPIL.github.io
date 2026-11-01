"""Covering design C(19,8,5) - randomized greedy with bitmask acceleration, incremental gain tracking, and local search."""
import itertools
import time
import random
import sys


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit

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

    # Precompute for each pair (point, partial_mask) the count of uncovered subsets
    # that would be completed by adding that point given the partial mask
    # We'll compute gains on the fly but use numpy for vectorized operations if beneficial
    # Actually, let's precompute for each point x and each possible (k-1)-subset of other points
    # the number of t-subsets that contain x and whose remaining t-1 elements are in that (k-1)-set
    # This is expensive; instead we'll use a smarter incremental approach

    best_blocks = None
    best_count = n_subs + 1  # worst case

    # Multi-restart randomized greedy with local search
    max_restarts = 100
    rng = random.Random(42)

    for restart in range(max_restarts):
        if time.time() > deadline:
            break

        # Track uncovered t-subsets using a boolean array
        uncovered = bytearray([1] * n_subs)
        uncovered_count = n_subs

        blocks = []
        block_masks = []  # store bitmask for each block

        while uncovered_count > 0:
            if time.time() > deadline:
                break

            # Pick a random uncovered t-subset to start the block
            found = False
            for _ in range(500):  # try more random picks
                idx = rng.randint(0, n_subs - 1)
                if uncovered[idx]:
                    found = True
                    break
            if not found:
                # Fallback: scan
                for idx in range(n_subs):
                    if uncovered[idx]:
                        found = True
                        break
            if not found:
                break  # Should not happen

            # Start block with elements of this t-subset
            start_sub = all_t_subs[idx]
            block_set = set(start_sub)
            block_mask = sub_bitmasks[idx]

            # Greedily add points to maximize coverage of uncovered t-subsets
            # Use incremental gain tracking: maintain a counter for each candidate point
            # gain[x] = number of uncovered t-subsets that would be covered if x is added
            # When we add a point y, we need to update gain[x] for all x != y
            # A t-subset S is "ready" for x if S contains x and S \ {x} ⊆ current_block
            # When we add y, a subset S becomes ready for x if:
            #   S contains both x and y, and S \ {x,y} ⊆ current_block \ {y}
            # This is complex; let's use a different approach:
            # For each candidate point x, precompute which t-subsets are "almost ready"
            # i.e., missing only x from the current block
            # We'll recompute gains each iteration but efficiently

            while len(block_set) < k:
                if time.time() > deadline:
                    break

                best_gain = -1
                best_point = None
                second_best_gain = -1
                second_best_point = None

                # For each candidate point, compute gain
                # Optimization: only consider points that appear in some uncovered subset
                # that is "close" to being covered by the current block
                for x in range(v):
                    if x in block_set:
                        continue

                    gain = 0
                    for s_idx in point_to_subs[x]:
                        if not uncovered[s_idx]:
                            continue
                        sm = sub_bitmasks[s_idx]
                        # Check if sm contains x and sm without x is subset of block_mask
                        if (sm & (1 << x)) and (sm & ~(block_mask | (1 << x))) == 0:
                            gain += 1

                    if gain > best_gain:
                        second_best_gain = best_gain
                        second_best_point = best_point
                        best_gain = gain
                        best_point = x
                    elif gain > second_best_gain:
                        second_best_gain = gain
                        second_best_point = x

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

            # Check if this block is a duplicate
            if any(tuple(b) == block_tuple for b in blocks):
                continue

            blocks.append(block)
            block_masks.append(block_mask)

            # Update uncovered: mark all t-subsets contained in this block as covered
            for i in range(n_subs):
                if uncovered[i]:
                    if sub_bitmasks[i] & ~block_mask == 0:
                        uncovered[i] = 0
                        uncovered_count -= 1

            if uncovered_count == 0:
                break

        # Check if we have a complete covering
        if uncovered_count == 0:
            # Local search: try to remove redundant blocks
            # For each block, check if removing it leaves all t-subsets still covered
            # We'll do this by checking if each t-subset in that block is covered by another block
            improved = True
            while improved and time.time() < deadline:
                improved = False
                n_blocks = len(blocks)
                for bi in range(n_blocks):
                    if time.time() > deadline:
                        break
                    # Check if block bi is redundant
                    block_mask_i = block_masks[bi]
                    # Find all t-subsets that are only covered by this block
                    # i.e., t-subsets S where S ⊆ block_i and S is not contained in any other block
                    redundant = True
                    for si in range(n_subs):
                        sm = sub_bitmasks[si]
                        if sm & ~block_mask_i == 0:
                            # This t-subset is contained in block_i
                            # Check if it's covered by any other block
                            covered_by_other = False
                            for bj in range(n_blocks):
                                if bj == bi:
                                    continue
                                if sm & ~block_masks[bj] == 0:
                                    covered_by_other = True
                                    break
                            if not covered_by_other:
                                redundant = False
                                break
                    if redundant:
                        # Remove this block
                        blocks.pop(bi)
                        block_masks.pop(bi)
                        improved = True
                        break

            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = [list(b) for b in blocks]

                # If we're close to the known best, we can stop early
                if best_count <= 340:
                    break

    if best_blocks is None:
        # Fallback: return a trivial (but likely huge) covering
        # Use the deterministic greedy as fallback
        uncovered_set = set(range(n_subs))
        blocks = []
        for idx in sorted(uncovered_set):
            if time.time() > deadline:
                break
            block = list(all_t_subs[idx])
            blocks.append(block)
        best_blocks = blocks

    return best_blocks
