"""
Covering design C(v,k,t) via structured greedy + vectorized local search.
"""
import itertools
import time
import random
from collections import defaultdict


def construct(v, k, t):
    """
    Construct a covering design C(v,k,t): choose as few k-subsets of {0..v-1}
    as possible so that every t-subset is contained in at least one block.

    Uses:
    - Structured greedy construction (deterministic + randomized restarts)
    - Precomputed incidence for fast gain calculation
    - Bitset-based local search: block deletion and point swaps to create redundancy
    - Deadline guard to return best result before timeout
    """
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin before 150s kill

    def elapsed():
        return time.time() - start_time

    def time_remaining():
        return DEADLINE - elapsed()

    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t_subsets = len(all_t_subsets)
    t_subset_index = {ts: i for i, ts in enumerate(all_t_subsets)}

    # For fast gain computation: for each point x, list of t-subset indices containing x
    point_to_tsubsets = defaultdict(list)
    for i, ts in enumerate(all_t_subsets):
        for p in ts:
            point_to_tsubsets[p].append(i)

    def get_block_triples(block):
        """Return set of t-subset indices covered by a block."""
        return set(t_subset_index[ts] for ts in itertools.combinations(block, t))

    def greedy_construct(seed=0, structured=False):
        """
        Greedy set cover. If structured=True, use deterministic ordering to
        exploit symmetry (pick uncovered t-subset with smallest lexicographic
        elements, prefer points that appear in many uncovered triples).
        """
        rng = random.Random(seed)
        uncovered = set(range(n_t_subsets))
        blocks = []

        if not uncovered:
            return blocks

        # Precompute for each point, count of uncovered t-subsets (updated dynamically)
        uncovered_count_per_point = [0] * v
        for i in range(n_t_subsets):
            for p in all_t_subsets[i]:
                uncovered_count_per_point[p] += 1

        while uncovered and time_remaining() > 2.0:
            # Pick a seed t-subset
            if structured and seed == 0:
                # Deterministic: pick the uncovered t-subset with smallest index
                seed_ts_idx = min(uncovered)
            else:
                # Random pick for diversity
                seed_ts_idx = rng.choice(list(uncovered))

            seed_ts = all_t_subsets[seed_ts_idx]
            block = list(seed_ts)
            block_set = set(block)

            # Greedily add points until block has k elements
            while len(block) < k and time_remaining() > 1.0:
                # For each candidate point x not in block, compute gain
                best_x = None
                best_gain = -1
                best_secondary = -1

                # Precompute (t-1)-subsets of current block
                b_t_minus_1 = list(itertools.combinations(block, t - 1))

                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for S in b_t_minus_1:
                        ts = tuple(sorted(S + (x,)))
                        idx = t_subset_index.get(ts, -1)
                        if idx != -1 and idx in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                        best_secondary = uncovered_count_per_point[x]
                    elif gain == best_gain and gain > 0:
                        if uncovered_count_per_point[x] > best_secondary:
                            best_secondary = uncovered_count_per_point[x]
                            best_x = x
                    elif gain == 0 and best_x is None:
                        best_x = x
                        best_gain = 0
                        best_secondary = 0

                if best_x is None:
                    # All in block; pick first available
                    for x in range(v):
                        if x not in block_set:
                            best_x = x
                            break
                    if best_x is None:
                        break
                    best_gain = 0

                block.append(best_x)
                block_set.add(best_x)

            # Sort the block
            block.sort()
            blocks.append(block)

            # Update uncovered and uncovered_count_per_point
            block_t_subs = list(itertools.combinations(block, t))
            for ts in block_t_subs:
                idx = t_subset_index.get(ts, -1)
                if idx != -1 and idx in uncovered:
                    uncovered.discard(idx)
                    for p in all_t_subsets[idx]:
                        uncovered_count_per_point[p] -= 1

        # If we ran out of time, finish the remaining uncovered triples with a quick deterministic pass
        while uncovered and time_remaining() > 0.5:
            seed_ts_idx = min(uncovered)
            seed_ts = all_t_subsets[seed_ts_idx]
            block = list(seed_ts)
            block_set = set(block)
            while len(block) < k:
                best_x = None
                best_gain = -1
                b_t_minus_1 = list(itertools.combinations(block, t - 1))
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for S in b_t_minus_1:
                        ts = tuple(sorted(S + (x,)))
                        idx = t_subset_index.get(ts, -1)
                        if idx != -1 and idx in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x is None:
                    for x in range(v):
                        if x not in block_set:
                            best_x = x
                            break
                    if best_x is None:
                        break
                block.append(best_x)
                block_set.add(best_x)
            block.sort()
            blocks.append(block)
            block_t_subs = list(itertools.combinations(block, t))
            for ts in block_t_subs:
                idx = t_subset_index.get(ts, -1)
                if idx != -1:
                    uncovered.discard(idx)

        return blocks

    def local_search(blocks, seed=0, max_time=30.0):
        """
        Local search to minimize blocks:
        1. Try to delete blocks whose triples are all covered by other blocks.
        2. Try point swaps to create redundancy, then delete.
        Uses coverage counts for O(1) checks.
        """
        rng = random.Random(seed + 1000)
        end_time = start_time + max_time

        # Build coverage counts
        coverage = [0] * n_t_subsets
        block_triples_list = []
        for block in blocks:
            triples = get_block_triples(block)
            block_triples_list.append(triples)
            for idx in triples:
                coverage[idx] += 1

        def try_delete_block(bi):
            """Try to delete block bi. Returns True if successful."""
            for idx in block_triples_list[bi]:
                if coverage[idx] <= 1:
                    return False
            for idx in block_triples_list[bi]:
                coverage[idx] -= 1
            blocks.pop(bi)
            block_triples_list.pop(bi)
            return True

        def try_swap_point(bi, old_x, new_x):
            """Try to swap old_x with new_x in block bi. Returns True if successful."""
            if new_x in blocks[bi]:
                return False
            old_block = blocks[bi]
            old_triples = block_triples_list[bi]
            # Check if any triple in old_block would lose its only coverage
            for idx in old_triples:
                if coverage[idx] <= 1:
                    # This triple is only covered by this block; if we remove old_x,
                    # we need to ensure it's still covered by the new block
                    ts = all_t_subsets[idx]
                    if old_x not in ts:
                        continue  # Triple doesn't contain old_x, so it remains covered
                    # Triple contains old_x and is only covered by this block.
                    return False

            # Compute new block's triples
            new_block = sorted([new_x] + [p for p in old_block if p != old_x])
            new_triples = get_block_triples(new_block)

            # Check that no triple in old_block that is NOT in new_block has coverage <= 1
            for idx in old_triples:
                if idx not in new_triples and coverage[idx] <= 1:
                    return False

            # Apply swap
            for idx in old_triples:
                coverage[idx] -= 1
            for idx in new_triples:
                coverage[idx] += 1
            blocks[bi] = new_block
            block_triples_list[bi] = new_triples
            return True

        # Main local search loop
        improved = True
        while improved and time_remaining() > 2.0 and (time.time() < end_time):
            improved = False

            # Pass 1: Try to delete blocks in random order
            indices = list(range(len(blocks)))
            rng.shuffle(indices)
            for bi in indices:
                if time_remaining() < 2.0 or time.time() >= end_time:
                    break
                if try_delete_block(bi):
                    improved = True

            # Pass 2: If no deletion worked, try swaps to create redundancy
            if not improved and len(blocks) > 0:
                for _ in range(100):
                    if time_remaining() < 2.0 or time.time() >= end_time:
                        break
                    bi = rng.randint(0, len(blocks) - 1)
                    old_x = rng.choice(blocks[bi])
                    new_x = rng.randint(0, v - 1)
                    if new_x == old_x or new_x in blocks[bi]:
                        continue
                    if try_swap_point(bi, old_x, new_x):
                        # After swap, try to delete some blocks
                        indices2 = list(range(len(blocks)))
                        rng.shuffle(indices2)
                        for bi2 in indices2:
                            if try_delete_block(bi2):
                                improved = True
                                break

        return blocks

    def validate(blocks):
        """Check if blocks cover all t-subsets."""
        covered = set()
        for block in blocks:
            for ts in itertools.combinations(block, t):
                idx = t_subset_index.get(ts, -1)
                if idx != -1:
                    covered.add(idx)
        return len(covered) == n_t_subsets

    # Main search: try multiple seeds and keep the best
    best_blocks = None
    best_len = float('inf')

    # Phase 1: Structured greedy (deterministic)
    if time_remaining() > 10.0:
        blocks = greedy_construct(seed=0, structured=True)
        if validate(blocks):
            best_blocks = blocks
            best_len = len(blocks)

    # Phase 2: Randomized greedy restarts
    seed = 1
    while time_remaining() > 10.0 and seed < 15:
        blocks = greedy_construct(seed=seed, structured=False)
        if validate(blocks) and len(blocks) < best_len:
            best_blocks = blocks
            best_len = len(blocks)
        seed += 1

    # Phase 3: Local search on the best solution
    if best_blocks is not None and time_remaining() > 5.0:
        best_blocks = local_search(best_blocks, seed=0, max_time=min(30.0, time_remaining() - 5.0))
        best_len = len(best_blocks)

    # Fallback: if no valid solution found, create a trivial one
    if best_blocks is None:
        blocks = []
        seen = set()
        for ts in all_t_subsets:
            remaining = [x for x in range(v) if x not in ts]
            extra = remaining[:k - t]
            block = tuple(sorted(ts + extra))
            if block not in seen:
                seen.add(block)
                blocks.append(list(block))
        best_blocks = blocks
        best_len = len(blocks)

    return best_blocks
