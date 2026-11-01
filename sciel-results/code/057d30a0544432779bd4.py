import itertools
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 112.0  # Safety margin

    # 1. Precomputation
    all_tsets = list(itertools.combinations(range(v), t))
    num_tsets = len(all_tsets)
    tset_to_idx = {tset: i for i, tset in enumerate(all_tsets)}

    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)

    t_subsets_per_block = len(list(itertools.combinations(range(k), t)))
    BlockTsets = np.zeros((num_blocks, t_subsets_per_block), dtype=np.int32)

    for i, block in enumerate(all_blocks):
        count = 0
        for sub in itertools.combinations(block, t):
            BlockTsets[i, count] = tset_to_idx[sub]
            count += 1

    # For local search: for each t-set, which blocks cover it
    # Build inverse index: tset_idx -> list of block indices
    tset_to_blocks = [[] for _ in range(num_tsets)]
    for b in range(num_blocks):
        for ts in BlockTsets[b]:
            tset_to_blocks[ts].append(b)

    # 2. Greedy with MRV (Minimum Remaining Values) heuristic
    # At each step, find the rarest uncovered t-set, then pick the block
    # that covers the most uncovered t-sets among those containing that t-set.
    # This is more focused than pure max-gain.

    def greedy_mrv():
        uncovered = np.ones(num_tsets, dtype=bool)
        blocks = []

        while uncovered.any():
            # Find uncovered t-sets
            unc_idx = np.where(uncovered)[0]

            # For each uncovered t-set, count how many candidate blocks (not yet used) cover it
            # We need to find the "rarest" uncovered t-set: the one with fewest available covering blocks
            # But computing this for all uncovered t-sets each iteration is expensive.
            # Approximation: pick a random uncovered t-set, then find best block for it.
            # Better: compute for a sample, or use a faster heuristic.

            # Alternative: compute gain for all blocks, but weight by rarity.
            # gain[b] = sum over uncovered t-sets in block b of (1 / count of blocks covering that t-set)
            # This is expensive. Let's use a simpler approach.

            # Compute standard gains
            gains = uncovered[BlockTsets].sum(axis=1)
            max_gain = gains.max()
            if max_gain == 0:
                break

            # Among blocks with max gain, pick one that covers the rarest t-set
            candidates = np.where(gains == max_gain)[0]

            # For each candidate, compute the "rarity score": min coverage count among its uncovered t-sets
            # Lower rarity score = covers rarer t-sets = better
            best_candidate = candidates[0]
            best_rarity = float('inf')

            for c in candidates:
                # Get uncovered t-sets in this block
                block_ts = BlockTsets[c]
                unc_in_block = block_ts[uncovered[block_ts]]
                if len(unc_in_block) == 0:
                    continue
                # Rarity: minimum number of blocks covering any of these t-sets
                # Use precomputed tset_to_blocks length
                min_cov = min(len(tset_to_blocks[ts]) for ts in unc_in_block)
                if min_cov < best_rarity:
                    best_rarity = min_cov
                    best_candidate = c

            blocks.append(best_candidate)
            uncovered[BlockTsets[best_candidate]] = False

        return blocks

    # 3. Local search: 1-for-1 block swaps
    def local_search(blocks):
        """Try to reduce block count by swapping or removing blocks."""
        blocks = list(blocks)

        # First, remove redundant blocks (as before)
        cover_count = np.zeros(num_tsets, dtype=np.int32)
        for b in blocks:
            cover_count[BlockTsets[b]] += 1

        i = len(blocks) - 1
        while i >= 0:
            b = blocks[i]
            can_remove = True
            for ts in BlockTsets[b]:
                if cover_count[ts] == 1:
                    can_remove = False
                    break
            if can_remove:
                for ts in BlockTsets[b]:
                    cover_count[ts] -= 1
                blocks.pop(i)
            else:
                i -= 1

        # Now try 1-for-1 swaps: replace a block with another that maintains coverage
        # and allows removing another block, or simply reduces some metric.
        # Actually, 1-for-1 swap means: remove block A, add block B, where B covers
        # what A uniquely covered, and possibly allows removing block C.
        # Simpler: try to replace a pair of blocks with one block.
        # Or: try to find a block that, when added, allows removing 2+ existing blocks.

        # Let's try: for each block in the list, see if we can replace it with a different
        # block that covers the same unique t-sets plus more, enabling further removals.
        # This is complex. Instead, let's do a simpler local search:
        # Randomly try to swap one block for another, then re-polish.

        n = len(blocks)
        if n == 0:
            return blocks

        improved = True
        max_iters = 500  # Limit local search iterations
        iter_count = 0

        while improved and iter_count < max_iters:
            improved = False
            iter_count += 1

            # Rebuild cover counts
            cover_count = np.zeros(num_tsets, dtype=np.int32)
            for b in blocks:
                cover_count[BlockTsets[b]] += 1

            # Find blocks that are "critical" (cover some t-set uniquely)
            critical_blocks = set()
            for b in blocks:
                for ts in BlockTsets[b]:
                    if cover_count[ts] == 1:
                        critical_blocks.add(b)
                        break

            # Non-critical blocks can potentially be swapped
            non_critical = [b for b in blocks if b not in critical_blocks]

            if not non_critical:
                break

            # Try swapping a non-critical block with a random other block
            for b_idx in non_critical:
                # Find t-sets that b_idx covers
                b_ts = BlockTsets[b_idx]

                # Find a replacement block that covers all t-sets that b_idx uniquely covers
                # (but b_idx is non-critical, so it has no uniquely covered t-sets)
                # So any block works as a replacement in terms of coverage.
                # The benefit is if the new block helps make another block removable.

                # Pick a random candidate block
                new_b = np.random.randint(num_blocks)
                if new_b == b_idx or new_b in blocks:
                    continue

                # Try the swap
                old_block = b_idx
                blocks[blocks.index(old_block)] = new_b

                # Re-polish
                new_cover_count = np.zeros(num_tsets, dtype=np.int32)
                for b in blocks:
                    new_cover_count[BlockTsets[b]] += 1

                # Check if we can remove any block now
                removed_any = False
                j = len(blocks) - 1
                while j >= 0:
                    b = blocks[j]
                    can_remove = True
                    for ts in BlockTsets[b]:
                        if new_cover_count[ts] == 1:
                            can_remove = False
                            break
                    if can_remove:
                        for ts in BlockTsets[b]:
                            new_cover_count[ts] -= 1
                        blocks.pop(j)
                        removed_any = True
                    else:
                        j -= 1

                if removed_any:
                    improved = True
                    break
                else:
                    # Revert
                    blocks[blocks.index(new_b)] = old_block

            if time.time() > deadline - 5:
                break

        return blocks

    # 4. Main loop: multiple restarts with different random seeds
    best_blocks = None
    best_len = float('inf')

    while time.time() < deadline:
        blocks = greedy_mrv()
        blocks = local_search(blocks)

        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks[:]

        # If we've hit a good result, we can stop early
        if best_len <= 270:
            break

    if best_blocks is None:
        return []

    result = [list(all_blocks[b]) for b in best_blocks]
    return result
