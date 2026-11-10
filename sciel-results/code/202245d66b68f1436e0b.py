import time
import itertools
import random
import numpy as np

def construct(v, k, t):
    start = time.time()
    deadline = start + 118.0  # Safety margin before 120s kill

    # Precompute t-subsets and their indices
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_index = {s: i for i, s in enumerate(all_tsubs)}

    # Precompute all candidate blocks (all k-subsets of v points)
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)

    # Precompute coverage mask for each block as a Python int (bitset over n_tsubs)
    block_masks = [0] * n_blocks
    for bi, block in enumerate(all_blocks):
        mask = 0
        for combo in itertools.combinations(block, t):
            mask |= (1 << tsub_index[combo])
        block_masks[bi] = mask

    # For each t-subset, precompute list of block indices that cover it
    tsub_to_blocks = [[] for _ in range(n_tsubs)]
    for bi, block in enumerate(all_blocks):
        block_set = set(block)
        for combo in itertools.combinations(block, t):
            tsub_to_blocks[tsub_index[combo]].append(bi)

    best_blocks = []
    best_count = float('inf')

    def compute_cov_counts(blocks):
        cov_count = np.zeros(n_tsubs, dtype=np.int32)
        for bi in blocks:
            mask = block_masks[bi]
            m = mask
            while m:
                lb = m & (-m)
                idx = lb.bit_length() - 1
                cov_count[idx] += 1
                m ^= lb
        return cov_count

    def local_search(block_indices, time_limit):
        """Remove redundant blocks and perform 1-for-1 swaps to escape local minima."""
        blocks = list(block_indices)
        if not blocks:
            return blocks

        improved = True
        while improved and time.time() < time_limit:
            improved = False
            
            # 1. Recompute coverage counts
            cov_count = compute_cov_counts(blocks)
            
            # 2. Remove redundant blocks
            # Identify blocks that can be removed
            redundant_blocks = []
            for i in range(len(blocks) - 1, -1, -1):
                if time.time() > time_limit:
                    break
                bi = blocks[i]
                mask = block_masks[bi]
                m = mask
                redundant = True
                while m:
                    lb = m & (-m)
                    idx = lb.bit_length() - 1
                    if cov_count[idx] <= 1:
                        redundant = False
                        break
                    m ^= lb
                if redundant:
                    redundant_blocks.append(i)
            
            if redundant_blocks:
                # Remove one (or all) redundant blocks
                # Removing all at once might be safe if we recompute
                for i in sorted(redundant_blocks, reverse=True):
                    blocks.pop(i)
                improved = True
                continue # Loop to recompute counts
            
            # 3. Try 1-for-1 swaps
            # Find a block with unique t-subsets
            # To be efficient, we can iterate blocks and check if they have any unique t-subsets
            # If a block has no unique t-subsets, it's redundant (handled above).
            # So all remaining blocks have >= 1 unique t-subset.
            
            swap_found = False
            for i in range(len(blocks)):
                if time.time() > time_limit:
                    break
                bi = blocks[i]
                mask = block_masks[bi]
                
                # Find unique t-subsets for this block
                unique_mask = 0
                m = mask
                while m:
                    lb = m & (-m)
                    idx = lb.bit_length() - 1
                    if cov_count[idx] == 1:
                        unique_mask |= lb
                    m ^= lb
                
                if unique_mask == 0:
                    # Should have been caught in redundancy check, but just in case
                    blocks.pop(i)
                    improved = True
                    swap_found = True
                    break
                
                # Try to find a block j not in blocks that covers unique_mask
                # We can sample some candidates from tsub_to_blocks of one of the unique t-subsets
                # Or iterate all blocks? n_blocks is 346k. Too slow to iterate all for each swap.
                # Strategy: Pick one unique t-subset from bi, get candidates from tsub_to_blocks, 
                # and check if they cover the entire unique_mask.
                
                # Pick a random unique t-subset to guide candidate selection
                # Get index of one set bit in unique_mask
                um = unique_mask
                lb = um & (-um)
                target_idx = lb.bit_length() - 1
                
                candidates = tsub_to_blocks[target_idx]
                # Shuffle candidates to avoid bias
                random.shuffle(candidates)
                
                for bj in candidates:
                    if bj in blocks:
                        continue
                    
                    # Check if bj covers all unique t-subsets of bi
                    # (block_masks[bj] & unique_mask) == unique_mask
                    if (block_masks[bj] & unique_mask) == unique_mask:
                        # Perform swap
                        blocks[i] = bj
                        improved = True
                        swap_found = True
                        break
                
                if swap_found:
                    break
            
        return blocks

    # Greedy with bitsets: pick block with max marginal gain
    def greedy_restart(seed, time_limit):
        random.seed(seed)
        uncovered_mask = (1 << n_tsubs) - 1  # all bits set
        chosen = []

        while uncovered_mask != 0:
            if time.time() > time_limit:
                break

            # Pick a random uncovered t-subset to guide
            m = uncovered_mask
            unc_indices = []
            mm = uncovered_mask
            while mm:
                lb = mm & (-mm)
                unc_indices.append(lb.bit_length() - 1)
                mm ^= lb

            if not unc_indices:
                break

            crit_idx = unc_indices[random.randint(0, len(unc_indices) - 1)]

            # Get all blocks covering this t-subset
            candidates = tsub_to_blocks[crit_idx]
            if not candidates:
                break

            # Find candidate with max gain (number of uncovered t-subsets it covers)
            best_bi = -1
            best_gain = -1
            for bi in candidates:
                gain = (block_masks[bi] & uncovered_mask).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_bi = bi

            if best_bi < 0:
                break

            chosen.append(best_bi)
            uncovered_mask &= ~block_masks[best_bi]

        return chosen

    # Run multiple restarts
    n_restarts = 200
    for restart in range(n_restarts):
        if time.time() > deadline - 15.0:
            break

        # Greedy phase
        chosen = greedy_restart(restart, deadline - 15.0)
        if not chosen:
            continue

        # Local search phase
        time_limit_ls = deadline - 5.0
        cleaned = local_search(chosen, time_limit_ls)

        if len(cleaned) < best_count:
            best_count = len(cleaned)
            best_blocks = [list(all_blocks[bi]) for bi in cleaned]

    # If no blocks found, fallback: return empty (shouldn't happen)
    if not best_blocks:
        # Try one more greedy with fixed seed
        chosen = greedy_restart(0, deadline - 5.0)
        if chosen:
            cleaned = local_search(chosen, deadline - 2.0)
            best_blocks = [list(all_blocks[bi]) for bi in cleaned]

    return best_blocks
