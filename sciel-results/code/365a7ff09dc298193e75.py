import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    # 1. Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    all_k_subsets = list(itertools.combinations(range(v), k))
    num_k_subsets = len(all_k_subsets)
    
    # Create coverage matrix: M[b, t] = 1 if block b covers t-subset t
    # Using uint8 or bool
    M = np.zeros((num_k_subsets, num_t_subsets), dtype=bool)
    
    for b_idx, block in enumerate(all_k_subsets):
        for s in itertools.combinations(block, t):
            t_idx = t_subset_to_idx[s]
            M[b_idx, t_idx] = True
            
    # Map each t-subset to the blocks that cover it (for local search)
    t_subset_blocks = []
    for t_idx in range(num_t_subsets):
        blocks = np.where(M[:, t_idx])[0]
        t_subset_blocks.append(blocks)

    # 2. Phase 1: Greedy Packing (Approximate MWIS)
    # We keep the original logic but might speed it up?
    # Original logic:
    # pool = set(range(num_k_subsets))
    # core_blocks = []
    # uncovered_mask = np.ones(num_t_subsets, dtype=bool)
    
    # To speed up "select block with most uncovered", we can use M.
    # gain = M[:, uncovered_mask].sum(axis=1)
    
    pool = set(range(num_k_subsets))
    core_blocks = []
    uncovered_mask = np.ones(num_t_subsets, dtype=bool)
    
    while pool:
        # Calculate gains for all blocks in pool
        # Note: M is full, but we only care about pool.
        # For 50k blocks, calculating sum for all is fast in NumPy.
        gains = M[:, uncovered_mask].sum(axis=1)
        
        # Mask out blocks not in pool
        pool_mask = np.zeros(num_k_subsets, dtype=bool)
        for b in pool:
            pool_mask[b] = True
            
        gains[~pool_mask] = -1
        
        best_block = np.argmax(gains)
        max_gain = gains[best_block]
        
        if max_gain <= 0:
            break
            
        core_blocks.append(best_block)
        
        # Update uncovered mask
        uncovered_mask[M[best_block, :]] = False
        
        # Remove conflicts
        # Conflict: shares t points.
        # This is equivalent to: C is in pool if C does NOT share any t-subset with best_block.
        # So, we remove all C that share ANY t-subset with best_block.
        # neighbors_to_remove = union of t_subset_blocks[t_idx] for t_idx in block_t_indices[best_block]
        
        # We can get the t-subsets covered by best_block from M
        covered_ts = np.where(M[best_block, :])[0]
        
        neighbors_to_remove = set()
        for t_idx in covered_ts:
            neighbors_to_remove.update(t_subset_blocks[t_idx])
            
        pool.difference_update(neighbors_to_remove)

    # 3. Phase 2: Solve Remaining Set Cover
    # Use Greedy Set Cover on the rest.
    active_blocks = set(core_blocks)
    cover_count = np.zeros(num_t_subsets, dtype=int)
    
    for b_idx in active_blocks:
        cover_count[M[b_idx, :]] += 1
        
    still_uncovered = np.where(cover_count == 0)[0]
    
    if len(still_uncovered) > 0:
        # Greedy Set Cover for remainder
        # We can use all blocks not in active_blocks
        # Or just all blocks, and check if already active.
        
        while len(still_uncovered) > 0:
            # Calculate gains for remaining blocks
            # Only consider blocks not in active_blocks?
            # If a block is in active_blocks, it covers some still_uncovered?
            # No, if it was in core, it was added. But maybe it covers *new* still_uncovered?
            # No, cover_count is updated. If cover_count[t] > 0, it's not still_uncovered.
            # So blocks in active_blocks cover 0 still_uncovered.
            # So we can restrict to blocks NOT in active_blocks.
            
            candidate_mask = np.ones(num_k_subsets, dtype=bool)
            for b in active_blocks:
                candidate_mask[b] = False
                
            # Gains: sum of M[:, still_uncovered]
            # still_uncovered is a list of indices.
            gains = M[:, still_uncovered].sum(axis=1)
            gains[~candidate_mask] = -1
            
            best_block = np.argmax(gains)
            max_gain = gains[best_block]
            
            if max_gain <= 0:
                break
                
            active_blocks.add(best_block)
            cover_count[M[best_block, :]] += 1
            
            # Update still_uncovered
            # Remove t-subsets covered by best_block
            covered_ts = np.where(M[best_block, :])[0]
            # still_uncovered = set(still_uncovered) - set(covered_ts)
            # Vectorized:
            mask_uncovered = np.ones(num_t_subsets, dtype=bool)
            mask_uncovered[still_uncovered] = False # This is wrong logic.
            
            # Simpler:
            still_uncovered_set = set(still_uncovered)
            for t_idx in covered_ts:
                if t_idx in still_uncovered_set:
                    still_uncovered_set.remove(t_idx)
            still_uncovered = list(still_uncovered_set)

    # 4. Minimization
    def minimize(active_blocks_set, cover_counts):
        changed = True
        while changed:
            changed = False
            items = list(active_blocks_set)
            random.shuffle(items)
            for b_idx in items:
                if b_idx not in active_blocks_set:
                    continue
                can_remove = True
                # Check if any t-subset is uniquely covered
                # cover_counts is a numpy array
                # If min(cover_counts[M[b_idx, :]]) > 1, we can remove
                if np.min(cover_counts[M[b_idx, :]]) > 1:
                    active_blocks_set.remove(b_idx)
                    cover_counts[M[b_idx, :]] -= 1
                    changed = True
        return active_blocks_set, cover_counts

    active_blocks, cover_count = minimize(active_blocks, cover_count)

    # 5. Local Search
    start_time = time.time()
    time_limit = 115.0
    
    # Pre-compute lengths
    t_subset_block_lens = [len(lst) for lst in t_subset_blocks]

    while time.time() - start_time < time_limit:
        improved = False
        current_blocks = list(active_blocks)
        random.shuffle(current_blocks)
        
        for B in current_blocks:
            if B not in active_blocks:
                continue
            
            # Find t-subsets uniquely covered by B
            # cover_count[M[B, :]] == 1
            unique_ts = np.where(cover_count[M[B, :]] == 1)[0]
            
            if len(unique_ts) == 0:
                continue

            # Sort by length of t_subset_blocks
            unique_ts_sorted = sorted(unique_ts, key=lambda x: t_subset_block_lens[x])
            
            candidates = None
            for t_idx in unique_ts_sorted:
                blocks_list = t_subset_blocks[t_idx]
                if candidates is None:
                    candidates = set(blocks_list)
                else:
                    candidates.intersection_update(blocks_list)
                
                if not candidates:
                    break
            
            if not candidates:
                continue
            
            valid_candidates = [c for c in candidates if c != B and c not in active_blocks]
            
            if not valid_candidates:
                continue
            
            B_new = random.choice(valid_candidates)
            
            active_blocks.remove(B)
            active_blocks.add(B_new)
            
            cover_count[M[B, :]] -= 1
            cover_count[M[B_new, :]] += 1
            
            active_blocks, cover_count = minimize(active_blocks, cover_count)
            improved = True
            
        # Do not break if not improved. Keep searching.
        # But to avoid infinite loop of no progress, we can add a counter.
        # For now, just let it run until time limit.

    result = [list(all_k_subsets[b_idx]) for b_idx in active_blocks]
    return result
