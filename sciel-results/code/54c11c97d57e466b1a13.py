import numpy as np
import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    # Safety margin: The evaluator kills at 150s, but the prompt says 120s limit.
    # We will use 115s to be safe.
    deadline = start_time + 115.0
    
    points = list(range(v))
    
    # 1. Precomputation
    # Generate all t-subsets
    t_subsets = list(itertools.combinations(points, t))
    n_t_subsets = len(t_subsets)
    t_subset_to_idx = {sub: i for i, sub in enumerate(t_subsets)}
    
    # Generate all k-blocks
    k_blocks = list(itertools.combinations(points, k))
    n_k_blocks = len(k_blocks)
    
    # Precompute block_cover_indices: for each block, indices of t-subsets it covers
    block_cover_indices = []
    for block in k_blocks:
        subs = list(itertools.combinations(block, t))
        indices = [t_subset_to_idx[sub] for sub in subs]
        block_cover_indices.append(np.array(indices, dtype=np.int32))
        
    # Precompute tsub_to_blocks: for each t-subset, indices of blocks that cover it
    # This is crucial for efficient gain updates
    tsub_to_blocks = [[] for _ in range(n_t_subsets)]
    for b_idx, inds in enumerate(block_cover_indices):
        for s_idx in inds:
            tsub_to_blocks[s_idx].append(b_idx)
            
    # Convert to numpy arrays for faster indexing
    tsub_to_blocks_np = [np.array(lst, dtype=np.int32) for lst in tsub_to_blocks]
    
    # Precompute block_overlap: for each pair of blocks, the number of t-subsets they share
    # This is expensive but enables fast 2-for-1 swap checks
    # For C(19,7,4): ~50k blocks, so 50k^2 / 2 = 1.25 billion pairs - too expensive
    # Instead, we'll compute overlaps on-the-fly using tsub_to_blocks
    
    # Initial gain for each block is C(k, t)
    k_choose_t = len(block_cover_indices[0])
    
    best_blocks = None
    best_len = float('inf')
    
    rng = np.random.default_rng(42)
    
    def local_search(selected_indices, cover_count, deadline):
        """Efficient local search with redundancy removal, 1-for-1 swaps, and 2-for-1 swaps."""
        improved = True
        iterations = 0
        max_iterations = 10000
        
        while improved and time.time() < deadline and iterations < max_iterations:
            improved = False
            iterations += 1
            
            # Step 2a: Redundancy Removal (vectorized)
            # A block is redundant if all t-subsets it covers have count > 1
            # We can check this efficiently by computing min cover count for each block
            redundant_mask = np.zeros(len(selected_indices), dtype=bool)
            for i, b in enumerate(selected_indices):
                if np.min(cover_count[block_cover_indices[b]]) > 1:
                    redundant_mask[i] = True
                    
            if np.any(redundant_mask):
                for i in np.where(redundant_mask)[0]:
                    b = selected_indices[i]
                    selected_indices.pop(i)
                    cover_count[block_cover_indices[b]] -= 1
                    i -= 1  # Adjust index after removal
                improved = True
                continue
                
            # Step 2b: Compute unique counts for each block
            # unique_count[i] = number of t-subsets covered only by block i
            unique_counts = np.zeros(len(selected_indices), dtype=np.int32)
            unique_tsubs = {}
            
            for i, b in enumerate(selected_indices):
                subs = block_cover_indices[b]
                mask = (cover_count[subs] == 1)
                unique_counts[i] = np.sum(mask)
                if unique_counts[i] > 0:
                    unique_tsubs[i] = subs[mask]
                    
            if len(selected_indices) == 0:
                break
                
            min_unique = np.min(unique_counts)
            
            # Step 2c: 1-for-1 Swaps
            # Candidates for removal are those with min_unique
            candidates_out = np.where(unique_counts == min_unique)[0]
            np.random.shuffle(candidates_out)
            
            swap_made = False
            for idx_out in candidates_out:
                b_out = selected_indices[idx_out]
                s_uniq = unique_tsubs.get(idx_out, np.array([], dtype=np.int32))
                
                if len(s_uniq) == 0:
                    continue
                    
                first_s = s_uniq[0]
                candidates_in = tsub_to_blocks_np[first_s]
                
                # Build set of currently selected blocks for fast lookup
                selected_set = set(selected_indices)
                
                valid_candidates = []
                for c in candidates_in:
                    if c in selected_set:
                        continue
                    # Check if c covers all other tsubs in s_uniq
                    covers_all = True
                    for s in s_uniq[1:]:
                        if c not in tsub_to_blocks[s]:
                            covers_all = False
                            break
                    if covers_all:
                        valid_candidates.append(c)
                        
                if valid_candidates:
                    b_in = random.choice(valid_candidates)
                    
                    # Perform swap
                    selected_indices[idx_out] = b_in
                    
                    # Update cover counts
                    cover_count[block_cover_indices[b_out]] -= 1
                    cover_count[block_cover_indices[b_in]] += 1
                    
                    swap_made = True
                    improved = True
                    break # Restart loop after one swap to recalculate unique counts
            
            if swap_made:
                continue
                
            # Step 2d: 2-for-1 Swaps (remove two blocks, add one)
            # This is more complex but can find better solutions
            # For each pair of blocks, check if their unique t-subsets can be covered by a single new block
            n_sel = len(selected_indices)
            if n_sel > 2 and time.time() < deadline - 10:  # Reserve 10s for remaining iterations
                # Sample a subset of pairs to check (checking all pairs is O(n²))
                max_pairs_to_check = 500
                pair_indices = list(itertools.combinations(range(n_sel), 2))
                if len(pair_indices) > max_pairs_to_check:
                    pair_indices = random.sample(pair_indices, max_pairs_to_check)
                
                for i, j in pair_indices:
                    if time.time() > deadline:
                        break
                    b1 = selected_indices[i]
                    b2 = selected_indices[j]
                    
                    # Get unique t-subsets for each
                    subs1 = block_cover_indices[b1]
                    subs2 = block_cover_indices[b2]
                    
                    # Unique to b1: covered by b1 but not by any other selected block
                    # Approximation: t-subsets where cover_count == 1 and in subs1
                    mask1 = (cover_count[subs1] == 1)
                    mask2 = (cover_count[subs2] == 1)
                    
                    uniq1 = subs1[mask1]
                    uniq2 = subs2[mask2]
                    
                    if len(uniq1) == 0 and len(uniq2) == 0:
                        continue
                        
                    # The union of unique t-subsets must be covered by a single new block
                    all_uniq = np.concatenate([uniq1, uniq2]) if len(uniq1) > 0 or len(uniq2) > 0 else np.array([], dtype=np.int32)
                    if len(all_uniq) == 0:
                        continue
                        
                    # Find candidate blocks that cover all of all_uniq
                    # Start with blocks covering the first unique t-subset
                    first_s = all_uniq[0]
                    candidates = tsub_to_blocks_np[first_s]
                    
                    selected_set = set(selected_indices)
                    
                    valid_candidates = []
                    for c in candidates:
                        if c in selected_set:
                            continue
                        # Check if c covers all other tsubs in all_uniq
                        covers_all = True
                        for s in all_uniq[1:]:
                            if c not in tsub_to_blocks[s]:
                                covers_all = False
                                break
                        if covers_all:
                            valid_candidates.append(c)
                            
                    if valid_candidates:
                        b_in = random.choice(valid_candidates)
                        
                        # Remove b1 and b2, add b_in
                        # Remove in reverse order to preserve indices
                        for idx in sorted([i, j], reverse=True):
                            b = selected_indices[idx]
                            selected_indices.pop(idx)
                            cover_count[block_cover_indices[b]] -= 1
                        
                        selected_indices.append(b_in)
                        cover_count[block_cover_indices[b_in]] += 1
                        
                        improved = True
                        break  # Restart local search after modification
            
            if not improved:
                break  # Local minimum reached
                
        return selected_indices
    
    # Multiple restarts to escape local minima
    # We will run greedy + local search multiple times if time permits
    while time.time() < deadline:
        
        # --- Phase 1: Greedy Construction ---
        # gains[b] = number of currently uncovered t-subsets covered by block b
        gains = np.full(n_k_blocks, k_choose_t, dtype=np.int32)
        uncovered = np.ones(n_t_subsets, dtype=bool)
        selected_indices = []
        
        # Optimization: Keep track of uncovered count to stop early if 0
        uncovered_count = n_t_subsets
        
        while uncovered_count > 0:
            if time.time() > deadline:
                break
                
            # Find block with max gain
            # Note: np.argmax is O(N), which is fine for 50k blocks
            b = np.argmax(gains)
            
            if gains[b] == 0:
                # No block covers any uncovered subset? 
                # This shouldn't happen if a cover exists, but break to avoid infinite loop
                break
                
            selected_indices.append(b)
            
            # Update gains and uncovered status
            # Only update for t-subsets that were actually uncovered
            for s in block_cover_indices[b]:
                if uncovered[s]:
                    uncovered[s] = False
                    uncovered_count -= 1
                    # Decrement gain for all blocks covering this t-subset
                    # This is the key optimization: O(num_blocks_covering_s) instead of O(N_blocks)
                    gains[tsub_to_blocks_np[s]] -= 1
                    
        # If we didn't finish greedy due to time, we might have an invalid cover
        # We should check validity later, or just skip this iteration if incomplete
        if uncovered_count > 0:
            continue
            
        # --- Phase 2: Local Search ---
        # Compute cover counts
        cover_count = np.zeros(n_t_subsets, dtype=np.int32)
        for b in selected_indices:
            cover_count[block_cover_indices[b]] += 1
            
        # Run local search
        selected_indices = local_search(selected_indices, cover_count, deadline)
        
        # End Local Search
        
        # Update best
        current_len = len(selected_indices)
        if current_len < best_len:
            best_len = current_len
            best_blocks = [list(k_blocks[i]) for i in selected_indices]
            
        # Check time for next restart
        if time.time() > deadline:
            break
            
    # Fallback if no valid cover was found (should not happen)
    if best_blocks is None:
        # Try a simple greedy without local search, just to return *something*
        # But likely the first iteration succeeded.
        # If not, we return an empty list which is invalid.
        # Let's just return the last selected_indices if it covers everything.
        # For safety, if best_blocks is None, we try one more quick greedy.
        gains = np.full(n_k_blocks, k_choose_t, dtype=np.int32)
        uncovered = np.ones(n_t_subsets, dtype=bool)
        selected_indices = []
        uncovered_count = n_t_subsets
        while uncovered_count > 0 and time.time() < deadline:
            b = np.argmax(gains)
            if gains[b] == 0: break
            selected_indices.append(b)
            for s in block_cover_indices[b]:
                if uncovered[s]:
                    uncovered[s] = False
                    uncovered_count -= 1
                    gains[tsub_to_blocks_np[s]] -= 1
        best_blocks = [list(k_blocks[i]) for i in selected_indices]

    return best_blocks
