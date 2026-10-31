import numpy as np
import itertools
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # 1. Precomputation
    points = list(range(v))
    
    # Generate all t-subsets and map to indices
    t_subsets = list(itertools.combinations(points, t))
    n_t_subsets = len(t_subsets)
    t_subset_to_idx = {sub: i for i, sub in enumerate(t_subsets)}
    
    # Generate all k-blocks
    k_blocks = list(itertools.combinations(points, k))
    n_k_blocks = len(k_blocks)
    
    # Build incidence matrix M (n_k_blocks x n_t_subsets)
    # M[b, s] = 1 if t-subset s is contained in k-block b
    M = np.zeros((n_k_blocks, n_t_subsets), dtype=np.int8)
    
    # Pre-compute for each block, the indices of t-subsets it covers
    # This is useful for local search
    block_cover_indices = []
    for b, block in enumerate(k_blocks):
        subs = list(itertools.combinations(block, t))
        indices = np.array([t_subset_to_idx[sub] for sub in subs], dtype=np.int32)
        block_cover_indices.append(indices)
        M[b, indices] = 1
        
    # 2. Helper function: Local Search (Redundancy Removal + Swap)
    def local_search(selected_indices, max_swaps=50):
        if not selected_indices:
            return selected_indices
        
        # Compute counts for each t-subset
        counts = np.zeros(n_t_subsets, dtype=np.int32)
        for b_idx in selected_indices:
            counts[block_cover_indices[b_idx]] += 1
            
        redundant_found = True
        while redundant_found:
            redundant_found = False
            new_selected = []
            for b_idx in selected_indices:
                subs = block_cover_indices[b_idx]
                # Check if all covered subsets have count > 1
                # If min count > 1, block is redundant
                if np.min(counts[subs]) > 1:
                    # Remove block
                    counts[subs] -= 1
                    redundant_found = True
                else:
                    new_selected.append(b_idx)
            selected_indices = new_selected
            if time.time() > deadline:
                break
                
        # Now try swap-based improvement: try to replace one block with another
        # that covers the same "critical" subsets but allows further removal
        improved = True
        swap_attempts = 0
        while improved and swap_attempts < max_swaps and time.time() < deadline:
            improved = False
            swap_attempts += 1
            n_sel = len(selected_indices)
            if n_sel == 0:
                break
            
            # Recompute counts
            counts = np.zeros(n_t_subsets, dtype=np.int32)
            for b_idx in selected_indices:
                counts[block_cover_indices[b_idx]] += 1
            
            # Find blocks that are "critical" (have at least one t-subset with count == 1)
            # For each critical block, try to find a replacement that also covers those critical subsets
            # and allows the original block to be removed
            for i, b_idx in enumerate(list(selected_indices)):
                if time.time() > deadline:
                    break
                subs_orig = block_cover_indices[b_idx]
                # Find t-subsets that are only covered by this block
                critical_subs = subs_orig[counts[subs_orig] == 1]
                if len(critical_subs) == 0:
                    # This block is redundant, should have been caught above
                    continue
                
                # Try to find a block that covers all critical_subs
                # and is not already in selected_indices
                # For efficiency, only check a subset of candidate blocks
                # Score candidate blocks by how many of the critical subs they cover
                # and prefer blocks that are already "close" to being useful
                
                # Limit search to speed up
                candidates = []
                for c_idx in range(n_k_blocks):
                    if c_idx in selected_indices:
                        continue
                    subs_cand = block_cover_indices[c_idx]
                    # Check if candidate covers all critical subs
                    # Use set intersection for speed
                    if len(np.intersect1d(critical_subs, subs_cand, assume_unique=True)) == len(critical_subs):
                        # Count how many total t-subsets this candidate covers that are currently uncovered or under-covered
                        # Actually, we want to minimize the number of new t-subsets that become uncovered
                        # But since we're adding a block, no t-subset becomes uncovered
                        # The benefit is if we can then remove b_idx
                        # So we just need to check if after adding c_idx and removing b_idx, all t-subsets are still covered
                        candidates.append(c_idx)
                
                if candidates:
                    # Pick the best candidate: one that maximizes the number of t-subsets with count > 1
                    # (i.e., creates more redundancy, making further removals possible)
                    best_cand = None
                    best_score = -1
                    for c_idx in candidates[:100]:  # Limit to 100 for speed
                        subs_cand = block_cover_indices[c_idx]
                        # Score: number of t-subsets where count > 1 after adding c_idx and removing b_idx
                        # Approximate: count of t-subsets in subs_cand that have count > 1 currently
                        score = np.sum(counts[subs_cand] > 1)
                        if score > best_score:
                            best_score = score
                            best_cand = c_idx
                    
                    if best_cand is not None:
                        # Perform the swap
                        new_selected = [b for b in selected_indices if b != b_idx]
                        new_selected.append(best_cand)
                        
                        # Verify validity
                        check_counts = np.zeros(n_t_subsets, dtype=np.int32)
                        for b_idx2 in new_selected:
                            check_counts[block_cover_indices[b_idx2]] += 1
                        if np.all(check_counts > 0):
                            selected_indices = new_selected
                            improved = True
                            break  # Restart the swap search with new state
        
        return selected_indices

    # 3. Helper function: Greedy Construction (optimized)
    def greedy_construction(rng, randomized=False):
        uncovered_mask = np.ones(n_t_subsets, dtype=np.int8)
        selected_indices = []
        
        while np.any(uncovered_mask):
            if time.time() > deadline:
                break
                
            # Calculate gains using matrix-vector product
            gains = M @ uncovered_mask
            
            # Find best block
            if not randomized:
                best_block = np.argmax(gains)
                if gains[best_block] == 0:
                    break
                selected_indices.append(best_block)
                M_row = M[best_block]
                uncovered_mask[M_row == 1] = 0
            else:
                # Randomized: pick from top candidates with probability proportional to gain
                valid_blocks = np.where(gains > 0)[0]
                if len(valid_blocks) == 0:
                    break
                
                # Get top K candidates
                k_cand = min(20, len(valid_blocks))
                if len(valid_blocks) > k_cand:
                    # Get top k_cand by gain
                    top_indices = np.argpartition(gains, -k_cand)[-k_cand:]
                    # Filter to valid
                    top_valid = [i for i in top_indices if gains[i] > 0]
                else:
                    top_valid = list(valid_blocks)
                
                # Weighted random selection based on gain
                weights = np.array([gains[i] for i in top_valid], dtype=np.float64)
                weights /= weights.sum()
                best_block = top_valid[rng.choice(len(top_valid), p=weights)]
                
                selected_indices.append(best_block)
                M_row = M[best_block]
                uncovered_mask[M_row == 1] = 0
                
        return selected_indices

    # 4. Main Search Loop
    best_blocks = None
    best_len = float('inf')
    
    rng = np.random.default_rng(42)
    
    iteration = 0
    while time.time() < deadline:
        iteration += 1
        
        # Alternate between different strategies
        strategy = iteration % 3
        
        if strategy == 0:
            # Pure Greedy
            selected_indices = greedy_construction(rng, randomized=False)
        elif strategy == 1:
            # Randomized Greedy
            selected_indices = greedy_construction(rng, randomized=True)
        else:
            # Greedy with different random seed
            rng_iter = np.random.default_rng(iteration * 12345)
            selected_indices = greedy_construction(rng_iter, randomized=True)
        
        # Apply Local Search
        if selected_indices:
            refined_indices = local_search(selected_indices)
            
            # Verify validity: every t-subset must be covered
            is_valid = True
            if refined_indices:
                check_counts = np.zeros(n_t_subsets, dtype=np.int32)
                for b_idx in refined_indices:
                    check_counts[block_cover_indices[b_idx]] += 1
                if np.any(check_counts == 0):
                    is_valid = False
            else:
                is_valid = (n_t_subsets == 0)

            if is_valid and len(refined_indices) < best_len:
                best_len = len(refined_indices)
                best_blocks = [list(k_blocks[i]) for i in refined_indices]

    # Fallback if no blocks found (should not happen)
    if best_blocks is None:
        selected_indices = greedy_construction(rng, randomized=False)
        if selected_indices:
            refined_indices = local_search(selected_indices)
            check_counts = np.zeros(n_t_subsets, dtype=np.int32)
            for b_idx in refined_indices:
                check_counts[block_cover_indices[b_idx]] += 1
            if np.all(check_counts > 0):
                best_blocks = [list(k_blocks[i]) for i in refined_indices]
            else:
                best_blocks = [list(k_blocks[i]) for i in refined_indices]
        else:
            best_blocks = []

    return best_blocks
