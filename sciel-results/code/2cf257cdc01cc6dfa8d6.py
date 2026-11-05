import numpy as np
import itertools
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # 1. Generate all t-subsets and map to indices
    t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {sub: i for i, sub in enumerate(t_subsets)}
    num_t_subsets = len(t_subsets)
    
    # 2. Generate all k-subsets (blocks)
    blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(blocks)
    
    # Precompute block_indices: shape (num_blocks, t_subsets_per_block)
    # t_subsets_per_block = C(k, t) = C(7, 3) = 35
    block_indices = np.empty((num_blocks, 35), dtype=np.int32)
    for i, block in enumerate(blocks):
        for j, t_sub in enumerate(itertools.combinations(block, t)):
            block_indices[i, j] = t_subset_to_idx[t_sub]
            
    # Precompute frequency of each t-subset
    # Count how many blocks cover each t-subset
    freq = np.bincount(block_indices.flatten(), minlength=num_t_subsets)
    
    # Inverse frequency for weighting
    inv_freq = 1.0 / (freq + 1e-9)
    
    uncovered = np.ones(num_t_subsets, dtype=bool)
    selected_blocks_idx = []
    
    # Greedy Phase
    while np.any(uncovered):
        if time.time() > deadline:
            break
            
        w = np.where(uncovered, inv_freq, 0.0)
        gains = np.sum(w[block_indices], axis=1)
        
        best_idx = np.argmax(gains)
        best_gain = gains[best_idx]
        
        if best_gain == 0:
            break
            
        selected_blocks_idx.append(best_idx)
        uncovered[block_indices[best_idx]] = False
        
    # Convert to list of indices for easier manipulation
    selected = set(selected_blocks_idx)
    
    # Maintain coverage counts for redundancy and local search
    # coverage_count[t] = number of selected blocks covering t
    coverage_count = np.zeros(num_t_subsets, dtype=np.int32)
    for idx in selected:
        coverage_count[block_indices[idx]] += 1
        
    # Helper to check if a set of indices forms a valid covering (all counts > 0)
    # We only need to check if min(coverage_count) > 0
    
    # Phase 2: Redundancy Removal
    # Remove any block that is not essential
    while True:
        if time.time() > deadline:
            break
        removed_any = False
        # Iterate over a copy of the list to allow modification
        for idx in list(selected):
            # Temporarily remove idx
            coverage_count[block_indices[idx]] -= 1
            if np.min(coverage_count) > 0:
                # Safe to remove
                selected.remove(idx)
                removed_any = True
                # Do not restore, continue with reduced set
            else:
                # Restore
                coverage_count[block_indices[idx]] += 1
                if time.time() > deadline:
                    break
        if not removed_any:
            break
            
    # Phase 3: 2-for-1 Local Search
    # Try to replace two blocks with one
    # Precompute for each triple, list of blocks covering it
    # triple_to_blocks[t] = list of block indices covering t
    triple_to_blocks = [[] for _ in range(num_t_subsets)]
    for i in range(num_blocks):
        for t_idx in block_indices[i]:
            triple_to_blocks[t_idx].append(i)
            
    improved = True
    while improved:
        improved = False
        if time.time() > deadline:
            break
            
        # Get list of selected blocks
        sel_list = list(selected)
        n_sel = len(sel_list)
        
        for i in range(n_sel):
            if time.time() > deadline:
                break
            for j in range(i + 1, n_sel):
                if time.time() > deadline:
                    break
                    
                idx_i = sel_list[i]
                idx_j = sel_list[j]
                
                # Calculate triples that would become uncovered if we remove idx_i and idx_j
                # A triple t is uncovered if:
                # 1. It is covered by idx_i or idx_j
                # 2. After removal, count becomes 0.
                # This means:
                # - If t is in idx_i XOR idx_j: count must be 1 (only one of them covers it, and no others)
                # - If t is in idx_i AND idx_j: count must be 2 (both cover it, and no others)
                
                # We can iterate through triples covered by idx_i or idx_j
                # Union of indices
                # Since C(7,3)=35, this is small.
                t_set_i = set(block_indices[idx_i])
                t_set_j = set(block_indices[idx_j])
                t_union = t_set_i.union(t_set_j)
                
                critical_triples = []
                for t in t_union:
                    c = coverage_count[t]
                    in_i = t in t_set_i
                    in_j = t in t_set_j
                    # Number of these two blocks covering t
                    n_cover_by_pair = (1 if in_i else 0) + (1 if in_j else 0)
                    
                    # If we remove both, new count is c - n_cover_by_pair
                    # We need new count > 0 to be safe, so if new count == 0, it's critical.
                    if c - n_cover_by_pair == 0:
                        critical_triples.append(t)
                        
                if not critical_triples:
                    # This pair is redundant? Should have been caught in Phase 2 if single, 
                    # but maybe pair redundancy?
                    # If removing both leaves no uncovered, we can remove both.
                    # But we want 2-for-1 or 2-for-0.
                    # If 2-for-0, we just remove both.
                    selected.remove(idx_i)
                    selected.remove(idx_j)
                    coverage_count[block_indices[idx_i]] -= 1
                    coverage_count[block_indices[idx_j]] -= 1
                    improved = True
                    break # Restart loop
                else:
                    # We need a block k not in selected that covers ALL critical_triples
                    # Pick the first critical triple to filter candidates
                    first_t = critical_triples[0]
                    candidates = triple_to_blocks[first_t]
                    
                    for k in candidates:
                        if k in selected:
                            continue
                        # Check if k covers all critical_triples
                        # k covers t if t is in block_indices[k]
                        # We can use a set for block_indices[k] for O(1) lookup, 
                        # but creating sets for every candidate is expensive.
                        # Instead, check if critical_triples is a subset of block_indices[k]
                        # Since critical_triples is small, we can just check each one.
                        
                        # Optimization: Precompute set of triples for each block?
                        # Or just check:
                        valid = True
                        for t in critical_triples:
                            # Check if t is in block_indices[k]
                            # block_indices[k] is a numpy array of 35 ints.
                            # np.in1d or searchsorted?
                            # Since it's small, a simple check might be faster if we convert to set once per k?
                            # But we don't want to convert all blocks to sets.
                            # Let's just use a helper or simple loop.
                            # Actually, we can precompute `block_to_triple_set` for all blocks once.
                            # 346k blocks * 35 ints. Memory is fine.
                            # Let's do that outside the loop if not already done.
                            pass
                            
                        # Let's refactor to precompute block_triple_sets
                        # This will be done in the code below.
                        
                        # For now, placeholder logic
                        # We'll implement the precomputation in the final code.
                        
                        pass
                        
            if improved:
                break
                
    # Convert back to list of blocks
    return [list(blocks[idx]) for idx in selected]
