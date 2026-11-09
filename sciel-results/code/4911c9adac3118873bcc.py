import numpy as np
import time
import random
import heapq
import itertools

def kissing_number11() -> np.ndarray:
    d = 11
    start_time = time.time()
    deadline = start_time + 590  # Safety margin
    
    best_set_global = []
    best_size_global = 0
    
    # R values to try
    # R=1, 2 are small. R=3, 4, 5 are promising.
    # The number of candidates grows rapidly.
    # R=3: ~1500 candidates.
    # R=4: ~7000 candidates.
    # R=5: ~20000+ candidates.
    # We should prioritize R=3 and R=4.
    
    R_values = [3, 4, 5]
    
    for R in R_values:
        if time.time() > deadline - 30:
            break
            
        # 1. Generate candidates
        candidates = []
        
        # Helper to add vector
        def add_vec(v):
            if np.any(v != 0):
                candidates.append(v)

        # Norm 1
        for i in range(d):
            v = np.zeros(d, dtype=np.int64)
            v[i] = 1
            add_vec(v)
            v[i] = -1
            add_vec(v)
            
        # Norm 2
        for i in range(d):
            for j in range(i + 1, d):
                for s1 in [1, -1]:
                    for s2 in [1, -1]:
                        v = np.zeros(d, dtype=np.int64)
                        v[i] = s1
                        v[j] = s2
                        add_vec(v)
                        
        # Norm 3
        if R >= 3:
            for i in range(d):
                for j in range(i + 1, d):
                    for k in range(j + 1, d):
                        for s1 in [1, -1]:
                            for s2 in [1, -1]:
                                for s3 in [1, -1]:
                                    v = np.zeros(d, dtype=np.int64)
                                    v[i] = s1
                                    v[j] = s2
                                    v[k] = s3
                                    add_vec(v)
                                    
        # Norm 4
        if R >= 4:
            # 2e_i
            for i in range(d):
                v = np.zeros(d, dtype=np.int64)
                v[i] = 2
                add_vec(v)
                v[i] = -2
                add_vec(v)
            # e_i +/- e_j +/- e_k +/- e_l
            for i in range(d):
                for j in range(i + 1, d):
                    for k in range(j + 1, d):
                        for l in range(k + 1, d):
                            for s1 in [1, -1]:
                                for s2 in [1, -1]:
                                    for s3 in [1, -1]:
                                        for s4 in [1, -1]:
                                            v = np.zeros(d, dtype=np.int64)
                                            v[i] = s1
                                            v[j] = s2
                                            v[k] = s3
                                            v[l] = s4
                                            add_vec(v)
                                            
        # Norm 5
        if R >= 5:
            # 2e_i +/- e_j
            for i in range(d):
                for j in range(d):
                    if i == j: continue
                    for s1 in [1, -1]:
                        for s2 in [1, -1]:
                            v = np.zeros(d, dtype=np.int64)
                            v[i] = 2 * s1
                            v[j] = s2
                            add_vec(v)
            # e_i +/- e_j +/- e_k +/- e_l +/- e_m (5 non-zeros)
            # This is too many? C(11,5)*2^5 = 462 * 32 = 14784.
            # Total candidates for R=5 will be ~ 1562 + 5302 + 14784 + 22*10*4 (norm 5 type 2)
            # 22*10*4 = 880.
            # Total ~ 22000.
            # This might be too slow for Python loops in graph construction.
            # We might skip R=5 or optimize.
            # Let's include it but be careful.
            for i in range(d):
                for j in range(i + 1, d):
                    for k in range(j + 1, d):
                        for l in range(k + 1, d):
                            for m in range(l + 1, d):
                                for s1 in [1, -1]:
                                    for s2 in [1, -1]:
                                        for s3 in [1, -1]:
                                            for s4 in [1, -1]:
                                                for s5 in [1, -1]:
                                                    v = np.zeros(d, dtype=np.int64)
                                                    v[i] = s1
                                                    v[j] = s2
                                                    v[k] = s3
                                                    v[l] = s4
                                                    v[m] = s5
                                                    add_vec(v)

        # Unique candidates
        unique_candidates = []
        seen = set()
        for v in candidates:
            key = tuple(v)
            if key not in seen:
                seen.add(key)
                unique_candidates.append(v)
                
        cand_array = np.array(unique_candidates, dtype=np.int64)
        norms_sq = np.sum(cand_array**2, axis=1)
        
        # Filter for current R
        valid_mask = norms_sq <= R
        sub_cands = cand_array[valid_mask]
        sub_norms = norms_sq[valid_mask]
        n_sub = len(sub_cands)
        
        if n_sub < 2:
            continue
            
        # Compute distance matrix
        # dist_sq[i,j] = ||p_i - p_j||^2
        # Vectorized:
        # dots = sub_cands @ sub_cands.T
        # dist_sq = sub_norms[:, None] + sub_norms[None, :] - 2 * dots
        # This creates an n_sub x n_sub matrix.
        # For n_sub=7000, 49M entries. 49M * 8 bytes = 392 MB.
        # This is feasible.
        
        dots = sub_cands @ sub_cands.T
        dist_sq = sub_norms[:, None] + sub_norms[None, :] - 2 * dots
        
        # Conflict if dist_sq < R
        conflict_mask = dist_sq < R
        np.fill_diagonal(conflict_mask, False)
        
        # Convert to list of lists of neighbors for greedy
        # Or keep as boolean matrix for fast neighbor lookup?
        # Boolean matrix is faster for "remove neighbors" if we use numpy slicing?
        # available[neighbors] = False.
        
        deg = conflict_mask.sum(axis=1)
        
        # Heuristic: Greedy MIS with local search
        # We will run multiple trials.
        
        num_trials = 50 if n_sub < 2000 else 10
        
        best_set_local = []
        best_size_local = 0
        
        for trial in range(num_trials):
            if time.time() > deadline - 10:
                break
                
            # Randomize tie-breaking in greedy
            # We can add small random noise to degrees
            deg_noisy = deg + np.random.random(n_sub) * 0.1
            
            # Greedy selection
            # Use a heap for efficiency?
            # heapq requires (priority, index).
            # But we need to remove neighbors.
            # Simple approach:
            # 1. Sort indices by noisy degree.
            # 2. Iterate through sorted list.
            # 3. If node is available, select it, mark it and neighbors unavailable.
            
            sorted_indices = np.argsort(deg_noisy)
            
            available = np.ones(n_sub, dtype=bool)
            selected_indices = []
            
            for idx in sorted_indices:
                if not available[idx]:
                    continue
                    
                # Select idx
                selected_indices.append(idx)
                
                # Remove neighbors
                # conflict_mask[idx] is True for neighbors
                neighbors = np.where(conflict_mask[idx])[0]
                available[neighbors] = False
                available[idx] = False
                
            current_size = len(selected_indices)
            
            # Local Search
            # Try to improve the set
            # 1. Try to add any available point (none should be available by construction of greedy MIS)
            # 2. Try to swap: remove one point, add others.
            
            # This greedy produces a Maximal Independent Set.
            # To get a better one, we can do:
            # While True:
            #   improved = False
            #   for u in selected_indices:
            #       remove u
            #       try to add all neighbors of u that are now available
            #       if size increases, keep
            #       else restore u
            
            # Given time, we might skip complex local search and rely on many greedy trials.
            
            if current_size > best_size_local:
                best_size_local = current_size
                best_set_local = selected_indices[:]
                
        if best_size_local > best_size_global:
            best_size_global = best_size_local
            best_set_global = [sub_cands[i] for i in best_set_local]
            
    # Verify and return
    if best_size_global >= 2:
        result = np.array(best_set_global, dtype=np.int64)
        max_norm = np.max(np.sum(result**2, axis=1))
        valid = True
        for i in range(len(result)):
            for j in range(i + 1, len(result)):
                d2 = np.dot(result[i] - result[j], result[i] - result[j])
                if d2 < max_norm:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            return result
        
    # Fallback
    return np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)

if __name__ == "__main__":
    res = kissing_number11()
    print(f"Size: {len(res)}")
