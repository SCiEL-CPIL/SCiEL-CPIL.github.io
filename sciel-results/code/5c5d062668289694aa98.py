import numpy as np
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin before the 120s kill
    
    D = 11
    
    # We want to find the maximum number of integer vectors v_1, ..., v_k in Z^D
    # such that for all i != j: ||v_i - v_j|| >= max_i ||v_i||
    
    # Strategy: 
    # 1. Use a pool of "short" vectors (norm <= 2) to keep the max norm small.
    #    If max norm is 2, we need pairwise distance >= 2.
    #    If max norm is sqrt(3), we need pairwise distance >= sqrt(3).
    #    
    # 2. The constraint ||v_i - v_j|| >= R where R = max ||v_k||.
    #    If we restrict to vectors of norm exactly 2 (i.e., 4 non-zeros with +/-1),
    #    then R=2 and we need ||v_i - v_j|| >= 2.
    #    ||v_i - v_j||^2 >= 4.
    #    ||v_i||^2 + ||v_j||^2 - 2 v_i.v_j >= 4
    #    4 + 4 - 2 v_i.v_j >= 4  =>  v_i.v_j <= 2.
    #    So we need a set of vectors with 4 non-zeros (values +/-1) such that 
    #    their pairwise dot products are <= 2.
    #    
    # 3. Alternatively, mix norms. But if we include any vector of norm 2, R=2.
    #    Vectors of norm sqrt(3) (3 non-zeros) can coexist if distance to all norm-2 
    #    vectors is >= 2.
    #    ||v_3 - v_4||^2 >= 4 where v_3 has norm sqrt(3), v_4 has norm 2.
    #    3 + 4 - 2 v_3.v_4 >= 4 => v_3.v_4 <= 1.5 => v_3.v_4 <= 1 (since integer).
    #    
    # 4. Let's focus on the largest pool first: vectors with 4 non-zeros.
    #    Number of such vectors: C(11,4) * 2^4 = 330 * 16 = 5280.
    #    We need to find the largest subset with pairwise dot product <= 2.
    #    
    # 5. We can also include vectors with 3 non-zeros (norm sqrt(3)).
    #    Count: C(11,3) * 2^3 = 165 * 8 = 1320.
    #    Constraint with norm-2 vectors: dot <= 1.
    #    Constraint with other norm-sqrt(3) vectors: 
    #    ||v_a - v_b||^2 >= 4 (since R=2 if any norm-2 vector is present)
    #    3 + 3 - 2 v_a.v_b >= 4 => v_a.v_b <= 1.
    #    
    # 6. If we don't include any norm-2 vectors, R = sqrt(3) (if we include norm-sqrt(3) vectors).
    #    Then we need pairwise distance >= sqrt(3).
    #    For two norm-sqrt(3) vectors: 3+3-2dot >= 3 => dot <= 1.5 => dot <= 1.
    #    For norm-1 vectors (1 nonzero): 1+3-2dot >= 3 => dot <= 0.5 => dot <= 0.
    #    For two norm-1 vectors: 1+1-2dot >= 3 => dot <= -0.5 => dot <= -1.
    #       But dot of two distinct norm-1 vectors is 0 (if different coords) or 1 (same coord, same sign).
    #       So two norm-1 vectors can only coexist if they are orthogonal (dot=0), but 0 >= -0.5 is false.
    #       Wait: condition is dist >= R = sqrt(3). dist^2 >= 3.
    #       1+1-2*0 = 2 < 3. So two orthogonal norm-1 vectors CANNOT coexist if R=sqrt(3).
    #       So if R=sqrt(3), we can have at most 1 norm-1 vector? No, we can't have any two.
    #       Actually, we can have at most 1 vector of norm < sqrt(3) if we want pairwise dist >= sqrt(3)?
    #       Let's check: if we have one norm-1 vector u and one norm-sqrt(3) vector v.
    #       dist^2 = 1 + 3 - 2 u.v >= 3 => u.v <= 0.5 => u.v <= 0.
    #       So u must be orthogonal to v. This is possible.
    #       But we can't have two norm-1 vectors.
    #    
    # 7. It's likely better to have R=2 and pack as many norm-2 and norm-sqrt(3) vectors as possible.
    #    Let's try a greedy approach with random restarts on the combined pool.
    #    Pool: all vectors with 3 or 4 non-zeros.
    #    Total: 1320 + 5280 = 6600 vectors.
    
    # Generate candidates
    candidates = []
    
    # 3-nonzero vectors
    for i, j, k in itertools.combinations(range(D), 3):
        for si in [1, -1]:
            for sj in [1, -1]:
                for sk in [1, -1]:
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    v[k] = sk
                    candidates.append(v)
                    
    # 4-nonzero vectors
    for i, j, k, l in itertools.combinations(range(D), 4):
        for si in [1, -1]:
            for sj in [1, -1]:
                for sk in [1, -1]:
                    for sl in [1, -1]:
                        v = np.zeros(D, dtype=np.int64)
                        v[i] = si
                        v[j] = sj
                        v[k] = sk
                        v[l] = sl
                        candidates.append(v)
                        
    candidates = np.array(candidates)
    N = len(candidates)
    
    # Precompute norms and squared norms
    norms_sq = np.sum(candidates**2, axis=1)
    norms = np.sqrt(norms_sq)
    
    # The maximum norm in our pool is 2 (for 4-nonzero vectors).
    # So if we select any 4-nonzero vector, R = 2.
    # If we select only 3-nonzero vectors, R = sqrt(3).
    
    # Let's assume we will select some 4-nonzero vectors, so R = 2.
    # Constraint: ||v_i - v_j|| >= 2 for all selected i, j.
    # This is equivalent to:
    #   If both are norm 2: dot <= 2
    #   If one is norm 2, one is norm sqrt(3): dot <= 1
    #   If both are norm sqrt(3): dot <= 1
    
    # We can precompute the dot product matrix? 
    # N = 6600. N^2 = 43.5M entries. 43.5M * 8 bytes = 350 MB. 
    # This is feasible but might be slow to compute and store.
    # Let's try to avoid storing the full matrix and compute on the fly, 
    # or use a smarter greedy that batches checks.
    
    # Actually, let's just do a greedy with vectorized checks.
    # We'll try multiple random orderings.
    
    best_set = []
    
    # To speed up, we can sort candidates by some heuristic? 
    # Or just random permutations.
    
    for seed in range(50):
        if time.time() > deadline:
            break
            
        np.random.seed(seed)
        perm = np.random.permutation(N)
        
        selected_indices = []
        selected_vectors = []
        
        for idx in perm:
            if time.time() > deadline:
                break
                
            v = candidates[idx]
            v_norm_sq = norms_sq[idx]
            
            if not selected_vectors:
                selected_indices.append(idx)
                selected_vectors.append(v)
                continue
            
            # Determine R
            # R is the max norm among selected and v.
            # Since we only have norm 2 and sqrt(3), 
            # if v is norm 2 or any selected is norm 2, R=2.
            # Otherwise R=sqrt(3).
            
            # Check constraint against all selected vectors.
            # dist^2 >= R^2
            # ||v - u||^2 = ||v||^2 + ||u||^2 - 2 v.u >= R^2
            
            sel_arr = np.array(selected_vectors)
            dots = sel_arr @ v
            sel_norms_sq = np.sum(sel_arr**2, axis=1)
            
            dists_sq = v_norm_sq + sel_norms_sq - 2 * dots
            
            # R depends on the set.
            # If v is norm 2, R=2.
            # If v is norm sqrt(3), R is 2 if any selected is norm 2, else sqrt(3).
            
            if v_norm_sq == 4:
                R_sq = 4
            else:
                # v is norm sqrt(3)
                # Check if any selected has norm 2
                if np.any(sel_norms_sq == 4):
                    R_sq = 4
                else:
                    R_sq = 3
            
            if np.all(dists_sq >= R_sq - 1e-9):
                selected_indices.append(idx)
                selected_vectors.append(v)
                
        if len(selected_vectors) > len(best_set):
            best_set = selected_vectors.copy()
            
    if best_set:
        return np.array(best_set)
    else:
        return np.array(candidates[0]).reshape(1, -1)
