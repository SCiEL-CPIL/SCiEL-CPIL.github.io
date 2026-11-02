import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10 second margin
    
    # Generate pool of candidates
    # Norm squared 1, 2, 3, 4
    candidates = []
    
    # Norm 1
    for i in range(D):
        for s in [1, -1]:
            v = np.zeros(D, dtype=np.int64)
            v[i] = s
            candidates.append(v)
            
    # Norm 2
    for i, j in itertools.combinations(range(D), 2):
        for si in [1, -1]:
            for sj in [1, -1]:
                v = np.zeros(D, dtype=np.int64)
                v[i] = si
                v[j] = sj
                candidates.append(v)
                
    # Norm 3
    for i, j, k in itertools.combinations(range(D), 3):
        for si in [1, -1]:
            for sj in [1, -1]:
                for sk in [1, -1]:
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    v[k] = sk
                    candidates.append(v)
                    
    # Norm 4 (4 non-zero)
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
                        
    # Norm 4 (2 e_i)
    for i in range(D):
        for s in [2, -2]:
            v = np.zeros(D, dtype=np.int64)
            v[i] = s
            candidates.append(v)

    candidates = np.array(candidates)
    norms_sq = np.sum(candidates**2, axis=1)
    
    # Sort by norm squared
    sort_idx = np.argsort(norms_sq)
    candidates = candidates[sort_idx]
    norms_sq = norms_sq[sort_idx]
    
    best_set = []
    best_norms_sq = []
    
    # Run multiple greedy passes with random perturbations
    # To save time, we do a few passes.
    num_passes = 50
    
    for p in range(num_passes):
        if time.time() > deadline:
            break
            
        # Perturb the order slightly for randomization
        # Keep primary sort by norm, but shuffle within same norm
        # Simple approach: add small random noise to sort key
        noise = np.random.rand(len(candidates)) * 0.1
        keys = norms_sq + noise
        idx = np.argsort(keys)
        
        current_set = []
        current_norms_sq = []
        
        # Preallocate arrays for speed?
        # We'll just use lists and convert at the end
        
        for i in idx:
            if time.time() > deadline:
                break
                
            v = candidates[i]
            v_norm_sq = norms_sq[i]
            
            # Check compatibility
            # If current_set is empty, add
            if len(current_set) == 0:
                current_set.append(v)
                current_norms_sq.append(v_norm_sq)
                continue
                
            # Vectorized check
            # dist_sq = v_norm_sq + current_norms_sq - 2 * (current_set * v).sum(axis=1)
            # cond = dist_sq >= max(v_norm_sq, current_norms_sq)
            
            # Optimization: if v_norm_sq is large, the condition is harder.
            # We can break early if we find an incompatible vector.
            
            # For speed, we can check against a subset first?
            # Or just vectorize.
            
            # current_set is a list of arrays. Convert to array for dot product?
            # This is expensive.
            
            # Alternative: maintain current_set as a numpy array.
            # But it grows.
            
            # Let's try a simpler loop for the inner check, but use numpy for the dot product if possible.
            # Actually, for small D=11, manual loop might be faster than numpy overhead for small vectors?
            # No, numpy is better.
            
            # Let's accumulate in a list and check in batches?
            # No, we need to know if it's compatible to add.
            
            # We'll use a simple loop for the check, as K is small (~300).
            compatible = True
            for u, u_norm_sq in zip(current_set, current_norms_sq):
                # dot = np.dot(u, v)
                # dist_sq = v_norm_sq + u_norm_sq - 2 * dot
                # max_norm_sq = max(v_norm_sq, u_norm_sq)
                # if dist_sq < max_norm_sq:
                #     compatible = False
                #     break
                
                # Unrolled for speed?
                dot = 0
                for k in range(D):
                    dot += u[k] * v[k]
                dist_sq = v_norm_sq + u_norm_sq - 2 * dot
                if dist_sq < (v_norm_sq if v_norm_sq > u_norm_sq else u_norm_sq):
                    compatible = False
                    break
            
            if compatible:
                current_set.append(v)
                current_norms_sq.append(v_norm_sq)
                
        if len(current_set) > len(best_set):
            best_set = current_set
            best_norms_sq = current_norms_sq
            # Optional: local search?
            
    if not best_set:
        # Fallback to seed
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)
        
    return np.array(best_set, dtype=np.int64)
