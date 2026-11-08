import itertools
import numpy as np
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    # Generate all weight-3 vectors
    w3_candidates = []
    for indices in itertools.combinations(range(D), 3):
        for s1 in (1, -1):
            for s2 in (1, -1):
                for s3 in (1, -1):
                    v = [0] * D
                    v[indices[0]] = s1
                    v[indices[1]] = s2
                    v[indices[2]] = s3
                    w3_candidates.append(v)
                    
    w3_candidates = np.array(w3_candidates, dtype=np.int64)
    n_cand = len(w3_candidates)
    
    # Precompute dot product matrix
    dots = w3_candidates @ w3_candidates.T
    
    # Fallback: Weight 2 set
    w2_candidates = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i] = si
                v[j] = sj
                w2_candidates.append(v)
    w2_set = np.array(w2_candidates, dtype=np.int64)
    best_set = w2_set
    best_size = len(w2_set)
    
    # Greedy selection for w3
    for restart in range(50):
        if time.time() > deadline:
            break
            
        perm = np.random.permutation(n_cand)
        
        selected_indices = []
        
        for idx in perm:
            if time.time() > deadline:
                break
            
            # Check if idx is compatible with all selected_indices
            # dots[idx, selected_indices] <= 1
            if len(selected_indices) == 0:
                compatible = True
            else:
                # Vectorized check
                if np.all(dots[idx, selected_indices] <= 1):
                    compatible = True
                else:
                    compatible = False
                    
            if compatible:
                selected_indices.append(idx)
                
        current_size = len(selected_indices)
        if current_size > best_size:
            best_size = current_size
            best_set = w3_candidates[selected_indices]
            
    return best_set
