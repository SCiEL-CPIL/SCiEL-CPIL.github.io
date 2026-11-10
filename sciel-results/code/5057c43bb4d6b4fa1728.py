import numpy as np
import time

def kissing_number11() -> np.ndarray:
    d = 11
    n_vecs = 1 << d
    # Precompute all vectors
    points_all = np.zeros((n_vecs, d), dtype=np.int64)
    for i in range(n_vecs):
        bits = format(i, f'0{d}b')
        for j in range(d):
            points_all[i, j] = 1 if bits[j] == '1' else -1
            
    best_k = 0
    best_points = []
    
    start_time = time.time()
    timeout = 500 # Safety margin
    
    # Run multiple greedy attempts
    for attempt in range(1000):
        if time.time() - start_time > timeout:
            break
            
        rng = np.random.default_rng(attempt)
        perm = rng.permutation(n_vecs)
        points_shuffled = points_all[perm]
        
        selected = []
        
        for p in points_shuffled:
            valid = True
            # Optimization: vectorized check?
            # For small selected, loop is fine.
            for q in selected:
                # Hamming dist
                # p, q are int64 arrays
                # diff = p - q
                # count non-zero
                if np.sum(p != q) < 3:
                    valid = False
                    break
            if valid:
                selected.append(p)
                
        if len(selected) > best_k:
            best_k = len(selected)
            best_points = selected
            
    if best_k == 0:
        return np.array([np.ones(d), -1*np.ones(d)], dtype=np.int64)
        
    return np.array(best_points, dtype=np.int64)
