import numpy as np
import time
import itertools

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 135
    
    # Generate candidate vectors with small norms
    candidates = []
    
    # Norm 1: single coordinate (22 vectors)
    for i in range(D):
        for s in (1, -1):
            v = [0]*D
            v[i] = s
            candidates.append(v)
    
    # Norm sqrt(2): two coordinates (220 vectors)
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0]*D
                v[i], v[j] = si, sj
                candidates.append(v)
    
    candidates = np.array(candidates, dtype=np.float64)
    n_cands = len(candidates)
    norms = np.linalg.norm(candidates, axis=1)
    
    # Precompute all-pairs compatibility
    dots = candidates @ candidates.T
    dist_sqs = norms[:, None]**2 + norms[None, :]**2 - 2*dots
    max_norm_sqs = np.maximum(norms[:, None]**2, norms[None, :]**2)
    compatible = dist_sqs >= max_norm_sqs - 1e-9
    np.fill_diagonal(compatible, False)
    
    # Greedy with multiple random orders
    best_count = 0
    best_selected = None
    
    np.random.seed(42)
    n_trials = 300
    
    for trial in range(n_trials):
        if time.time() > deadline:
            break
        
        order = np.random.permutation(n_cands)
        selected = []
        
        for idx in order:
            if time.time() > deadline:
                break
            
            if not selected:
                selected.append(idx)
                continue
            
            # Check if compatible with all selected
            if all(compatible[idx, s] for s in selected):
                selected.append(idx)
        
        if len(selected) > best_count:
            best_count = len(selected)
            best_selected = selected[:]
    
    if best_selected is None:
        return np.array([], dtype=np.int64)
    
    result = candidates[best_selected].astype(np.int64)
    return result
