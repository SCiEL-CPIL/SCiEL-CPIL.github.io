import numpy as np
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    D = 11
    candidates = []
    
    # Generate vectors with 1 to 5 non-zeros, values +/- 1
    for weight in range(1, 6):
        for indices in itertools.combinations(range(D), weight):
            for signs in itertools.product([1, -1], repeat=weight):
                v = np.zeros(D, dtype=np.int64)
                for idx, sig in zip(indices, signs):
                    v[idx] = sig
                candidates.append(v)
                
    candidates = np.array(candidates)
    N = len(candidates)
    norms = np.linalg.norm(candidates, axis=1)
    
    # Sort by norm ascending
    sort_idx = np.argsort(norms)
    candidates = candidates[sort_idx]
    norms = norms[sort_idx]
    
    best_set = []
    best_size = 0
    
    # Try a few random permutations of the sorted list?
    # Or just run greedy on the sorted list.
    # Since sorting by norm is a strong heuristic, we might not need many restarts.
    # But random restarts on the "ties" or slight perturbations might help.
    
    for seed in range(5):
        if time.time() > deadline:
            break
            
        # If seed > 0, add some random noise to the norms to break ties differently?
        # Or just random permutation of the whole set?
        # Random permutation of the whole set ignores the norm ordering.
        # Let's try: Sort by norm, then within same norm, randomize.
        
        # For simplicity, let's just do one run on the sorted list, and a few on random lists.
        # Actually, the norm-sorted greedy is likely the best.
        
        if seed == 0:
            order = np.arange(N)
        else:
            # Randomize
            order = np.random.permutation(N)
            
        selected_indices = []
        selected_vectors = []
        current_max_norm = 0
        
        for idx in order:
            if time.time() > deadline:
                break
                
            v = candidates[idx]
            v_norm = norms[idx]
            
            if not selected_vectors:
                selected_indices.append(idx)
                selected_vectors.append(v)
                current_max_norm = v_norm
                continue
            
            new_max_norm = max(current_max_norm, v_norm)
            
            # Check distances
            # Use numpy for speed
            sel_arr = np.array(selected_vectors)
            dists = np.linalg.norm(sel_arr - v, axis=1)
            
            if np.min(dists) >= new_max_norm - 1e-9:
                selected_indices.append(idx)
                selected_vectors.append(v)
                current_max_norm = new_max_norm
                
        if len(selected_vectors) > best_size:
            best_size = len(selected_vectors)
            best_set = selected_vectors.copy()
            
    if best_set:
        return np.array(best_set)
    else:
        return np.array(candidates[0]).reshape(1, -1)
