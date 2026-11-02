import numpy as np
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 105.0  # 15s safety margin before 120s limit
    
    d = 11
    
    # Generate candidate integer vectors with small coordinates
    # We'll use vectors with coordinates in {-2, -1, 0, 1, 2}
    # This gives us a manageable candidate pool
    
    # Generate all vectors with at most 4 non-zero coordinates (for efficiency)
    # and norm at most 2*sqrt(4) = 4
    candidates = []
    
    # Include simple unit vectors and their combinations
    # First, all vectors with exactly 1 non-zero coordinate
    for i in range(d):
        candidates.append(np.array([0]*i + [1] + [0]*(d-i-1)))
        candidates.append(np.array([0]*i + [-1] + [0]*(d-i-1)))
    
    # Vectors with exactly 2 non-zero coordinates, each in {-1, 1}
    for i in range(d):
        for j in range(i+1, d):
            for a in [-1, 1]:
                for b in [-1, 1]:
                    v = np.zeros(d, dtype=np.int64)
                    v[i] = a
                    v[j] = b
                    candidates.append(v)
    
    # Vectors with 3 non-zero coordinates
    for i in range(d):
        for j in range(i+1, d):
            for k in range(j+1, d):
                for a in [-1, 1]:
                    for b in [-1, 1]:
                        for c in [-1, 1]:
                            v = np.zeros(d, dtype=np.int64)
                            v[i] = a
                            v[j] = b
                            v[k] = c
                            candidates.append(v)
    
    candidates = np.array(candidates, dtype=np.int64)
    
    # Compute norms
    norms = np.linalg.norm(candidates, axis=1)
    
    # Remove zero vectors (shouldn't be any) and sort by norm
    valid_mask = norms > 0
    candidates = candidates[valid_mask]
    norms = norms[valid_mask]
    
    # Sort by norm (smaller first for diversity)
    sort_idx = np.argsort(norms)
    candidates = candidates[sort_idx]
    norms = norms[sort_idx]
    
    # Greedy selection with local search
    selected = []
    selected_norms = []
    
    # First pass: greedy by minimum distance
    for i in range(len(candidates)):
        if time.time() > deadline:
            break
        
        v = candidates[i]
        v_norm = norms[i]
        
        # Check if this vector is compatible with all selected
        compatible = True
        for j, sv in enumerate(selected):
            dist = np.linalg.norm(v - sv)
            if dist < v_norm - 1e-9:  # Need dist >= max norm
                compatible = False
                break
        
        if compatible:
            selected.append(v)
            selected_norms.append(v_norm)
    
    # Convert to array
    if len(selected) == 0:
        return np.array([candidates[0]], dtype=np.int64)
    
    best_set = np.array(selected, dtype=np.int64)
    best_count = len(selected)
    
    # Local search: try to add more vectors
    remaining = [i for i in range(len(candidates)) if candidates[i] not in selected]
    
    # Better local search: iteratively try to add vectors
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_set = list(best_set)
        current_norms = [np.linalg.norm(v) for v in current_set]
        max_norm = max(current_norms) if current_norms else 0
        
        for i in range(len(candidates)):
            if time.time() > deadline:
                break
            
            v = candidates[i]
            v_norm = norms[i]
            
            # Skip if already in set
            if v in current_set:
                continue
            
            # Check compatibility: dist(v, u) >= max(v_norm, u_norm, max_norm)
            # Actually, the constraint is: for all pairs (u, v), dist(u,v) >= max(||u||, ||v||)
            # So we need dist(v, u) >= max(v_norm, u_norm) for all u in set
            # And also v_norm should be considered against max_norm
            
            compatible = True
            for j, u in enumerate(current_set):
                dist = np.linalg.norm(v - u)
                required = max(v_norm, np.linalg.norm(u))
                if dist < required - 1e-9:
                    compatible = False
                    break
            
            if compatible:
                current_set.append(v)
                current_norms.append(v_norm)
                improved = True
                break
    
    if improved:
        best_set = np.array(current_set, dtype=np.int64)
        best_count = len(current_set)
    
    # Further polish: try pairwise swaps
    if time.time() < deadline:
        # Try to improve by swapping one vector for another
        for _ in range(50):  # Limited number of swap attempts
            if time.time() > deadline:
                break
            
            swapped = False
            for i in range(len(best_set)):
                if time.time() > deadline:
                    break
                
                # Remove best_set[i] and see if we can add two better vectors
                remaining_set = [best_set[j] for j in range(len(best_set)) if j != i]
                
                for j in range(len(candidates)):
                    if time.time() > deadline:
                        break
                    
                    v = candidates[j]
                    if v in remaining_set:
                        continue
                    
                    # Check if v can replace best_set[i] and add one more
                    # For simplicity, just try to add v to remaining_set
                    v_norm = norms[j]
                    compatible = True
                    for u in remaining_set:
                        dist = np.linalg.norm(v - u)
                        if dist < max(v_norm, np.linalg.norm(u)) - 1e-9:
                            compatible = False
                            break
                    
                    if compatible:
                        new_set = remaining_set + [v]
                        # Try to add one more
                        for k in range(len(candidates)):
                            w = candidates[k]
                            if w in new_set:
                                continue
                            w_norm = norms[k]
                            compat2 = True
                            for u in new_set:
                                dist2 = np.linalg.norm(w - u)
                                if dist2 < max(w_norm, np.linalg.norm(u)) - 1e-9:
                                    compat2 = False
                                    break
                            if compat2:
                                new_set.append(w)
                                if len(new_set) > len(best_set):
                                    best_set = np.array(new_set, dtype=np.int64)
                                    best_count = len(best_set)
                                swapped = True
                                break
                        if swapped:
                            break
                
                if swapped:
                    break
            
            if not swapped:
                break
    
    return best_set
