import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = 140.0  # 10 seconds safety margin before 150s kill
    
    D = 11
    candidates = []
    
    # Type 1: Weight 1, value +/- 1 (norm^2 = 1)
    for i in range(D):
        for s in (1, -1):
            v = np.zeros(D, dtype=np.int64)
            v[i] = s
            candidates.append(v)
            
    # Type 2: Weight 1, value +/- 2 (norm^2 = 4)
    for i in range(D):
        for s in (1, -1):
            v = np.zeros(D, dtype=np.int64)
            v[i] = s * 2
            candidates.append(v)
            
    # Type 3: Weight 2, values +/- 1 (norm^2 = 2)
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i] = si
                v[j] = sj
                candidates.append(v)
                
    # Type 4: Weight 3, values +/- 1 (norm^2 = 3)
    for positions in itertools.combinations(range(D), 3):
        for signs in itertools.product([1, -1], repeat=3):
            v = np.zeros(D, dtype=np.int64)
            for pos, sign in zip(positions, signs):
                v[pos] = sign
            candidates.append(v)
            
    # Type 5: Weight 4, values +/- 1 (norm^2 = 4)
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for pos, sign in zip(positions, signs):
                v[pos] = sign
            candidates.append(v)
            
    candidates = np.array(candidates, dtype=np.int64)
    n = len(candidates)
    
    norms_sq = np.sum(candidates**2, axis=1)
    
    # Compute distance matrix
    dots = candidates @ candidates.T
    dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * dots
    
    # Strict conflict: dist_sq < 4 (needed if any norm^2=4 vector is in set)
    conflict_strict = (dist_sq < 4)
    np.fill_diagonal(conflict_strict, False)
    
    # Loose conflict: dist_sq < 2 (only needed if ALL vectors have norm^2=2)
    conflict_loose = (dist_sq < 2)
    np.fill_diagonal(conflict_loose, False)
    
    # Identify which vectors have norm^2=2 (only Type 3)
    norm2_mask = (norms_sq == 2)
    norm2_indices = np.where(norm2_mask)[0]
    n_norm2 = len(norm2_indices)
    
    best_selected = []
    best_size = 0
    best_result = None
    
    np.random.seed(42)
    
    # Strategy 1: Find large independent set under strict constraint (dist_sq >= 4)
    # This is valid for any mix of vectors
    n_strict_restarts = 50
    for attempt in range(n_strict_restarts):
        if time.time() > deadline - 10:
            break
            
        remaining = np.ones(n, dtype=bool)
        degrees = conflict_strict.sum(axis=1).copy()
        selected = []
        
        while remaining.any():
            if time.time() > deadline - 5:
                break
                
            rem_indices = np.where(remaining)[0]
            if len(rem_indices) == 0:
                break
            
            rem_degrees = degrees[rem_indices]
            min_deg = np.min(rem_degrees)
            min_mask = rem_degrees == min_deg
            min_indices_local = np.where(min_mask)[0]
            
            chosen_local = np.random.choice(min_indices_local)
            chosen_global = rem_indices[chosen_local]
            
            selected.append(chosen_global)
            
            neigh_mask = conflict_strict[chosen_global]
            
            to_remove_mask = np.zeros(n, dtype=bool)
            to_remove_mask[chosen_global] = True
            to_remove_mask[neigh_mask] = True
            
            actual_remove = to_remove_mask & remaining
            
            if np.any(actual_remove):
                deg_decrease = conflict_strict[actual_remove][:, remaining].sum(axis=0)
                degrees[remaining] -= deg_decrease
                
            remaining[actual_remove] = False
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_selected = selected
            best_result = candidates[best_selected]
    
    # Strategy 2: Find large independent set among norm^2=2 vectors under loose constraint (dist_sq >= 2)
    # This is valid ONLY if all selected vectors have norm^2=2
    if n_norm2 > 0 and time.time() < deadline - 15:
        # Extract subgraph for norm^2=2 vectors
        conflict_sub = conflict_loose[np.ix_(norm2_indices, norm2_indices)]
        n_sub = n_norm2
        
        n_loose_restarts = 30
        for attempt in range(n_loose_restarts):
            if time.time() > deadline - 10:
                break
                
            remaining = np.ones(n_sub, dtype=bool)
            degrees = conflict_sub.sum(axis=1).copy()
            selected_sub = []
            
            while remaining.any():
                if time.time() > deadline - 5:
                    break
                    
                rem_indices = np.where(remaining)[0]
                if len(rem_indices) == 0:
                    break
                
                rem_degrees = degrees[rem_indices]
                min_deg = np.min(rem_degrees)
                min_mask = rem_degrees == min_deg
                min_indices_local = np.where(min_mask)[0]
                
                chosen_local = np.random.choice(min_indices_local)
                chosen_global = rem_indices[chosen_local]
                
                selected_sub.append(chosen_global)
                
                neigh_mask = conflict_sub[chosen_global]
                
                to_remove_mask = np.zeros(n_sub, dtype=bool)
                to_remove_mask[chosen_global] = True
                to_remove_mask[neigh_mask] = True
                
                actual_remove = to_remove_mask & remaining
                
                if np.any(actual_remove):
                    deg_decrease = conflict_sub[actual_remove][:, remaining].sum(axis=0)
                    degrees[remaining] -= deg_decrease
                    
                remaining[actual_remove] = False
            
            if len(selected_sub) > best_size:
                best_size = len(selected_sub)
                global_indices = norm2_indices[selected_sub]
                best_selected = global_indices
                best_result = candidates[best_selected]
    
    # Verify the result is valid
    if best_result is not None and len(best_result) > 0:
        result = best_result
        # Final verification
        n_sel = len(result)
        if n_sel >= 2:
            dots_sel = result @ result.T
            norms_sel = np.sum(result**2, axis=1)
            dist_sel = norms_sel[:, None] + norms_sel[None, :] - 2 * dots_sel
            np.fill_diagonal(dist_sel, np.inf)
            min_dist = np.min(dist_sel)
            max_norm = np.max(np.sqrt(norms_sel))
            if min_dist < max_norm - 1e-9:
                # Invalid, fall back to empty or just return as-is (shouldn't happen)
                pass
        return result
    else:
        return np.array([], dtype=np.int64).reshape(0, D)

