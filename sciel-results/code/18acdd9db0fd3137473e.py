import numpy as np
import itertools

def construct():
    D = 11
    
    # Generate candidate vectors with norm squared up to 12
    candidates = []
    
    # Helper to add vector
    def add_vec(v):
        candidates.append(v)
        
    # Norm 2: two +/-1
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i] = si
                v[j] = sj
                add_vec(v)
                
    # Norm 4:
    # Type A: one +/-2
    for i in range(D):
        for s in (1, -1):
            v = np.zeros(D, dtype=np.int64)
            v[i] = s * 2
            add_vec(v)
    # Type B: four +/-1
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for pos, sign in zip(positions, signs):
                v[pos] = sign
            add_vec(v)
            
    # Norm 6:
    # Type C: three +/-1
    for positions in itertools.combinations(range(D), 3):
        for signs in itertools.product([1, -1], repeat=3):
            v = np.zeros(D, dtype=np.int64)
            for pos, sign in zip(positions, signs):
                v[pos] = sign
            add_vec(v)
    # Type D: one +/-2, two +/-1
    for i in range(D):
        for si in (1, -1):
            for j, k in itertools.combinations([x for x in range(D) if x != i], 2):
                for sj in (1, -1):
                    for sk in (1, -1):
                        v = np.zeros(D, dtype=np.int64)
                        v[i] = si * 2
                        v[j] = sj
                        v[k] = sk
                        add_vec(v)
                        
    # Norm 8:
    # Type E: four +/-1 (Already in Norm 4? No, 4 ones is Norm 4. Wait. 1^2*4 = 4. Yes.)
    # So Norm 8 Type E: ?
    # 1^2 + 1^2 + 1^2 + 1^2 = 4.
    # To get 8:
    # 2^2 + 2^2 = 8.
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i] = si * 2
                v[j] = sj * 2
                add_vec(v)
    # 2^2 + 1^2 + 1^2 + 1^2 + 1^2 = 4+4=8? No, 4+1+1+1+1=8.
    # One 2, four 1s.
    for i in range(D):
        for si in (1, -1):
            others = [x for x in range(D) if x != i]
            for positions in itertools.combinations(others, 4):
                for signs in itertools.product([1, -1], repeat=4):
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = si * 2
                    for pos, sign in zip(positions, signs):
                        v[pos] = sign
                    add_vec(v)
                    
    # Norm 10:
    # 2^2 + 2^2 + 1^2 + 1^2 = 4+4+1+1=10.
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                others = [x for x in range(D) if x != i and x != j]
                for k, l in itertools.combinations(others, 2):
                    for sk in (1, -1):
                        for sl in (1, -1):
                            v = np.zeros(D, dtype=np.int64)
                            v[i] = si * 2
                            v[j] = sj * 2
                            v[k] = sk
                            v[l] = sl
                            add_vec(v)
    # 2^2 + 1^2 + 1^2 + 1^2 + 1^2 + 1^2 = 4+6=10.
    # One 2, six 1s.
    for i in range(D):
        for si in (1, -1):
            others = [x for x in range(D) if x != i]
            for positions in itertools.combinations(others, 6):
                for signs in itertools.product([1, -1], repeat=6):
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = si * 2
                    for pos, sign in zip(positions, signs):
                        v[pos] = sign
                    add_vec(v)
                    
    # Norm 12:
    # 2^2 + 2^2 + 2^2 = 12.
    for i, j, k in itertools.combinations(range(D), 3):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = si * 2
                    v[j] = sj * 2
                    v[k] = sk * 2
                    add_vec(v)
    # 2^2 + 2^2 + 1^2 + 1^2 + 1^2 = 4+4+3=11.
    # 2^2 + 1^2 * 8 = 4+8=12.
    # One 2, eight 1s.
    for i in range(D):
        for si in (1, -1):
            others = [x for x in range(D) if x != i]
            for positions in itertools.combinations(others, 8):
                for signs in itertools.product([1, -1], repeat=8):
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = si * 2
                    for pos, sign in zip(positions, signs):
                        v[pos] = sign
                    add_vec(v)
                    
    candidates = np.array(candidates, dtype=np.int64)
    n = len(candidates)
    
    norms_sq = np.sum(candidates**2, axis=1)
    
    # Unique norms to test as max_norm_sq
    unique_norms = np.unique(norms_sq)
    
    best_size = 0
    best_selected = []
    
    np.random.seed(42)
    
    # For each possible max norm squared, find max independent set
    for max_norm_sq in unique_norms:
        # Filter candidates with norm <= max_norm_sq
        mask = norms_sq <= max_norm_sq
        idx = np.where(mask)[0]
        if len(idx) == 0: continue
        
        sub_candidates = candidates[idx]
        sub_norms = norms_sq[idx]
        m = len(sub_candidates)
        
        # Compute conflict: dist^2 < max_norm_sq
        dots = sub_candidates @ sub_candidates.T
        dist_sq = sub_norms[:, None] + sub_norms[None, :] - 2 * dots
        conflict = (dist_sq < max_norm_sq)
        np.fill_diagonal(conflict, False)
        
        # Greedy Max Independent Set
        remaining = np.ones(m, dtype=bool)
        degrees = conflict.sum(axis=1).copy()
        selected_local = []
        
        while remaining.any():
            rem_indices = np.where(remaining)[0]
            if len(rem_indices) == 0: break
            
            rem_degrees = degrees[rem_indices]
            min_deg = np.min(rem_degrees)
            min_mask = rem_degrees == min_deg
            min_indices_local = np.where(min_mask)[0]
            
            chosen_local = np.random.choice(min_indices_local)
            chosen_global = rem_indices[chosen_local]
            
            selected_local.append(chosen_global)
            
            neigh_mask = conflict[chosen_global]
            to_remove_mask = np.zeros(m, dtype=bool)
            to_remove_mask[chosen_global] = True
            to_remove_mask[neigh_mask] = True
            
            actual_remove = to_remove_mask & remaining
            
            if np.any(actual_remove):
                deg_decrease = conflict[actual_remove][:, remaining].sum(axis=0)
                degrees[remaining] -= deg_decrease
                
            remaining[actual_remove] = False
        
        if len(selected_local) > best_size:
            best_size = len(selected_local)
            # Map back to global indices
            best_selected = idx[selected_local]
            
    result = candidates[best_selected]
    return result
