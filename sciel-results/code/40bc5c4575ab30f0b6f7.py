import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = 140.0  # 10 seconds safety margin
    
    D = 11
    
    # 1. Generate E8 minimal vectors (norm sq 2) and norm sq 4 vectors
    # E8 is in R8.
    # Minimal (norm 2):
    # Type 1: +/-1, +/-1, 0... (112 vectors)
    # Type 2: +/-0.5... (128 vectors) -> Scale by 2 to make integers?
    # If we scale E8 by 2, norm sq becomes 8.
    # We want final norm sq 4.
    # So we should use unscaled E8 for norm 2 parts?
    # But we need integer vectors.
    # E8 Type 2 (half-integers) are not integers.
    # So we cannot use them directly if we require integer output.
    # However, the task says "integer vectors".
    # So we can only use the integer part of E8?
    # E8 integer vectors of norm 2: 112.
    # E8 integer vectors of norm 4:
    # - (2,0...): 16
    # - (1,1,1,1...): 1120
    # Total integer E8 vectors with norm 2 or 4: 112 + 1136 = 1248.
    
    e8_vectors = []
    # Norm 2
    for i, j in itertools.combinations(range(8), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(8, dtype=np.int64)
                v[i] = si
                v[j] = sj
                e8_vectors.append(v)
    # Norm 4
    for i in range(8):
        for s in (1, -1):
            v = np.zeros(8, dtype=np.int64)
            v[i] = s * 2
            e8_vectors.append(v)
            
    for positions in itertools.combinations(range(8), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(8, dtype=np.int64)
            for pos, sign in zip(positions, signs):
                v[pos] = sign
            e8_vectors.append(v)
            
    e8_vectors = np.array(e8_vectors)
    e8_norms_sq = np.sum(e8_vectors**2, axis=1)
    
    # 2. Generate A2 vectors
    # A2 in Z3 sum 0.
    # We will use 3 coords, then drop 1.
    # Norm sq in Z3: a^2+b^2+c^2.
    # We need total norm sq 4.
    # Possible A2 norms: 0, 2, 6...
    # If A2 norm is 2: 6 vectors (perms of 1,-1,0).
    # If A2 norm is 0: 1 vector (0,0,0).
    # If A2 norm is 4? None.
    # If A2 norm is 6? Too big if E8 norm >= 0.
    
    a2_vectors = []
    a2_vectors.append(np.array([0, 0, 0]))
    for i, j, k in itertools.permutations(range(3), 3):
        for si in (1, -1):
            for sj in (1, -1):
                # sum must be 0. si+sj+sk=0 => sk = -si-sj
                # If si=1, sj=1, sk=-2. Norm 1+1+4=6.
                # If si=1, sj=-1, sk=0. Norm 1+1+0=2.
                if si + sj == 0:
                    v = np.zeros(3, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    v[k] = 0
                    a2_vectors.append(v)
                elif si + sj == 2:
                    v = np.zeros(3, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    v[k] = -2
                    a2_vectors.append(v)
                elif si + sj == -2:
                    v = np.zeros(3, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    v[k] = 2
                    a2_vectors.append(v)
                    
    # Remove duplicates
    a2_unique = []
    seen = set()
    for v in a2_vectors:
        t = tuple(v)
        if t not in seen:
            seen.add(t)
            a2_unique.append(v)
    a2_vectors = np.array(a2_unique)
    a2_norms_sq = np.sum(a2_vectors**2, axis=1)
    
    # 3. Generate A1 vectors
    # A1 in Z1.
    # Norms: 0, 1, 4...
    a1_vectors = np.array([[0], [1], [-1], [2], [-2]], dtype=np.int64)
    a1_norms_sq = np.sum(a1_vectors**2, axis=1)
    
    # 4. Combine
    # We want total norm sq = 4.
    # E8 norm + A2 norm + A1 norm = 4.
    
    candidates = []
    
    # Indices for E8 norms
    e8_idx_2 = np.where(e8_norms_sq == 2)[0]
    e8_idx_4 = np.where(e8_norms_sq == 4)[0]
    e8_idx_0 = np.where(e8_norms_sq == 0)[0] # None
    
    a2_idx_0 = np.where(a2_norms_sq == 0)[0]
    a2_idx_2 = np.where(a2_norms_sq == 2)[0]
    
    a1_idx_0 = np.where(a1_norms_sq == 0)[0]
    a1_idx_1 = np.where(a1_norms_sq == 1)[0]
    a1_idx_4 = np.where(a1_norms_sq == 4)[0]
    
    # Case 1: E8(4) + A2(0) + A1(0)
    for i in e8_idx_4:
        for j in a2_idx_0:
            for k in a1_idx_0:
                v = np.concatenate([e8_vectors[i], a2_vectors[j][:2], a1_vectors[k]])
                candidates.append(v)
                
    # Case 2: E8(2) + A2(2) + A1(0)
    for i in e8_idx_2:
        for j in a2_idx_2:
            for k in a1_idx_0:
                v = np.concatenate([e8_vectors[i], a2_vectors[j][:2], a1_vectors[k]])
                candidates.append(v)
                
    # Case 3: E8(2) + A2(0) + A1(2) -> No A1 norm 2.
    
    # Case 4: E8(0) + A2(2) + A1(2) -> No.
    
    # Case 5: E8(2) + A2(1) + A1(1) -> No A2 norm 1.
    
    # Case 6: E8(0) + A2(0) + A1(4)
    for i in e8_idx_0:
        for j in a2_idx_0:
            for k in a1_idx_4:
                v = np.concatenate([e8_vectors[i], a2_vectors[j][:2], a1_vectors[k]])
                candidates.append(v)
                
    # Case 7: E8(4) + A2(0) + A1(0) is covered.
    
    # Note: E8(2) + A2(1) + A1(1) is impossible with integer A2.
    # What about E8(2) + A2(2) + A1(0)? Covered.
    # What about E8(4) + A2(0) + A1(0)? Covered.
    
    # Are there any other combinations?
    # E8(2) + A2(0) + A1(2) No.
    # E8(0) + A2(4) + A1(0) No A2 norm 4.
    
    # So we have candidates from Cases 1, 2, 6.
    
    if not candidates:
        # Fallback
        return np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        
    candidates = np.array(candidates, dtype=np.int64)
    n = len(candidates)
    
    norms_sq = np.sum(candidates**2, axis=1)
    # All should be 4.
    # Verify
    if not np.all(norms_sq == 4):
        # Filter
        mask = norms_sq == 4
        candidates = candidates[mask]
        n = len(candidates)
        norms_sq = norms_sq[mask]
        
    max_norm_sq = 4
    
    dots = candidates @ candidates.T
    dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * dots
    conflict = (dist_sq < max_norm_sq)
    np.fill_diagonal(conflict, False)
    
    best_selected = []
    best_size = 0
    
    np.random.seed(42)
    
    # Greedy Set Packing
    attempt = 0
    while time.time() < deadline:
        remaining = np.ones(n, dtype=bool)
        degrees = conflict.sum(axis=1).copy()
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
            
            neigh_mask = conflict[chosen_global]
            
            to_remove_mask = np.zeros(n, dtype=bool)
            to_remove_mask[chosen_global] = True
            to_remove_mask[neigh_mask] = True
            
            actual_remove = to_remove_mask & remaining
            
            if np.any(actual_remove):
                deg_decrease = conflict[actual_remove][:, remaining].sum(axis=0)
                degrees[remaining] -= deg_decrease
                
            remaining[actual_remove] = False
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_selected = selected
        
        attempt += 1
        
    if len(best_selected) == 0:
        best_selected = [0]
        
    result = candidates[best_selected]
    return result
