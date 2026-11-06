import itertools
import time
import random
import numpy as np

N = 16
V = N * N

def _precompute():
    """Precompute distance matrix and conflict masks."""
    points = list(itertools.product(range(N), repeat=2))
    pts = np.array(points, dtype=np.int32)
    
    # Compute squared distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    D = np.sum(diff**2, axis=2)
    
    C = [[0] * V for _ in range(V)]
    
    dist_groups = [{} for _ in range(V)]
    for i in range(V):
        for j in range(V):
            if i == j:
                continue
            d = D[i, j]
            if d not in dist_groups[i]:
                dist_groups[i][d] = 0
            dist_groups[i][d] |= (1 << j)
            
    for i in range(V):
        for j in range(i + 1, V):
            mask = 0
            dij = D[i, j]
            
            mask |= dist_groups[i].get(dij, 0)
            mask |= dist_groups[j].get(dij, 0)
            
            for k in range(V):
                if k == i or k == j:
                    continue
                if D[i, k] == D[j, k]:
                    mask |= (1 << k)
            
            mask &= ~(1 << i)
            mask &= ~(1 << j)
            
            C[i][j] = mask
            C[j][i] = mask
            
    return points, D, C

_points, _D, _C = None, None, None

def _init():
    global _points, _D, _C
    if _points is None:
        _points, _D, _C = _precompute()

def _get_blocked_mask(S_mask, C):
    blocked = 0
    indices = []
    m = S_mask
    while m:
        lsb = m & -m
        idx = lsb.bit_length() - 1
        indices.append(idx)
        m ^= lsb
        
    n = len(indices)
    for i in range(n):
        for j in range(i + 1, n):
            blocked |= C[indices[i]][indices[j]]
            
    return blocked

def _is_valid(S_mask, C):
    indices = []
    m = S_mask
    while m:
        lsb = m & -m
        idx = lsb.bit_length() - 1
        indices.append(idx)
        m ^= lsb
        
    n = len(indices)
    for i in range(n):
        for j in range(i + 1, n):
            if S_mask & C[indices[i]][indices[j]]:
                return False
    return True

def _is_maximal(S_mask, C, V):
    blocked = _get_blocked_mask(S_mask, C)
    return (blocked | S_mask) == (1 << V) - 1

def _greedy_build(seed, C, V):
    rng = random.Random(seed)
    S_mask = 0
    U_mask = (1 << V) - 1
    
    while True:
        blocked = _get_blocked_mask(S_mask, C)
        candidates_mask = U_mask & ~blocked & ~S_mask
        
        if not candidates_mask:
            break
            
        candidates = []
        m = candidates_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            candidates.append(idx)
            m ^= lsb
            
        if not candidates:
            break
            
        best_score = -1
        best_candidate = None
        
        rng.shuffle(candidates)
        
        for p in candidates:
            valid = True
            curr_indices = []
            m = S_mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                curr_indices.append(idx)
                m ^= lsb
                
            for idx in curr_indices:
                if S_mask & C[p][idx]:
                    valid = False
                    break
            
            if not valid:
                continue
            
            new_blocked = _get_blocked_mask(S_mask | (1 << p), C)
            score = bin(new_blocked & ~blocked).count('1')
            
            if score > best_score:
                best_score = score
                best_candidate = p
            elif score == best_score and best_candidate is not None:
                if rng.random() < 0.5:
                    best_candidate = p
                    
        if best_candidate is None:
            break
            
        S_mask |= (1 << best_candidate)
        U_mask &= ~(1 << best_candidate)
        
    return S_mask

def _local_search(S_mask, C, V, max_iters=100):
    for _ in range(max_iters):
        improved = False
        indices = []
        m = S_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            indices.append(idx)
            m ^= lsb
            
        n = len(indices)
        
        # 1. Try removing 1 point
        for idx in indices:
            new_mask = S_mask & ~(1 << idx)
            if _is_maximal(new_mask, C, V):
                S_mask = new_mask
                improved = True
                break
        
        if improved:
            continue
            
        # 2. Try removing 2 points
        for i in range(n):
            for j in range(i + 1, n):
                new_mask = S_mask & ~(1 << indices[i]) & ~(1 << indices[j])
                if _is_maximal(new_mask, C, V):
                    S_mask = new_mask
                    improved = True
                    break
            if improved:
                break
        
        if improved:
            continue

        # 3. Try swapping 1 point (remove 1, add 1)
        found_swap = False
        for p in indices:
            base_mask = S_mask & ~(1 << p)
            
            q_candidates = (1 << V) - 1 & ~S_mask
            
            m = q_candidates
            while m:
                lsb = m & -m
                q = lsb.bit_length() - 1
                m ^= lsb
                
                trial_mask = base_mask | (1 << q)
                
                if not _is_valid(trial_mask, C):
                    continue
                    
                if _is_maximal(trial_mask, C, V):
                    S_mask = trial_mask
                    improved = True
                    found_swap = True
                    break
                    
            if found_swap:
                break
                
        if improved:
            continue

        # 4. Try 2-remove, 1-add
        found_swap21 = False
        for i in range(n):
            for j in range(i + 1, n):
                base_mask = S_mask & ~(1 << indices[i]) & ~(1 << indices[j])
                
                blocked_base = _get_blocked_mask(base_mask, C)
                candidates_mask = ((1 << V) - 1) & ~base_mask & ~blocked_base
                
                m_c = candidates_mask
                while m_c:
                    lsb = m_c & -m_c
                    k = lsb.bit_length() - 1
                    m_c ^= lsb
                    
                    trial_mask = base_mask | (1 << k)
                    if _is_maximal(trial_mask, C, V):
                        S_mask = trial_mask
                        found_swap21 = True
                        break
                        
                if found_swap21:
                    break
            if found_swap21:
                break
        
        if found_swap21:
            improved = True
            continue

        if not improved:
            break
            
    return S_mask

def construct():
    _init()
    C = _C
    V = _points and len(_points) or 256
    
    best_size = float('inf')
    best_mask = 0
    start_time = time.time()
    deadline = start_time + 290.0
    
    # Initial Greedy
    S_mask = _greedy_build(12345, C, V)
    S_mask = _local_search(S_mask, C, V, max_iters=50)
    best_size = bin(S_mask).count('1')
    best_mask = S_mask

    # ILS Loop
    rng = random.Random(42)
    while time.time() < deadline:
        # Perturb best_mask
        curr_mask = best_mask
        indices = []
        m = curr_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            indices.append(idx)
            m ^= lsb
            
        if len(indices) >= 3:
            # Remove 2 random distinct points
            i, j = rng.sample(range(len(indices)), 2)
            curr_mask &= ~(1 << indices[i])
            curr_mask &= ~(1 << indices[j])
            
        # Repair: Greedily add points to make it maximal
        while True:
            blocked = _get_blocked_mask(curr_mask, C)
            candidates = ((1 << V) - 1) & ~curr_mask & ~blocked
            if not candidates:
                break
            m_c = candidates
            lsb = m_c & -m_c
            p = lsb.bit_length() - 1
            curr_mask |= (1 << p)
            
        # Run local search
        curr_mask = _local_search(curr_mask, C, V, max_iters=20)
        
        size = bin(curr_mask).count('1')
        if size < best_size:
            best_size = size
            best_mask = curr_mask
            if best_size <= 5:
                break

    # Convert mask to list of points
    result = []
    m = best_mask
    while m:
        lsb = m & -m
        idx = lsb.bit_length() - 1
        result.append(_points[idx])
        m ^= lsb
        
    return result
