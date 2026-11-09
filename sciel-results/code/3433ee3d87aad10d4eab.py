import time
import random
import numpy as np

def _sq_nonzero_count(coefs):
    """Count nonzero coefficients in the exact square using numpy convolution."""
    n = len(coefs)
    if n == 0:
        return 0
    # Trim trailing zeros for speed
    while n > 1 and coefs[n-1] == 0:
        n -= 1
    arr = np.array(coefs[:n], dtype=np.int64)
    sq = np.convolve(arr, arr, mode='full')
    return int(np.count_nonzero(sq))

def _valid(coefs):
    if len(coefs) > 41:
        return False
    nz = 0
    for c in coefs:
        if not isinstance(c, int):
            return False
        if c < -64 or c > 64:
            return False
        if c != 0:
            nz += 1
    if nz != 12:
        return False
    if coefs[-1] == 0:
        return False
    return True

def _shift_to_degree_40(P):
    """Shift P so that its highest degree term is at index 40."""
    # Find last non-zero
    last_nz = -1
    for i in range(len(P)-1, -1, -1):
        if P[i] != 0:
            last_nz = i
            break
    if last_nz == -1:
        return None
        
    if last_nz == 40:
        return P[:]
        
    if last_nz > 40:
        return None
        
    shift = 40 - last_nz
    new_P = [0] * 41
    for i in range(len(P)):
        new_P[i + shift] = P[i]
    return new_P

def _try_square_of_sparse(rng, time_budget=10.0):
    """
    Search for P(x) = Q(x)^2 where Q has 4 terms.
    """
    deadline = time.time() + time_budget
    best_score = 10**9
    best_poly = None
    
    max_deg = 40
    
    for attempt in range(3000):
        if time.time() > deadline:
            break
        
        e0 = 0
        e1 = rng.randint(1, 10)
        e2 = rng.randint(e1+1, e1+15)
        e3 = rng.randint(e2+1, e2+15)
        
        if e3 > max_deg:
            continue
            
        exponents = [e0, e1, e2, e3]
        coeffs = [rng.choice([-1, 1, 2, -2]) for _ in range(4)]
        
        Q = [0] * (max_deg + 1)
        for i in range(4):
            Q[exponents[i]] = coeffs[i]
            
        P = np.convolve(Q, Q, mode='full')
        
        if len(P) > 41:
            continue
            
        P = P.tolist()
        
        nz_count = sum(1 for c in P if c != 0)
        
        if nz_count != 12:
            continue
            
        valid = True
        for c in P:
            if c < -64 or c > 64:
                valid = False
                break
        if not valid:
            continue
            
        if P[-1] == 0:
            continue
            
        P_shifted = _shift_to_degree_40(P)
        if P_shifted is None:
            continue
            
        score = _sq_nonzero_count(P_shifted)
        
        if score < best_score:
            best_score = score
            best_poly = P_shifted[:]
            
    return best_poly, best_score

def _local_search(P_init, rng, time_limit=15.0):
    current = P_init[:]
    current_score = _sq_nonzero_count(current)
    best = current[:]
    best_score = current_score
    deadline = time.time() + time_limit
    
    val_pool = [-8, -6, -4, -3, -2, -1, 1, 2, 3, 4, 6, 8]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        for idx in range(41):
            if time.time() > deadline:
                break
            if current[idx] == 0:
                continue
            old_val = current[idx]
            for new_val in val_pool:
                if new_val == old_val:
                    continue
                current[idx] = new_val
                if not _valid(current):
                    current[idx] = old_val
                    continue
                sc = _sq_nonzero_count(current)
                if sc < best_score:
                    best_score = sc
                    best = current[:]
                    improved = True
                else:
                    current[idx] = old_val
        
        for idx in range(41):
            if time.time() > deadline:
                break
            if current[idx] == 0:
                continue
            old_val = current[idx]
            for new_idx in range(41):
                if time.time() > deadline:
                    break
                if current[new_idx] != 0:
                    continue
                if new_idx == idx:
                    continue
                current[idx] = 0
                current[new_idx] = old_val
                if _valid(current):
                    sc = _sq_nonzero_count(current)
                    if sc < best_score:
                        best_score = sc
                        best = current[:]
                        improved = True
                current[new_idx] = 0
                current[idx] = old_val
        
        nz_idx = [i for i in range(41) if current[i] != 0]
        for i in range(len(nz_idx)):
            if time.time() > deadline:
                break
            for j in range(i+1, len(nz_idx)):
                if time.time() > deadline:
                    break
                idx1, idx2 = nz_idx[i], nz_idx[j]
                v1, v2 = current[idx1], current[idx2]
                current[idx1], current[idx2] = v2, v1
                if _valid(current):
                    sc = _sq_nonzero_count(current)
                    if sc < best_score:
                        best_score = sc
                        best = current[:]
                        improved = True
                current[idx1], current[idx2] = v1, v2
    
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    rng = random.Random(42)
    
    best_score = 10**9
    best_poly = None
    
    if time.time() < deadline - 20:
        poly, score = _try_square_of_sparse(rng, time_budget=20.0)
        if poly is not None and score < best_score:
            best_score = score
            best_poly = poly
    
    if best_poly is None:
        best_poly = [0] * 41
        for i in range(12):
            best_poly[i] = 1
        # Shift to 40
        best_poly = _shift_to_degree_40(best_poly)
        if best_poly is None:
             # Fallback if shift failed (shouldn't happen for degree 11)
             best_poly = [0]*41
             for i in range(12): best_poly[i]=1
             best_poly[40] = 1
             best_poly[11] = 0
        best_score = _sq_nonzero_count(best_poly)
    
    if time.time() < deadline - 5:
        best_poly, best_score = _local_search(best_poly, rng, time_limit=min(40.0, deadline - time.time() - 2))
        
    return best_poly
