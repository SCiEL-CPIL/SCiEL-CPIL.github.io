"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import itertools

DIMENSIONS = [2, 4, 4]

def _build_strassen_2x2():
    """Return the 7 Strassen terms for 2x2 matrix multiplication."""
    return [
        # M1 = (A00+A11)(B00+B11)
        {'U': [(0, 1), (3, 1)], 'V': [(0, 1), (3, 1)], 'W': [(0, 1), (3, 1)]},
        # M2 = (A10+A11)B00
        {'U': [(2, 1), (3, 1)], 'V': [(0, 1)], 'W': [(1, 1), (3, -1)]},
        # M3 = A00(B10-B11)
        {'U': [(0, 1)], 'V': [(1, 1), (3, -1)], 'W': [(2, 1), (3, 1)]},
        # M4 = A11(B01-B00)
        {'U': [(3, 1)], 'V': [(2, 1), (0, -1)], 'W': [(0, 1), (1, 1)]},
        # M5 = (A00+A01)B11
        {'U': [(0, 1), (1, 1)], 'V': [(3, 1)], 'W': [(0, -1), (2, 1)]},
        # M6 = (A10-A00)(B00+B01)
        {'U': [(2, 1), (0, -1)], 'V': [(0, 1), (1, 1)], 'W': [(3, 1)]},
        # M7 = (A01-A11)(B10+B11)
        {'U': [(1, 1), (3, -1)], 'V': [(2, 1), (3, 1)], 'W': [(0, 1)]}
    ]

def _construct_rank28():
    """Construct the baseline rank-28 Strassen-based decomposition."""
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    strassen_terms = _build_strassen_2x2()
    
    blocks = [
        {'j_offset': 0, 'k_offset': 0},
        {'j_offset': 2, 'k_offset': 0},
        {'j_offset': 0, 'k_offset': 2},
        {'j_offset': 2, 'k_offset': 2}
    ]
    
    col = 0
    for block in blocks:
        j_off = block['j_offset']
        k_off = block['k_offset']
        for term in strassen_terms:
            for (idx, coeff) in term['U']:
                i = idx // 2
                j = idx % 2
                global_i = i
                global_j = j + j_off
                U[global_i * m + global_j][col] += coeff
            for (idx, coeff) in term['V']:
                j = idx // 2
                k = idx % 2
                global_j = j + j_off
                global_k = k + k_off
                V[global_j * p + global_k][col] += coeff
            for (idx, coeff) in term['W']:
                k = idx // 2
                i = idx % 2
                global_k = k + k_off
                global_i = i
                W[global_k * n + global_i][col] += coeff
            col += 1
        
    U = U[:col]
    V = V[:col]
    W = W[:col]
    
    U = [[int(x) for x in row] for row in U]
    V = [[int(x) for x in row] for row in V]
    W = [[int(x) for x in row] for row in W]
    
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly represents matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Build the tensor: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # U shape: (n*m, rank), V shape: (m*p, rank), W shape: (p*n, rank)
    # U index: i*m+j, V index: j*p+k, W index: k*n+i
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = 0.0
                for r in range(rank):
                    actual += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                if abs(actual - expected) > 1e-9:
                    return False
    return True

def _is_half_integer(val):
    """Check if value is integer or half-integer."""
    doubled = val * 2
    return abs(doubled - round(doubled)) < 1e-9

def _all_half_integers(U, V, W):
    for matrix in [U, V, W]:
        for row in matrix:
            for val in row:
                if not _is_half_integer(float(val)):
                    return False
    return True

def _try_reduce_rank(U, V, W, n, m, p, max_time=30):
    """Try to reduce rank by zeroing out columns and checking validity."""
    start_time = time.time()
    best = (len(U[0]), U, V, W)
    
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Try removing one column at a time
    for r in range(rank):
        if time.time() - start_time > max_time - 10:
            break
        U_try = np.delete(U, r, axis=1)
        V_try = np.delete(V, r, axis=1)
        W_try = np.delete(W, r, axis=1)
        
        # Check if the remaining decomposition still works
        # This is only valid if column r was redundant
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0
                    actual = 0.0
                    for rr in range(rank - 1):
                        actual += U_try[i*m + j, rr] * V_try[j*p + k, rr] * W_try[k*n + i, rr]
                    if abs(actual - expected) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            U = U_try
            V = V_try
            W = W_try
            rank -= 1
            # Reset to try removing more
            r = 0
            if rank < best[0]:
                best = (rank, U.tolist(), V.tolist(), W.tolist())
    
    return best

def _random_local_search(U, V, W, n, m, p, max_time=60):
    """Random local search to find lower-rank decomposition."""
    start_time = time.time()
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    np.random.seed(42)
    
    # Try to find a rank (rank-1) decomposition by random perturbation
    for attempt in range(200):
        if time.time() - start_time > max_time - 5:
            break
        
        # Try to set one column to zero and adjust others
        r = np.random.randint(rank)
        U_try = U.copy()
        V_try = V.copy()
        W_try = W.copy()
        
        # Zero out column r
        U_try[:, r] = 0
        V_try[:, r] = 0
        W_try[:, r] = 0
        
        # Small perturbation to other columns
        noise_scale = 0.01
        for rr in range(rank):
            if rr != r:
                U_try[:, rr] += np.random.randn(U.shape[0]) * noise_scale
                V_try[:, rr] += np.random.randn(V.shape[0]) * noise_scale
                W_try[:, rr] += np.random.randn(W.shape[0]) * noise_scale
        
        # Check validity
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    actual = np.sum(U_try[i*m+j] * V_try[j*p+k] * W_try[k*n+i])
                    if abs(actual - 1.0) > 0.01:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            # Round to nearest half-integer
            U_try = np.round(U_try * 2) / 2
            V_try = np.round(V_try * 2) / 2
            W_try = np.round(W_try * 2) / 2
            
            # Re-check after rounding
            valid2 = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        actual = np.sum(U_try[i*m+j] * V_try[j*p+k] * W_try[k*n+i])
                        if abs(actual - 1.0) > 1e-6:
                            valid2 = False
                            break
                    if not valid2:
                        break
                if not valid2:
                    break
            
            if valid2:
                new_rank = rank - 1
                if new_rank < best_rank:
                    best_rank = new_rank
                    best_U, best_V, best_W = U_try.copy(), V_try.copy(), W_try.copy()
    
    if best_rank < rank:
        return best_rank, best_U.tolist(), best_V.tolist(), best_W.tolist()
    return rank, U.tolist(), V.tolist(), W.tolist()

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Start with rank-28 baseline
    result = _construct_rank28()
    U, V, W = result['U'], result['V'], result['W']
    
    # Verify baseline
    if not _verify_decomposition(U, V, W, n, m, p):
        raise ValueError("Baseline decomposition failed verification")
    
    best_result = result
    best_rank = len(U[0])
    
    # Try to reduce rank by removing redundant columns
    if time.time() < deadline - 20:
        reduced = _try_reduce_rank(U, V, W, n, m, p, max_time=20)
        if reduced[0] < best_rank:
            best_rank = reduced[0]
            best_result = {'U': reduced[1], 'V': reduced[2], 'W': reduced[3]}
    
    # Try local search
    if time.time() < deadline - 5:
        search_rank, sU, sV, sW = _random_local_search(U, V, W, n, m, p, max_time=30)
        if search_rank < best_rank:
            best_rank = search_rank
            best_result = {'U': sU, 'V': sV, 'W': sW}
    
    # Ensure half-integer constraint
    final_U = best_result['U']
    final_V = best_result['V']
    final_W = best_result['W']
    
    # Round to nearest half-integer
    final_U = [[round(v * 2) / 2 for v in row] for row in final_U]
    final_V = [[round(v * 2) / 2 for v in row] for row in final_V]
    final_W = [[round(v * 2) / 2 for v in row] for row in final_W]
    
    # Final verification
    if _verify_decomposition(final_U, final_V, final_W, n, m, p) and _all_half_integers(final_U, final_V, final_W):
        return {"U": final_U, "V": final_V, "W": final_W}
    
    # Fallback to baseline
    return _construct_rank28()
