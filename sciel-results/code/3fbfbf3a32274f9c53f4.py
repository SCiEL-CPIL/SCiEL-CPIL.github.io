"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import itertools

DIMENSIONS = [2, 4, 4]

def _build_strassen_2x2():
    """Return the 7 Strassen terms for 2x2 matrix multiplication."""
    return [
        {'U': [(0, 1), (3, 1)], 'V': [(0, 1), (3, 1)], 'W': [(0, 1), (3, 1)]},
        {'U': [(2, 1), (3, 1)], 'V': [(0, 1)], 'W': [(1, 1), (3, -1)]},
        {'U': [(0, 1)], 'V': [(1, 1), (3, -1)], 'W': [(2, 1), (3, 1)]},
        {'U': [(3, 1)], 'V': [(2, 1), (0, -1)], 'W': [(0, 1), (1, 1)]},
        {'U': [(0, 1), (1, 1)], 'V': [(3, 1)], 'W': [(0, -1), (2, 1)]},
        {'U': [(2, 1), (0, -1)], 'V': [(0, 1), (1, 1)], 'W': [(3, 1)]},
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
            r = 0
            if rank < best[0]:
                best = (rank, U.tolist(), V.tolist(), W.tolist())
    
    return best

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
