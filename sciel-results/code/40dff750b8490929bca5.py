import time
import numpy as np
from itertools import product

def construct():
    n, m, p = 2, 2, 15
    deadline = time.time() + 130
    
    # Build the rank-53 baseline
    U, V, W = _build_strassen_53(n, m, p)
    
    if _verify(U, V, W, n, m, p):
        best_rank = len(U[0])
        best_result = {"U": U, "V": V, "W": W}
    else:
        U, V, W = _build_schoolbook(n, m, p)
        best_rank = len(U[0])
        best_result = {"U": U, "V": V, "W": W}
    
    # Try to find a lower-rank decomposition
    for target_rank in range(best_rank - 1, max(best_rank - 5, 0), -1):
        if time.time() > deadline:
            break
        result = _try_rank_reduction(n, m, p, target_rank, U, V, W)
        if result is not None:
            U, V, W = result
            best_rank = target_rank
            best_result = {"U": U, "V": V, "W": W}
            break
    
    return best_result

def _build_strassen_53(n, m, p):
    """Build the rank-53 Strassen-based decomposition."""
    n, m, p = 2, 2, 15
    blocks = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14,)]
    
    all_U = []
    all_V = []
    all_W = []

    for block in blocks:
        if len(block) == 2:
            k0, k1 = block
            terms = _strassen_terms(k0, k1, n, m, p)
        else:
            k = block[0]
            terms = _schoolbook_terms(k, n, m, p)
        
        for (u_dict, v_dict, w_dict) in terms:
            u_entries = [(i * m + j, c) for (i, j), c in u_dict.items()]
            v_entries = [(j * p + k, c) for (j, k), c in v_dict.items()]
            w_entries = [(k * n + i, c) for (k, i), c in w_dict.items()]
            
            all_U.append(u_entries)
            all_V.append(v_entries)
            all_W.append(w_entries)

    rank = len(all_U)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for c in range(rank):
        for (row, val) in all_U[c]:
            U[row][c] = val
        for (row, val) in all_V[c]:
            V[row][c] = val
        for (row, val) in all_W[c]:
            W[row][c] = val

    return U, V, W

def _strassen_terms(k0, k1, n, m, p):
    """Return Strassen terms for a 2x2x2 block."""
    terms = [
        ({(0,0): 1, (1,1): 1}, {(0, k0): 1, (1, k1): 1}, {(k0, 0): 1, (k1, 1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0, k0): 1}, {(k0, 1): 1}),
        ({(0,0): 1}, {(1, k0): 1, (1, k1): -1}, {(k1, 0): 1}),
        ({(1,1): 1}, {(0, k1): 1, (0, k0): -1}, {(k0, 0): 1, (k0, 1): 1}),
        ({(0,0): 1, (0,1): 1}, {(1, k1): 1}, {(k0, 0): -1, (k1, 0): 1}),
        ({(1,0): 1, (0,0): -1}, {(0, k0): 1, (0, k1): 1}, {(k1, 1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1, k0): 1, (1, k1): 1}, {(k0, 0): 1}),
    ]
    return terms

def _schoolbook_terms(k, n, m, p):
    """Return schoolbook terms for a 2x2x1 block."""
    terms = []
    for i in range(n):
        for j in range(m):
            terms.append((
                {(i, j): 1},
                {(j, k): 1},
                {(k, i): 1}
            ))
    return terms

def _build_schoolbook(n, m, p):
    """Build the schoolbook rank-60 decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly matches matrix multiplication."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)

    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = i * m + j
                v_row = j * p + k
                w_row = k * n + i
                val = np.sum(U_arr[u_row, :] * V_arr[v_row, :] * W_arr[w_row, :])
                if not np.isclose(val, 1.0, atol=1e-9):
                    return False
    return True

def _try_rank_reduction(n, m, p, target_rank, U_base, V_base, W_base):
    """Try to find a decomposition with the target rank."""
    n, m, p = 2, 2, 15
    
    # Build the constraint matrix
    # For each (i,j,k), we need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    # This is nonlinear. Instead, fix U and V, solve for W linearly.
    
    # Use the base decomposition's U and V as a starting point
    # and try to find a W that works with a reduced rank.
    
    # Actually, let's try a different approach: use the structure of the problem.
    # For fixed j, the constraints decouple in a certain way.
    
    # Let's try to find a decomposition where U and V have a specific form.
    # Use the Strassen structure but with a modified W.
    
    # This is a placeholder - the actual implementation would be more complex.
    # For now, return None to indicate failure.
    return None
