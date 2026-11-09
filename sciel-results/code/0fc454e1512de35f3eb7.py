"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]
import numpy as np
import itertools
import time

def _tensor_from_factors(U, V, W):
    """Build the tensor T[i*m+j, j*p+k, k*n+i] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    r = U.shape[1]
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for r_idx in range(r):
        T += np.einsum('a,b,c->abc', U[:, r_idx], V[:, r_idx], W[:, r_idx])
    return T

def _verify(U, V, W, tol=1e-9):
    """Check if the decomposition is exact."""
    n, m, p = DIMENSIONS
    T = _tensor_from_factors(U, V, W)
    Texp = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                Texp[i*m+j, j*p+k, k*n+i] = 1.0
    return np.allclose(T, Texp, atol=tol)

def _try_search_from_seed(U, V, W, max_iters=200, tol=1e-9):
    """
    Try to find a lower-rank decomposition by randomly removing columns
    and then fixing the remaining ones. This is a heuristic search.
    """
    n, m, p = DIMENSIONS
    r = len(U[0])
    best_rank = r
    
    # Start from the current decomposition
    U_arr = [row[:] for row in U]
    V_arr = [row[:] for row in V]
    W_arr = [row[:] for row in W]
    
    # Try to remove columns one by one
    for target_rank in range(r-1, 10, -1):
        if not _verify(U_arr, V_arr, W_arr, tol):
            break
        
        # Try to remove a random column
        removed = False
        for _ in range(50):
            col_to_remove = np.random.randint(target_rank)
            
            U_new = [row[:col_to_remove] + row[col_to_remove+1:] for row in U_arr]
            V_new = [row[:col_to_remove] + row[col_to_remove+1:] for row in V_arr]
            W_new = [row[:col_to_remove] + row[col_to_remove+1:] for row in W_arr]
            
            if _verify(U_new, V_new, W_new, tol):
                U_arr = U_new
                V_arr = V_new
                W_arr = W_new
                best_rank = target_rank - 1
                removed = True
                break
        
        if not removed:
            break
    
    return U_arr, V_arr, W_arr, best_rank

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline, leaving 10s margin
    
    n, m, p = DIMENSIONS  # 2, 4, 5
    rank = 36
    U = [[0] * rank for _ in range(n * m)]  # 8 x 36
    V = [[0] * rank for _ in range(m * p)]  # 20 x 36
    W = [[0] * rank for _ in range(p * n)]  # 10 x 36
    col = 0
    
    strassen_terms = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    
    def add_strassen_block(i1, j1, k1):
        nonlocal col
        for u_pat, v_pat, w_pat in strassen_terms:
            for (di, dj), coeff in u_pat.items():
                U[(i1+di)*m + (j1+dj)][col] += coeff
            for (dj, dk), coeff in v_pat.items():
                V[(j1+dj)*p + (k1+dk)][col] += coeff
            for (di, dk), coeff in w_pat.items():
                W[(k1+dk)*n + (i1+di)][col] += coeff
            col += 1
    
    def add_schoolbook_2x2x1(i1, j1, k1):
        nonlocal col
        for di in range(2):
            for dj in range(2):
                i = i1 + di
                j = j1 + dj
                k = k1
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    
    add_strassen_block(0, 0, 0)
    add_strassen_block(0, 2, 0)
    add_strassen_block(0, 0, 2)
    add_strassen_block(0, 2, 2)
    add_schoolbook_2x2x1(0, 0, 4)
    add_schoolbook_2x2x1(0, 2, 4)
    
    # Verify the base decomposition
    if not _verify(U, V, W):
        # Fallback: return schoolbook if verification fails
        rank = 40
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Try to improve via search
    U_best, V_best, W_best, best_rank = U, V, W, rank
    if time.time() < deadline:
        U_try, V_try, W_try, rank_try = _try_search_from_seed(U, V, W, max_iters=100)
        if rank_try < best_rank and _verify(U_try, V_try, W_try):
            U_best, V_best, W_best, best_rank = U_try, V_try, W_try, rank_try
    
    return {"U": U_best, "V": V_best, "W": W_best}
