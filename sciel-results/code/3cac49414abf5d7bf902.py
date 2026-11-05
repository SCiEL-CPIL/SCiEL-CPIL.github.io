import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [2, 3, 5]
TIME_LIMIT = 140

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that sum_r U[:,r] * V[:,r] * W[:,r] == tensor T where T[i,j,k] = 1 for all i,j,k
    """
    rank = len(U[0]) if len(U) > 0 else 0
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                s = 0.0
                for r in range(rank):
                    s += U_arr[idx_u, r] * V_arr[idx_v, r] * W_arr[idx_w, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _build_rank28():
    """Build the known rank-28 decomposition."""
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    
    # Part 1: A2 B2 (j=2, k=0..4)
    for i in range(n):
        for k in range(p):
            U[i * m + 2][r] = 1
            V[2 * p + k][r] = 1
            W[k * n + i][r] = 1
            r += 1
            
    # Part 2: A1 B13 (j=0,1, k=4)
    for i in range(n):
        for j in range(2):
            U[i * m + j][r] = 1
            V[j * p + 4][r] = 1
            W[4 * n + i][r] = 1
            r += 1
            
    # Part 3: A1 B11 (j=0,1, k=0,1) using Strassen
    # M1
    U[0 * m + 0][r] = 1
    U[1 * m + 1][r] = 1
    V[0 * p + 0][r] = 1
    V[1 * p + 1][r] = 1
    W[0 * n + 0][r] = 1
    W[1 * n + 1][r] = 1
    r += 1
    
    # M2
    U[1 * m + 0][r] = 1
    U[1 * m + 1][r] = 1
    V[0 * p + 0][r] = 1
    W[0 * n + 1][r] = 1
    W[1 * n + 1][r] = -1
    r += 1
    
    # M3
    U[0 * m + 0][r] = 1
    V[0 * p + 1][r] = 1
    V[1 * p + 1][r] = -1
    W[1 * n + 0][r] = 1
    W[1 * n + 1][r] = 1
    r += 1
    
    # M4
    U[1 * m + 1][r] = 1
    V[1 * p + 0][r] = 1
    V[0 * p + 0][r] = -1
    W[0 * n + 0][r] = 1
    W[0 * n + 1][r] = 1
    r += 1
    
    # M5
    U[0 * m + 0][r] = 1
    U[0 * m + 1][r] = 1
    V[1 * p + 1][r] = 1
    W[0 * n + 0][r] = -1
    W[1 * n + 0][r] = 1
    r += 1
    
    # M6
    U[1 * m + 0][r] = 1
    U[0 * m + 0][r] = -1
    V[0 * p + 0][r] = 1
    V[0 * p + 1][r] = 1
    W[1 * n + 1][r] = 1
    r += 1
    
    # M7
    U[0 * m + 1][r] = 1
    U[1 * m + 1][r] = -1
    V[1 * p + 0][r] = 1
    V[1 * p + 1][r] = 1
    W[0 * n + 0][r] = 1
    r += 1
    
    # Part 4: A1 B12 (j=0,1, k=2,3) using Strassen
    # M1
    U[0 * m + 0][r] = 1
    U[1 * m + 1][r] = 1
    V[0 * p + 2][r] = 1
    V[1 * p + 3][r] = 1
    W[2 * n + 0][r] = 1
    W[3 * n + 1][r] = 1
    r += 1
    
    # M2
    U[1 * m + 0][r] = 1
    U[1 * m + 1][r] = 1
    V[0 * p + 2][r] = 1
    W[2 * n + 1][r] = 1
    W[3 * n + 1][r] = -1
    r += 1
    
    # M3
    U[0 * m + 0][r] = 1
    V[0 * p + 3][r] = 1
    V[1 * p + 3][r] = -1
    W[3 * n + 0][r] = 1
    W[3 * n + 1][r] = 1
    r += 1
    
    # M4
    U[1 * m + 1][r] = 1
    V[1 * p + 2][r] = 1
    V[0 * p + 2][r] = -1
    W[2 * n + 0][r] = 1
    W[2 * n + 1][r] = 1
    r += 1
    
    # M5
    U[0 * m + 0][r] = 1
    U[0 * m + 1][r] = 1
    V[1 * p + 3][r] = 1
    W[2 * n + 0][r] = -1
    W[3 * n + 0][r] = 1
    r += 1
    
    # M6
    U[1 * m + 0][r] = 1
    U[0 * m + 0][r] = -1
    V[0 * p + 2][r] = 1
    V[0 * p + 3][r] = 1
    W[3 * n + 1][r] = 1
    r += 1
    
    # M7
    U[0 * m + 1][r] = 1
    U[1 * m + 1][r] = -1
    V[1 * p + 2][r] = 1
    V[1 * p + 3][r] = 1
    W[2 * n + 0][r] = 1
    r += 1
    
    return U, V, W, rank

def _try_remove_term(U, V, W, r_remove, n, m, p):
    """Try removing term r_remove and see if the remaining terms still work."""
    rank = len(U[0])
    mask = [i for i in range(rank) if i != r_remove]
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    U_new = U_arr[:, mask]
    V_new = V_arr[:, mask]
    W_new = W_arr[:, mask]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                s = 0.0
                for r in range(len(mask)):
                    s += U_new[idx_u, r] * V_new[idx_v, r] * W_new[idx_w, r]
                if abs(s - 1.0) > 1e-9:
                    return None
    
    return [U_new, V_new, W_new]

def _search_lower_rank(U, V, W, n, m, p, time_budget):
    """
    Search for a lower rank decomposition by trying to remove terms.
    """
    start_time = time.time()
    current_rank = len(U[0])
    best_U, best_V, best_W, best_rank = U, V, W, current_rank
    
    # Try removing one term at a time
    for r_remove in range(current_rank):
        if time.time() - start_time > time_budget:
            break
        
        result = _try_remove_term(U, V, W, r_remove, n, m, p)
        if result is not None:
            U_new, V_new, W_new = result
            # Convert back to lists
            U_list = [[float(x) for x in row] for row in U_new]
            V_list = [[float(x) for x in row] for row in V_new]
            W_list = [[float(x) for x in row] for row in W_new]
            
            if len(U_list[0]) < best_rank:
                best_U, best_V, best_W, best_rank = U_list, V_list, W_list, len(U_list[0])
    
    return best_U, best_V, best_W, best_rank

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # First, build the known rank-28 decomposition
    U, V, W, rank = _build_rank28()
    
    # Verify it's correct
    if not _verify_decomposition(U, V, W, n, m, p):
        raise ValueError("Initial decomposition verification failed")
    
    # Try to reduce the rank
    time_remaining = TIME_LIMIT - (time.time() - start_time)
    if time_remaining > 5:
        U, V, W, rank = _search_lower_rank(U, V, W, n, m, p, time_remaining - 5)
    
    # Final verification
    if not _verify_decomposition(U, V, W, n, m, p):
        raise ValueError("Final decomposition verification failed")
    
    return {"U": U, "V": V, "W": W}
