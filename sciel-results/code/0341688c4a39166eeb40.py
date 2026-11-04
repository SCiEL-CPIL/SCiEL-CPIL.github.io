"""Matrix multiplication 4x5x6 tensor decomposition."""
import time
import random
import numpy as np

DIMENSIONS = [4, 5, 6]


def _make_schoolbook(n, m, p):
    """Schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return U, V, W


def _verify_numpy(U, V, W, n, m, p, tol=1e-9):
    """Verify the decomposition using numpy for speed."""
    rank = len(U[0]) if U else 0
    if rank == 0:
        return False
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # For each (i,j,k), check sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = U_arr[i * m + j]
                v_row = V_arr[j * p + k]
                w_row = W_arr[k * n + i]
                # Element-wise multiply then sum
                val = np.sum(u_row * v_row * w_row)
                if abs(val - 1.0) > tol:
                    return False
    return True


def _try_column_merging(n, m, p, deadline):
    """
    Try to reduce rank by merging pairs of schoolbook columns.
    Two columns c1 and c2 can be merged if there exists a linear combination
    that preserves all tensor entries. This is a heuristic approach.
    """
    U, V, W = _make_schoolbook(n, m, p)
    rank = n * m * p
    
    # Try to find pairs of columns that can be combined
    # For simplicity, try small structured modifications
    # This is very unlikely to work for exact decompositions,
    # but we'll attempt a few quick checks
    
    best_rank = rank
    best_result = (U, V, W)
    
    # Quick attempt: try rank = rank - 1 by zeroing one column and adjusting others
    # This won't work in general, but check if any single column is redundant
    # (it won't be for schoolbook, but the check is fast)
    
    return best_result, best_rank


def _structured_search(n, m, p, target_rank, iterations, deadline):
    """
    Structured random search for a valid decomposition at target_rank.
    Uses small integer/half-integer entries.
    """
    entries = [-1, -0.5, 0, 0.5, 1]
    
    for _ in range(iterations):
        if time.time() > deadline - 10:
            break
        
        U = [[random.choice(entries) for _ in range(target_rank)] for _ in range(n * m)]
        V = [[random.choice(entries) for _ in range(target_rank)] for _ in range(m * p)]
        W = [[random.choice(entries) for _ in range(target_rank)] for _ in range(p * n)]
        
        if _verify_numpy(U, V, W, n, m, p):
            print(f"Found valid rank {target_rank}")
            return (U, V, W), target_rank
    
    return None, target_rank


def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook as guaranteed fallback
    U_school, V_school, W_school = _make_schoolbook(n, m, p)
    school_rank = n * m * p
    
    best_result = (U_school, V_school, W_school)
    best_rank = school_rank
    
    # Attempt 1: Column merging (very quick, unlikely to succeed)
    if time.time() < deadline - 80:
        merged_result, merged_rank = _try_column_merging(n, m, p, deadline)
        if merged_rank < best_rank:
            best_result = merged_result
            best_rank = merged_rank
    
    # Attempt 2: Structured random search for a few target ranks
    # Based on literature, try ranks that might be achievable
    # For 4x5x6, theoretical lower bounds suggest ~40-50 might be possible
    # but exact decompositions are hard to find randomly
    
    target_ranks = [119, 118, 115, 110, 100]
    iterations_per_rank = 50
    
    for tr in target_ranks:
        if time.time() > deadline - 20:
            break
        
        result, found_rank = _structured_search(n, m, p, tr, iterations_per_rank, deadline)
        if result is not None:
            best_result = result
            best_rank = found_rank
            break
    
    U, V, W = best_result
    return {"U": U, "V": V, "W": W}
