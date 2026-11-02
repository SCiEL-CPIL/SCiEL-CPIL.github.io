"""Tensor decomposition search for 3x4x7 matrix multiplication."""
import time
import numpy as np
import random

DIMENSIONS = [3, 4, 7]
START_TIME = time.time()
TIME_LIMIT = 120.0

def _elapsed():
    return time.time() - START_TIME

def _schoolbook(n, m, p):
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
    return {"U": U, "V": V, "W": W}

def _verify_np(U, V, W, n, m, p):
    """Fast verification using NumPy."""
    if not U or not U[0]:
        return False
    rank = len(U[0])
    # Convert to numpy
    U_np = np.array(U, dtype=np.float64).reshape(n, m, rank)
    V_np = np.array(V, dtype=np.float64).reshape(m, p, rank)
    W_np = np.array(W, dtype=np.float64).reshape(p, n, rank)
    
    # Compute sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # U: (n, m, c)
    # V: (m, p, c)
    # W: (p, n, c) -> We need W[k, i, c]
    # einsum: i j c, j k c, k i c -> i j k
    try:
        T = np.einsum('ijc,jkc,kic->ijk', U_np, V_np, W_np)
    except Exception:
        return False
        
    # T should be all ones
    return np.allclose(T, 1.0, atol=1e-9)

def _search_rank(rank, n, m, p, max_attempts=500):
    """Random search for a valid decomposition of given rank."""
    # Entries from a small set of integers/half-integers
    # Using small numbers increases the chance of exact cancellation
    values = [0, 0, 0, 1, -1, 0.5, -0.5, 2, -2] # Weighted towards small
    
    for _ in range(max_attempts):
        if _elapsed() > TIME_LIMIT - 5:
            break
            
        # Generate random matrices
        # To speed up, we can generate lists and convert
        U = [[random.choice(values) for _ in range(rank)] for _ in range(n * m)]
        V = [[random.choice(values) for _ in range(rank)] for _ in range(m * p)]
        W = [[random.choice(values) for _ in range(rank)] for _ in range(p * n)]
        
        if _verify_np(U, V, W, n, m, p):
            return U, V, W
            
    return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook_result = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Lower bound for rank is max(n*m, m*p, p*n)
    lower_bound = max(n * m, m * p, p * n)
    
    # Try to find a decomposition with rank lower than schoolbook
    # We iterate from lower_bound up to schoolbook_rank - 1
    # But random search is inefficient for low ranks.
    # We try a few ranks: lower_bound, lower_bound + 5, ..., schoolbook - 5
    # Or just a few random ranks?
    # Given the time limit, we try ranks from lower_bound to 60.
    
    # To maximize chance, we try the most promising ranks first?
    # Actually, for matmul, rank is often close to n*m*p.
    # Let's try a range.
    
    # We will try to find ANY rank < 84.
    # We'll step through ranks from 40 to 83.
    # If we find one, we return it.
    
    # To save time, we limit attempts per rank.
    attempts_per_rank = 100
    
    for r in range(lower_bound, best_rank):
        if _elapsed() > TIME_LIMIT - 10:
            break
            
        # We can skip some ranks to try more attempts on others?
        # No, just iterate.
        
        res = _search_rank(r, n, m, p, max_attempts=attempts_per_rank)
        if res:
            U, V, W = res
            return {"U": U, "V": V, "W": W}
            
    return schoolbook_result
