"""Matrix multiplication 3x3x4 rank minimization."""
import time
import numpy as np
import random

DIMENSIONS = [3, 3, 4]

def _schoolbook():
    """Return the schoolbook decomposition of rank n*m*p."""
    n, m, p = DIMENSIONS
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

def _verify_numpy(U, V, W, n, m, p):
    """
    Verify that the decomposition is exact using NumPy.
    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    Target T is all 1s.
    """
    rank = len(U[0])
    U_np = np.array(U, dtype=np.int64)
    V_np = np.array(V, dtype=np.int64)
    W_np = np.array(W, dtype=np.int64)
    
    T = np.zeros((n, m, p), dtype=np.int64)
    
    # Optimization: Precompute indices?
    # For small sizes, direct loop is fine.
    for r in range(rank):
        u_col = U_np[:, r]
        v_col = V_np[:, r]
        w_col = W_np[:, r]
        
        for i in range(n):
            for j in range(m):
                u_val = u_col[i * m + j]
                if u_val == 0: continue
                for k in range(p):
                    v_val = v_col[j * p + k]
                    if v_val == 0: continue
                    w_val = w_col[k * n + i]
                    T[i, j, k] += u_val * v_val * w_val
                    
    if not np.all(T == 1):
        return False
    return True

def _random_search(rank, iterations, coeff_range, n, m, p, deadline):
    """
    Try to find a valid decomposition of given rank with random coefficients.
    """
    start_time = time.time()
    for _ in range(iterations):
        if time.time() > deadline:
            return None
            
        # Generate random U, V, W
        # Use small integers to keep values small
        U = [[random.choice(coeff_range) for _ in range(rank)] for _ in range(n * m)]
        V = [[random.choice(coeff_range) for _ in range(rank)] for _ in range(m * p)]
        W = [[random.choice(coeff_range) for _ in range(rank)] for _ in range(p * n)]
        
        if _verify_numpy(U, V, W, n, m, p):
            return U, V, W
            
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook as fallback
    U_sb, V_sb, W_sb = _schoolbook()
    
    # Try to find a lower rank decomposition
    # Known rank is 27. Try 27, 28, 29...
    # Use random search with small integer coefficients
    # Coeff range -1, 0, 1 is a good start
    
    for target_rank in range(27, 36):
        if time.time() > deadline - 10:
            break
            
        # Increase iterations for lower ranks
        iterations = 5000 if target_rank < 30 else 2000
        
        result = _random_search(target_rank, iterations, [-1, 0, 1], n, m, p, deadline)
        
        if result:
            U, V, W = result
            # Double check with exact Python ints if needed, but numpy int64 is safe for small numbers
            return {"U": U, "V": V, "W": W}
            
    # Fallback to schoolbook
    return {"U": U_sb, "V": V_sb, "W": W_sb}
