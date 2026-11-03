import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def _build_schoolbook(n, m, p):
    """Build the schoolbook decomposition of rank n*m*p."""
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
    return {"U": U, "V": V, "W": W}

def _verify_decomposition_fast(U, V, W, n, m, p):
    """
    Verify decomposition using NumPy einsum for speed.
    """
    rank = len(U[0]) if U else 0
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if any(len(row) != rank for row in U) or any(len(row) != rank for row in V) or any(len(row) != rank for row in W):
        return False

    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    U_res = U_np.reshape(n, m, rank)
    V_res = V_np.reshape(m, p, rank)
    W_res = W_np.reshape(p, n, rank)
    
    S = np.einsum('ijc,jkc,kic->ijk', U_res, V_res, W_res)
    
    return np.allclose(S, 1.0, atol=1e-9)

def _try_column_merge(n, m, p, U, V, W, deadline):
    """Try to merge columns to reduce rank."""
    U = [row[:] for row in U]
    V = [row[:] for row in V]
    W = [row[:] for row in W]
    rank = len(U[0])
    
    # Try to merge pairs of columns
    for c1 in range(rank):
        if time.time() > deadline - 5.0:
            break
        for c2 in range(c1 + 1, rank):
            # Try merging column c2 into c1
            U_new = [row[:] for row in U]
            V_new = [row[:] for row in V]
            W_new = [row[:] for row in W]
            
            # Add column c2 to column c1
            for i in range(len(U)):
                U_new[i][c1] += U_new[i][c2]
            for i in range(len(V)):
                V_new[i][c1] += V_new[i][c2]
            for i in range(len(W)):
                W_new[i][c1] += W_new[i][c2]
            
            # Remove column c2
            U_new = [row[:c2] + row[c2+1:] for row in U_new]
            V_new = [row[:c2] + row[c2+1:] for row in V_new]
            W_new = [row[:c2] + row[c2+1:] for row in W_new]
            
            if _verify_decomposition_fast(U_new, V_new, W_new, n, m, p):
                return {"U": U_new, "V": V_new, "W": W_new}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0
    
    # 1. Build Schoolbook (Rank 60)
    schoolbook = _build_schoolbook(n, m, p)
    best_result = schoolbook
    best_rank = n * m * p
    
    # 2. Try column merging
    if time.time() < deadline - 10.0:
        result = _try_column_merge(n, m, p, schoolbook["U"], schoolbook["V"], schoolbook["W"], deadline)
        if result is not None:
            rank = len(result["U"][0])
            if rank < best_rank:
                best_result = result
                best_rank = rank
    
    return best_result
# EVOLVE-BLOCK-END
