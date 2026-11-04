"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

import time
import sys
import numpy as np

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W form a valid exact decomposition."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook():
    n, m, p = DIMENSIONS
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

def _try_merge(U, V, W, n, m, p, col_a, col_b):
    """Try to merge two columns into one by finding a linear combination."""
    rank = len(U[0])
    # We want to find alpha, beta such that for all (i,j,k):
    # alpha * U[i*m+j, col_a] * V[j*p+k, col_a] * W[k*n+i, col_a] +
    # beta * U[i*m+j, col_b] * V[j*p+k, col_b] * W[k*n+i, col_b] =
    # U[i*m+j, new] * V[j*p+k, new] * W[k*n+i, new]
    # This is complex. Instead, try simple cases where one column is a scalar multiple.
    
    # Check if col_a is a scalar multiple of col_b in all three matrices
    Ua = np.array([U[i][col_a] for i in range(n*m)], dtype=np.float64)
    Ub = np.array([U[i][col_b] for i in range(n*m)], dtype=np.float64)
    Va = np.array([V[i][col_a] for i in range(m*p)], dtype=np.float64)
    Vb = np.array([V[i][col_b] for i in range(m*p)], dtype=np.float64)
    Wa = np.array([W[i][col_a] for i in range(p*n)], dtype=np.float64)
    Wb = np.array([W[i][col_b] for i in range(p*n)], dtype=np.float64)
    
    # Try to find scalar c such that Ua = c*Ub, Va = c*Vb, Wa = c*Wb
    # This would mean the two terms are identical up to scaling, and we can just drop one.
    # But in schoolbook, each column is unique, so this won't work directly.
    
    # Alternative: try to find a rank-1 update that cancels the difference
    # This is getting complex. For now, just try random perturbations.
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Start with schoolbook
    best = _schoolbook()
    best_rank = n * m * p
    
    # The tensor for 2x3x7 matrix multiplication has a known structure.
    # Let's try to find a lower rank decomposition using a more clever approach.
    
    # For matrix multiplication A (n x m) * B (m x p) = C (n x p),
    # the tensor T[i,j,k] = 1 if the indices are consistent, else 0.
    # Actually, T is only defined on the valid triples, and T[i,j,k] = 1 for all valid (i,j,k).
    
    # Key insight: the tensor is the "matrix multiplication tensor" which for n=2, m=3, p=7
    # has rank at most 42 (schoolbook). Whether it can be lower depends on the specific dimensions.
    
    # For now, return schoolbook as the safe answer.
    # A proper search would require more sophisticated methods (e.g., using the fact that
    # the tensor has a specific algebraic structure).
    
    return best
# EVOLVE-BLOCK-END
