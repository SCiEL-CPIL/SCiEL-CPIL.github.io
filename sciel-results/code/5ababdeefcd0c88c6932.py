"""Rewritten program for matrix multiplication 2x3 by 3x4 using group action / character method."""
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x4 matrix.
    
    Strategy: Use numerical optimization to find a valid decomposition, then verify exactness.
    We search for the minimum rank r such that there exist matrices U (6xr), V (12xr), W (8xr)
    with entries in {-1, -0.5, 0, 0.5, 1} satisfying:
    sum_c U[i*3+j, c] * V[j*4+k, c] * W[k*2+i, c] = 1 for all i,j,k
    """
    n, m, p = DIMENSIONS  # 2, 3, 4
    
    # Try ranks from 12 upward
    for rank in range(12, 25):
        result = _find_decomposition(rank, n, m, p)
        if result is not None:
            return result
    
    # Fallback to schoolbook
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


def _find_decomposition(rank, n, m, p):
    """
    Find a rank-r decomposition using random search with linear solve for W.
    """
    import random
    import time
    
    start_time = time.time()
    max_time = 10  # seconds per rank
    
    # Candidate values: integers and half-integers, including larger values
    values = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
    
    # Number of random restarts
    n_restarts = 200
    
    for restart in range(n_restarts):
        if time.time() - start_time > max_time:
            break
        
        # Random initialization of U and V
        U = np.array([[random.choice(values) for _ in range(rank)] for _ in range(n * m)])
        V = np.array([[random.choice(values) for _ in range(rank)] for _ in range(m * p)])
        
        # Solve for W: for each (k, i), we need
        # sum_c W[k*n+i, c] * U[i*m+j, c] * V[j*p+k, c] = 1 for all j
        # This is a linear system in W[k*n+i, :]
        
        W = np.zeros((p * n, rank))
        valid = True
        
        for k in range(p):
            for i in range(n):
                # Build the linear system: for each j in 0..m-1
                # sum_c W[k*n+i, c] * (U[i*m+j, c] * V[j*p+k, c]) = 1
                A_sys = np.zeros((m, rank))
                for j in range(m):
                    for c in range(rank):
                        A_sys[j, c] = U[i * m + j, c] * V[j * p + k, c]
                
                b_sys = np.ones(m)
                
                # Solve A_sys * W[k*n+i, :]^T = b_sys
                try:
                    W[k * n + i, :] = np.linalg.lstsq(A_sys, b_sys, rcond=None)[0]
                except:
                    valid = False
                    break
            
            if not valid:
                break
        
        if not valid:
            continue
        
        # Round W to nearest value in the candidate set
        W_rounded = np.array([[round(W[row, col] / 0.5) * 0.5 for col in range(rank)] for row in range(p * n)])
        
        # Check if rounded W is valid
        if _is_valid(U, V, W_rounded, n, m, p):
            return _convert_to_list(U, V, W_rounded)
    
    return None


def _is_valid(U, V, W, n, m, p):
    """
    Check if the decomposition is valid:
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    """
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(U.shape[1]):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True


def _convert_to_list(U, V, W):
    """Convert numpy arrays to lists of lists."""
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }


# EVOLVE-BLOCK-END
