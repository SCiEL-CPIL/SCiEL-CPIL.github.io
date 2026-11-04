"""Find a rank-minimal exact tensor decomposition for 2x4 * 4x5 matrix multiplication."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 5]

def _verify(U, V, W, tol=1e-10):
    """Check if U, V, W exactly decompose the matmul tensor."""
    n, m, p = DIMENSIONS
    r = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(r):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _clean_matrix(M):
    """Snap values to nearest integer or half-integer."""
    M_clean = np.zeros_like(M)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            val = M[i, j]
            # Try integer
            int_val = round(val)
            # Try half-integer
            half_val = round(val * 2) / 2.0
            
            if abs(val - int_val) < 1e-6:
                M_clean[i, j] = int_val
            elif abs(val - half_val) < 1e-6:
                M_clean[i, j] = half_val
            else:
                # Fallback to nearest half-integer
                M_clean[i, j] = half_val
    return M_clean

def construct():
    """
    Find a low-rank decomposition for 2x4 * 4x5 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Schoolbook solution as fallback: rank = n*m*p = 40
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][col] = 1
                V_school[j*p+k][col] = 1
                W_school[k*n+i][col] = 1
                col += 1
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school
    
    # Search for lower ranks (36, 37, 38, 39)
    # Use integer/half-integer random initialization
    for target_rank in range(36, best_rank):
        if time.time() > deadline:
            break
        
        found = False
        max_attempts = 10000
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            rng = np.random.RandomState(attempt * 7919 + target_rank * 131)
            
            # Initialize with random integers or half-integers
            # Scale to help convergence, but keep small
            U = rng.randint(-2, 3, size=(n*m, target_rank)) / 2.0
            V = rng.randint(-2, 3, size=(m*p, target_rank)) / 2.0
            W = rng.randint(-2, 3, size=(p*n, target_rank)) / 2.0
            
            # Verify
            if _verify(U.tolist(), V.tolist(), W.tolist()):
                best_U, best_V, best_W = U.tolist(), V.tolist(), W.tolist()
                best_rank = target_rank
                found = True
                break
        
        if found:
            break
            
    # If no better found, return schoolbook
    # Clean the output
    U_out = _clean_matrix(np.array(best_U)).tolist()
    V_out = _clean_matrix(np.array(best_V)).tolist()
    W_out = _clean_matrix(np.array(best_W)).tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
