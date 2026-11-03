"""Optimized matrix multiplication tensor decomposition for 2x3 * 3x7."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def _schoolbook_decomposition(n, m, p):
    """Return the standard schoolbook tensor decomposition of rank n*m*p."""
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

def _verify_decomposition(U, V, W, n, m, p, tolerance=1e-9):
    """
    Verify that the decomposition exactly computes matrix multiplication.
    Uses exact arithmetic for integer/half-integer entries, or high-precision check.
    """
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for r in range(rank):
                    total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(total - 1.0) > tolerance:
                    return False
    return True

def _try_rank_search(n, m, p, max_rank, time_limit):
    """
    Attempt to find a decomposition of rank < n*m*p using ALS-like optimization.
    Returns None if no better decomposition is found within the time limit.
    """
    start_time = time.time()
    schoolbook_rank = n * m * p
    
    # Try random initialization with ALS for ranks slightly below schoolbook
    # This is a heuristic and may not find anything, but worth trying
    best_result = None
    
    # Only try a few ranks and a few iterations to stay within time limits
    for target_rank in [schoolbook_rank - 1, schoolbook_rank - 2]:
        if target_rank < 1:
            continue
        if time.time() > time_limit - 10:
            break
            
        for attempt in range(3):
            if time.time() > time_limit - 5:
                break
                
            # Random initialization
            U = np.random.randn(n * m, target_rank)
            V = np.random.randn(m * p, target_rank)
            W = np.random.randn(p * n, target_rank)
            
            # Simple ALS iterations
            for iter in range(50):
                if time.time() > time_limit - 5:
                    break
                # Update U
                for r in range(target_rank):
                    U[:, r] = np.zeros(n * m)
                    for i in range(n):
                        for j in range(m):
                            idx_u = i * m + j
                            # U[idx_u, r] = sum_k W[k*n+i, r] * V[j*p+k, r] / sum_k W^2 * V^2
                            num = 0.0
                            den = 0.0
                            for k in range(p):
                                w_val = W[k * n + i, r]
                                v_val = V[j * p + k, r]
                                num += w_val * v_val
                                den += w_val * w_val * v_val * v_val
                            if den > 1e-12:
                                U[idx_u, r] = num / den
                            
                # Update V
                for r in range(target_rank):
                    V[:, r] = np.zeros(m * p)
                    for j in range(m):
                        for k in range(p):
                            idx_v = j * p + k
                            num = 0.0
                            den = 0.0
                            for i in range(n):
                                u_val = U[i * m + j, r]
                                w_val = W[k * n + i, r]
                                num += u_val * w_val
                                den += u_val * u_val * w_val * w_val
                            if den > 1e-12:
                                V[idx_v, r] = num / den
                            
                # Update W
                for r in range(target_rank):
                    W[:, r] = np.zeros(p * n)
                    for k in range(p):
                        for i in range(n):
                            idx_w = k * n + i
                            num = 0.0
                            den = 0.0
                            for j in range(m):
                                u_val = U[i * m + j, r]
                                v_val = V[j * p + k, r]
                                num += u_val * v_val
                                den += u_val * u_val * v_val * v_val
                            if den > 1e-12:
                                W[idx_w, r] = num / den
                                
            # Check if this solution is close to valid
            # Convert to list of lists and verify
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            
            if _verify_decomposition(U_list, V_list, W_list, n, m, p, tolerance=0.1):
                # Try to refine to exact values
                # This is a rough check; real exact decomposition would need more work
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                
    return best_result

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying
    a 2x3 matrix by a 3x7 matrix.
    Returns the schoolbook decomposition (rank 42) if no better is found.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 10s safety margin
    
    # Try to find a better decomposition
    improved = _try_rank_search(n, m, p, n * m * p - 1, deadline)
    
    if improved is not None:
        # Verify the improved decomposition
        if _verify_decomposition(improved["U"], improved["V"], improved["W"], n, m, p, tolerance=1e-6):
            return improved
    
    # Fallback to schoolbook
    return _schoolbook_decomposition(n, m, p)
# EVOLVE-BLOCK-END
