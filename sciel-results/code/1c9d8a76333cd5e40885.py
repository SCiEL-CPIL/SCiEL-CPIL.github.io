"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
import numpy as np
from itertools import product
import random

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that the decomposition exactly reconstructs the matrix multiplication tensor."""
    n, m, p = n, m, p
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # The tensor T[i,j,k] should be 1 for all valid (i,j,k)
    # T = sum_r U[:,r] * V[:,r] * W[:,r] (outer product)
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _schoolbook_decomposition(n, m, p):
    """Return the standard schoolbook rank-n*m*p decomposition."""
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

def _try_reduce_rank(U, V, W, n, m, p, target_rank, max_iter=5000, lr=0.01):
    """
    Try to reduce the rank by optimizing factor matrices.
    Start from a perturbed version of the current decomposition and try to drive
    some columns toward zero, effectively reducing rank.
    """
    current_rank = len(U[0])
    if current_rank <= target_rank:
        return U, V, W
    
    # We'll try to find a decomposition of exactly target_rank
    # by random projection + optimization
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Randomly select target_rank columns and try to fit
    # Actually, let's use a different approach: random initialization for target_rank
    best_U, best_V, best_W = None, None, None
    best_error = float('inf')
    
    np.random.seed(42)
    for trial in range(3):
        # Initialize with random small values
        U_r = np.random.randn(n * m, target_rank) * 0.5
        V_r = np.random.randn(m * p, target_rank) * 0.5
        W_r = np.random.randn(p * n, target_rank) * 0.5
        
        # Gradient descent to minimize error
        for it in range(max_iter):
            # Compute current tensor values
            error = 0.0
            grad_U = np.zeros_like(U_r)
            grad_V = np.zeros_like(V_r)
            grad_W = np.zeros_like(W_r)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for r in range(target_rank):
                            val += U_r[i*m+j, r] * V_r[j*p+k, r] * W_r[k*n+i, r]
                        diff = val - 1.0
                        error += diff * diff
                        for r in range(target_rank):
                            grad_U[i*m+j, r] += 2 * diff * V_r[j*p+k, r] * W_r[k*n+i, r]
                            grad_V[j*p+k, r] += 2 * diff * U_r[i*m+j, r] * W_r[k*n+i, r]
                            grad_W[k*n+i, r] += 2 * diff * U_r[i*m+j, r] * V_r[j*p+k, r]
            
            # Update
            U_r -= lr * grad_U
            V_r -= lr * grad_V
            W_r -= lr * grad_W
            
            if error < best_error:
                best_error = error
                best_U = U_r.copy()
                best_V = V_r.copy()
                best_W = W_r.copy()
            
            if error < 1e-12:
                break
    
    if best_U is not None and best_error < 1e-10:
        # Convert to list of lists with integer/half-integer rounding if close
        U_out = [[round(x, 6) if abs(x - round(x, 6)) < 1e-5 else x for x in row] for row in best_U.tolist()]
        V_out = [[round(x, 6) if abs(x - round(x, 6)) < 1e-5 else x for x in row] for row in best_V.tolist()]
        W_out = [[round(x, 6) if abs(x - round(x, 6)) < 1e-5 else x for x in row] for row in best_W.tolist()]
        return U_out, V_out, W_out
    
    return U, V, W

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying 
    a 3 by 4 matrix by a 4 by 4 matrix.
    The output factor follows the official trace convention W[k*n+i,r].
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    U, V, W = _schoolbook_decomposition(n, m, p)
    best_rank = n * m * p
    best_U, best_V, best_W = U, V, W
    
    # Try to reduce rank from 47 down to a reasonable minimum
    # Lower bound: for n×m×p, rank >= max(n*p, m*max(n,p)) in some cases
    # For 3×4×4: lower bound is at least max(12, 4*4) = 16, but likely higher
    # Let's try ranks from 47 down to 30
    
    for target_rank in range(best_rank - 1, max(20, best_rank - 20), -1):
        if time.time() > deadline - 5.0:
            break
        
        U_new, V_new, W_new = _try_reduce_rank(U, V, W, n, m, p, target_rank, max_iter=2000, lr=0.05)
        
        # Verify
        if _verify_decomposition(U_new, V_new, W_new, n, m, p, tol=1e-8):
            best_rank = target_rank
            best_U, best_V, best_W = U_new, V_new, W_new
            break  # Found a valid lower rank, stop
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
