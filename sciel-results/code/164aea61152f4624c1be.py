"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

import numpy as np
import time
import random
import sys

# EVOLVE-BLOCK-START

def _compute_error(U, V, W, n, m, p):
    """Compute the sum of squared errors for the decomposition."""
    # U: (n*m, R), V: (m*p, R), W: (p*n, R)
    # Target: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                # Use dot product for speed if possible, but loop is fine for small R
                # We can vectorize over c
                u_row = U[i*m + j]
                v_row = V[j*p + k]
                w_row = W[k*n + i]
                s = np.dot(u_row, v_row * w_row)
                err += (s - 1.0) ** 2
    return err

def _gradient_descent(U, V, W, n, m, p, steps=100, lr=0.01):
    """Perform gradient descent to minimize the error."""
    R = U.shape[1]
    for step in range(steps):
        # Compute gradients
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        # Efficient gradient computation
        # E = sum_{i,j,k} (sum_c U_{ij,c} V_{jk,c} W_{ki,c} - 1)^2
        # Let T_{ij,k,c} = V_{jk,c} W_{ki,c}
        # Then sum_c U_{ij,c} T_{ij,k,c} is the partial sum for (i,j,k)
        
        # We can compute this by iterating over i, j, k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_row = U[i*m + j]
                    v_row = V[j*p + k]
                    w_row = W[k*n + i]
                    
                    # s = sum_c u_row[c] * v_row[c] * w_row[c]
                    s = np.dot(u_row, v_row * w_row)
                    err_term = 2 * (s - 1.0)
                    
                    # dU_{ij,c} += err_term * v_row[c] * w_row[c]
                    dU[i*m + j] += err_term * v_row * w_row
                    # dV_{jk,c} += err_term * u_row[c] * w_row[c]
                    dV[j*p + k] += err_term * u_row * w_row
                    # dW_{ki,c} += err_term * u_row[c] * v_row[c]
                    dW[k*n + i] += err_term * u_row * v_row
        
        # Update
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        
        # Optional: Normalize or regularize to prevent explosion
        # But for small steps, it should be okay.
        
    return U, V, W

def _try_rank_with_grad(n, m, p, target_rank, time_limit, num_tries=10):
    """Try to find a decomposition with target_rank using gradient descent."""
    start_time = time.time()
    best_err = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    for trial in range(num_tries):
        if time.time() > start_time + time_limit:
            break
            
        # Random initialization
        U = np.random.randn(n * m, target_rank) * 0.1
        V = np.random.randn(m * p, target_rank) * 0.1
        W = np.random.randn(p * n, target_rank) * 0.1
        
        # Run gradient descent
        U, V, W = _gradient_descent(U, V, W, n, m, p, steps=200, lr=0.05)
        
        # Check error
        err = _compute_error(U, V, W, n, m, p)
        
        if err < best_err:
            best_err = err
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()
            
        # If error is very small, try to snap to half-integers
        if err < 1e-5:
            # Check if entries are close to half-integers
            U_rounded = np.round(best_U * 2) / 2
            V_rounded = np.round(best_V * 2) / 2
            W_rounded = np.round(best_W * 2) / 2
            
            # Verify rounded version
            err_rounded = _compute_error(U_rounded, V_rounded, W_rounded, n, m, p)
            if err_rounded < 1e-9:
                return {
                    "U": U_rounded.tolist(),
                    "V": V_rounded.tolist(),
                    "W": W_rounded.tolist()
                }
            
            # Also try integer rounding
            U_int = np.round(best_U)
            V_int = np.round(best_V)
            W_int = np.round(best_W)
            err_int = _compute_error(U_int, V_int, W_int, n, m, p)
            if err_int < 1e-9:
                return {
                    "U": U_int.tolist(),
                    "V": V_int.tolist(),
                    "W": W_int.tolist()
                }
                
    return None

def _schoolbook_decomposition(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
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

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # Schoolbook fallback
    schoolbook = _schoolbook_decomposition(n, m, p)
    schoolbook_rank = n * m * p
    
    # Try to find better ranks
    # We try ranks from schoolbook_rank - 1 down to schoolbook_rank - 10
    # If time permits, we can go lower.
    max_search_rank = schoolbook_rank - 10
    min_search_rank = max(1, schoolbook_rank - 20) # Don't go too low, unlikely
    
    # Adjust search range based on time
    # For 3x5x5, rank 75 is schoolbook.
    # Let's try 74, 73, 72, 71, 70.
    
    for target_rank in range(schoolbook_rank - 1, min_search_rank - 1, -1):
        if time.time() > deadline - 10.0:
            break
            
        # Allocate time for this rank
        # If we have 100s left, and 5 ranks to try, 20s each?
        # But gradient descent is fast.
        time_per_rank = min(20.0, deadline - time.time() - 5.0)
        
        if time_per_rank < 1.0:
            break
            
        result = _try_rank_with_grad(n, m, p, target_rank, time_per_rank, num_tries=5)
        
        if result is not None:
            return result
            
    return schoolbook

# EVOLVE-BLOCK-END
