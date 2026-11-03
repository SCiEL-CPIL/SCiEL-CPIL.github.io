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
                u_row = U[i*m + j]
                v_row = V[j*p + k]
                w_row = W[k*n + i]
                s = np.dot(u_row, v_row * w_row)
                err += (s - 1.0) ** 2
    return err

def _structured_init(n, m, p, target_rank, seed):
    """Create structured initialization using Kronecker product hints for 5=4+1."""
    rng = np.random.RandomState(seed)
    
    # Base random initialization
    U = rng.randn(n * m, target_rank) * 0.3
    V = rng.randn(m * p, target_rank) * 0.3
    W = rng.randn(p * n, target_rank) * 0.3
    
    # Add structure: exploit the 5 = 4+1 split
    # For indices involving the 5th element, use a different scale
    # This helps the optimizer find the right structure faster
    for c in range(target_rank):
        # Every 5th column gets a boost for the "1" part of 4+1
        if c % 5 == 4:
            # Boost entries related to the last index in the 5-dim space
            for i in range(n):
                for j in range(m):
                    if j == m - 1:  # last column of A
                        U[i*m + j, c] *= 1.5
            for j in range(m):
                for k in range(p):
                    if j == m - 1:  # last row of B
                        V[j*p + k, c] *= 1.5
            for k in range(p):
                for i in range(n):
                    if i == n - 1:  # last row of output
                        W[k*n + i, c] *= 1.5
    
    return U, V, W

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

def _try_rank_with_grad(n, m, p, target_rank, time_limit, num_tries=15):
    """Try to find a decomposition with target_rank using gradient descent."""
    start_time = time.time()
    best_err = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    for trial in range(num_tries):
        if time.time() > start_time + time_limit:
            break
            
        # Structured random initialization
        U, V, W = _structured_init(n, m, p, target_rank, seed=trial * 1000 + 42)
        
        # Run gradient descent with more steps and better lr
        U, V, W = _gradient_descent(U, V, W, n, m, p, steps=300, lr=0.15)
        
        # Check error
        err = _compute_error(U, V, W, n, m, p)
        
        if err < best_err:
            best_err = err
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()
            
        # If error is small, try to snap to half-integers and integers
        if err < 1e-4:
            # Try multiple scales for rounding
            for scale in [1, 2, 4, 8]:
                U_rounded = np.round(best_U * scale) / scale
                V_rounded = np.round(best_V * scale) / scale
                W_rounded = np.round(best_W * scale) / scale
                
                # Verify rounded version
                err_rounded = _compute_error(U_rounded, V_rounded, W_rounded, n, m, p)
                if err_rounded < 1e-9:
                    return {
                        "U": U_rounded.tolist(),
                        "V": V_rounded.tolist(),
                        "W": W_rounded.tolist()
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
    # We try ranks from schoolbook_rank - 1 down to schoolbook_rank - 15
    min_search_rank = max(1, schoolbook_rank - 15)
    
    # For 3x5x5, rank 75 is schoolbook.
    # Try 74, 73, 72, 71, 70, 69, 68, 67, 66, 65.
    
    for target_rank in range(schoolbook_rank - 1, min_search_rank - 1, -1):
        if time.time() > deadline - 10.0:
            break
            
        # Allocate time for this rank
        # Give more time per rank to allow for more trials
        time_per_rank = min(25.0, deadline - time.time() - 5.0)
        
        if time_per_rank < 1.0:
            break
            
        result = _try_rank_with_grad(n, m, p, target_rank, time_per_rank, num_tries=10)
        
        if result is not None:
            return result
            
    return schoolbook

# EVOLVE-BLOCK-END
