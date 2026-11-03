"""Matrix multiplication 3x3x4: rank-minimal exact tensor decomposition search."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 4]
n, m, p = DIMENSIONS

# The target tensor: T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
# Decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
# U: (n*m) x rank, V: (m*p) x rank, W: (p*n) x rank

def _check_exact(U, V, W):
    """Check if the decomposition is exact (within tight tolerance for integer/half-integer)."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Compute the tensor
    err = 0.0
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
                if max_err > 1e-9:
                    return False
    return True

def _to_half_integer(U, V, W):
    """Round entries to nearest half-integer and check validity."""
    U_r = np.round(2 * np.array(U, dtype=float)) / 2.0
    V_r = np.round(2 * np.array(V, dtype=float)) / 2.0
    W_r = np.round(2 * np.array(W, dtype=float)) / 2.0
    
    if _check_exact(U_r.tolist(), V_r.tolist(), W_r.tolist()):
        return [U_r.tolist(), V_r.tolist(), W_r.tolist()]
    return None

def _schoolbook():
    """Return the schoolbook decomposition of rank n*m*p."""
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

def _try_rank(rank, time_budget):
    """Try to find a valid rank-decomposition using iterative optimization."""
    deadline = time.time() + time_budget
    U_rows = n * m
    V_rows = m * p
    W_rows = p * n
    
    best_err = float('inf')
    
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        # Initialize randomly with small values
        np.random.seed(attempt * 7919)
        U = np.random.randn(U_rows, rank) * 0.5
        V = np.random.randn(V_rows, rank) * 0.5
        W = np.random.randn(W_rows, rank) * 0.5
        
        # Alternating minimization: fix two, solve for third
        for it in range(2000):
            if time.time() > deadline:
                break
            
            # Compute current error
            err_sum = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                        err_sum += (val - 1.0) ** 2
            
            if err_sum < 1e-12:
                break
            
            # Update U: for each (i,j), solve least squares
            # sum_r U[i*m+j,r] * (V[j*p+k,r] * W[k*n+i,r]) = 1 for each k
            # Actually, the constraint is: for all i,j,k: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
            # This is nonlinear. Use simple gradient descent.
            
            lr = 0.01 / (1 + it * 0.001)
            
            # Gradient for U[i*m+j, r]: sum_k (V[j*p+k,r] * W[k*n+i,r]) * (val - 1)
            # Actually: d/dU[i*m+j,r] of (sum_r' U[i*m+j,r'] V[j*p+k,r'] W[k*n+i,r'] - 1)^2
            # = 2 * (val_ijk - 1) * V[j*p+k,r] * W[k*n+i,r]
            # Summed over k for the gradient of U[i*m+j,r]
            
            # Update U
            for i in range(n):
                for j in range(m):
                    for r in range(rank):
                        grad = 0.0
                        for k in range(p):
                            val = sum(U[i*m+j, r2] * V[j*p+k, r2] * W[k*n+i, r2] for r2 in range(rank))
                            grad += (val - 1.0) * V[j*p+k, r] * W[k*n+i, r]
                        U[i*m+j, r] -= lr * grad
            
            # Update V
            for j in range(m):
                for k in range(p):
                    for r in range(rank):
                        grad = 0.0
                        for i in range(n):
                            val = sum(U[i*m+j, r2] * V[j*p+k, r2] * W[k*n+i, r2] for r2 in range(rank))
                            grad += (val - 1.0) * U[i*m+j, r] * W[k*n+i, r]
                        V[j*p+k, r] -= lr * grad
            
            # Update W
            for i in range(n):
                for k in range(p):
                    for r in range(rank):
                        grad = 0.0
                        for j in range(m):
                            val = sum(U[i*m+j, r2] * V[j*p+k, r2] * W[k*n+i, r2] for r2 in range(rank))
                            grad += (val - 1.0) * U[i*m+j, r] * V[j*p+k, r]
                        W[k*n+i, r] -= lr * grad
        
        # Check and round
        result = _to_half_integer(U.tolist(), V.tolist(), W.tolist())
        if result is not None:
            return result
        
        # Also check without rounding
        if _check_exact(U.tolist(), V.tolist(), W.tolist()):
            return [U.tolist(), V.tolist(), W.tolist()]
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin for 120s limit
    
    # Known bounds: rank is at most 36 (schoolbook), at least... 
    # For 3x3x3, best known is 23. For 3x3x4, it's between ~25 and 36.
    # Try ranks from 30 upward, spending more time on lower ranks.
    
    # Try a range of ranks. Start with a few lower ones with less time,
    # then higher ones.
    ranks_to_try = [30, 31, 32, 33, 34, 35]
    for r in ranks_to_try:
        if time.time() > deadline - 10:
            break
        time_budget = min(15, deadline - time.time())
        if time_budget < 2:
            break
        result = _try_rank(r, time_budget)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook
    return _schoolbook()
