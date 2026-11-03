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

def _compute_tensor_vals(U, V, W):
    """Compute all tensor values T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]"""
    vals = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                vals[i, j, k] = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
    return vals

def _try_rank(rank, time_budget):
    """Try to find a valid rank-decomposition using vectorized gradient descent."""
    deadline = time.time() + time_budget
    U_rows = n * m
    V_rows = m * p
    W_rows = p * n
    
    for attempt in range(200):
        if time.time() > deadline:
            break
        
        # Initialize randomly with small values
        np.random.seed(attempt * 7919)
        U = np.random.randn(U_rows, rank) * 0.3
        V = np.random.randn(V_rows, rank) * 0.3
        W = np.random.randn(W_rows, rank) * 0.3
        
        # Adaptive learning rate schedule
        base_lr = 0.05
        for it in range(5000):
            if time.time() > deadline:
                break
            
            # Compute tensor values
            vals = _compute_tensor_vals(U, V, W)
            err = vals - 1.0
            err_sq = np.sum(err ** 2)
            
            if err_sq < 1e-14:
                break
            
            lr = base_lr / (1 + it * 0.002)
            
            # Gradient for U[i*m+j, r] = sum_k err[i,j,k] * V[j*p+k,r] * W[k*n+i,r]
            # Vectorized: for each (i,j), grad_U[i*m+j,:] = sum_k err[i,j,k] * (V[j*p+k,:] * W[k*n+i,:])
            grad_U = np.zeros_like(U)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        grad_U[i*m+j] += err[i, j, k] * (V[j*p+k] * W[k*n+i])
            U -= lr * grad_U
            
            # Gradient for V[j*p+k, r] = sum_i err[i,j,k] * U[i*m+j,r] * W[k*n+i,r]
            grad_V = np.zeros_like(V)
            for j in range(m):
                for k in range(p):
                    for i in range(n):
                        grad_V[j*p+k] += err[i, j, k] * (U[i*m+j] * W[k*n+i])
            V -= lr * grad_V
            
            # Gradient for W[k*n+i, r] = sum_j err[i,j,k] * U[i*m+j,r] * V[j*p+k,r]
            grad_W = np.zeros_like(W)
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        grad_W[k*n+i] += err[i, j, k] * (U[i*m+j] * V[j*p+k])
            W -= lr * grad_W
            
            # Periodic check
            if it % 100 == 0 and err_sq < 1e-8:
                result = _to_half_integer(U.tolist(), V.tolist(), W.tolist())
                if result is not None:
                    return result
                if _check_exact(U.tolist(), V.tolist(), W.tolist()):
                    return [U.tolist(), V.tolist(), W.tolist()]
        
        # Final check after optimization
        result = _to_half_integer(U.tolist(), V.tolist(), W.tolist())
        if result is not None:
            return result
        if _check_exact(U.tolist(), V.tolist(), W.tolist()):
            return [U.tolist(), V.tolist(), W.tolist()]
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin for 120s limit
    
    # Try ranks from lower to higher, with more time for lower ranks
    # For 3x3x4, schoolbook is 36. Try to find something better.
    ranks_to_try = [28, 29, 30, 31, 32, 33, 34, 35]
    
    # Allocate time proportionally: more time for lower ranks
    total_time = deadline - start_time
    # Give more weight to lower ranks
    weights = [3.0, 2.5, 2.0, 1.5, 1.2, 1.0, 0.8, 0.6]
    total_weight = sum(weights)
    time_alloc = [total_time * w / total_weight for w in weights]
    
    for idx, r in enumerate(ranks_to_try):
        if time.time() > deadline - 5:
            break
        remaining = deadline - time.time()
        time_budget = min(time_alloc[idx], remaining - 5)
        if time_budget < 1:
            break
        result = _try_rank(r, time_budget)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook
    return _schoolbook()
