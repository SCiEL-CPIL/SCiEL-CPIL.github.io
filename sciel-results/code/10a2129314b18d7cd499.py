import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 5, 5]

def validate_exact(U, V, W, n, m, p):
    """Validate with exact fraction arithmetic."""
    r = len(W[0]) if W else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                val = Fraction(0, 1)
                for c in range(r):
                    val += Fraction(U[idx_u][c]) * Fraction(V[idx_v][c]) * Fraction(W[idx_w][c])
                if val != Fraction(1, 1):
                    return False
    return True

def validate_float(U, V, W, n, m, p, tol=1e-9):
    """Fast float validation."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    r = W.shape[1]
    # Target tensor M has 1s at (i*m+j, j*p+k, k*n+i)
    # We check if sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                val = np.dot(U[idx_u, :], V[idx_v, :] * W[idx_w, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def schoolbook_decomposition(n, m, p):
    """Standard rank n*m*p decomposition."""
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
    return U, V, W

def tensor_error(U, V, W, n, m, p):
    """Calculate L2 error of the tensor decomposition."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    r = W.shape[1]
    error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                val = np.dot(U[idx_u, :], V[idx_v, :] * W[idx_w, :])
                error += (val - 1.0) ** 2
    return error

def try_optimize_rank(n, m, p, rank, deadline, max_iter=1000, lr=0.05):
    """
    Try to find a decomposition of given rank using gradient descent.
    Returns (U, V, W) as lists if found, else None.
    """
    U = np.random.randn(n * m, rank) * 0.5
    V = np.random.randn(m * p, rank) * 0.5
    W = np.random.randn(p * n, rank) * 0.5
    
    # Initialize with some sparsity
    mask_u = np.random.rand(n * m, rank) < 0.3
    mask_v = np.random.rand(m * p, rank) < 0.3
    mask_w = np.random.rand(p * n, rank) < 0.3
    U *= mask_u
    V *= mask_v
    W *= mask_w
    
    best_error = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        if time.time() > deadline - 5:
            break
            
        # Calculate error
        err = tensor_error(U, V, W, n, m, p)
        if err < best_error:
            best_error = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
        if err < 1e-6:
            # Try to snap to half-integers
            U_snap = np.round(U * 2) / 2
            V_snap = np.round(V * 2) / 2
            W_snap = np.round(W * 2) / 2
            
            U_list = U_snap.tolist()
            V_list = V_snap.tolist()
            W_list = W_snap.tolist()
            
            if validate_exact(U_list, V_list, W_list, n, m, p):
                return U_list, V_list, W_list
            
        # Gradient step (approximate)
        # We use a simple random perturbation for speed and simplicity
        # instead of full gradient which is complex to implement for tensor
        step = 0.1
        U += np.random.randn(n * m, rank) * step
        V += np.random.randn(m * p, rank) * step
        W += np.random.randn(p * n, rank) * step
        
        # Soft thresholding to encourage sparsity
        U[np.abs(U) < 0.1] = 0
        V[np.abs(V) < 0.1] = 0
        W[np.abs(W) < 0.1] = 0

    # Final check on best found
    if best_error < 1e-4:
        U_snap = np.round(best_U * 2) / 2
        V_snap = np.round(best_V * 2) / 2
        W_snap = np.round(best_W * 2) / 2
        
        U_list = U_snap.tolist()
        V_list = V_snap.tolist()
        W_list = W_snap.tolist()
        
        if validate_exact(U_list, V_list, W_list, n, m, p):
            return U_list, V_list, W_list
            
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s buffer
    
    # Start with schoolbook
    U, V, W = schoolbook_decomposition(n, m, p)
    best_result = (U, V, W)
    best_rank = len(W[0])
    
    # Try to find lower ranks
    # Prioritize ranks that are likely to exist based on known bounds
    # For 2x5x5, schoolbook is 50.
    # Try 45, 40, 35, 30, 25
    
    candidate_ranks = [45, 40, 35, 30, 25]
    
    for rank in candidate_ranks:
        if rank >= best_rank:
            continue
        if time.time() > deadline - 20:
            break
            
        # Try multiple random starts
        found = False
        for attempt in range(5):
            if time.time() > deadline - 10:
                break
                
            result = try_optimize_rank(n, m, p, rank, deadline, max_iter=200)
            if result:
                U, V, W = result
                best_result = (U, V, W)
                best_rank = rank
                found = True
                break
        
        if found:
            # If we found a lower rank, we can stop or try to go lower
            # Given time constraints, we might stop here
            pass

    U, V, W = best_result
    return {"U": U, "V": V, "W": W}
