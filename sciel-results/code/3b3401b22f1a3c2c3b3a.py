import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1

    start_time = time.time()
    deadline = start_time + 115.0

    # Try to find a lower rank decomposition
    target_ranks = [25, 26, 27, 28, 29, 30, 32, 35, 40, 44]
    for target_rank in target_ranks:
        if target_rank >= rank_sb:
            continue
        if time.time() > deadline - 10:
            break
        try:
            result = search_rank(n, m, p, target_rank, deadline)
            if result is not None:
                return result
        except Exception:
            pass

    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

def search_rank(n, m, p, rank, deadline):
    """
    Search for a rank-r decomposition of the matrix multiplication tensor.
    Uses multiple strategies: structured initialization, random restarts, and local search.
    """
    best_result = None
    best_error = float('inf')
    
    # Strategy 1: Try structured initialization based on known decompositions
    # For 3x3x5, try to find patterns that might work
    
    # Strategy 2: Random restarts with local search
    max_restarts = 50
    restarts_done = 0
    
    for restart in range(max_restarts):
        if time.time() > deadline - 15:
            break
            
        # Initialize with small random values (integers or half-integers)
        init_type = restart % 3
        if init_type == 0:
            # Small integers
            U = np.random.randint(-1, 2, size=(n * m, rank)).astype(float)
            V = np.random.randint(-1, 2, size=(m * p, rank)).astype(float)
            W = np.random.randint(-1, 2, size=(p * n, rank)).astype(float)
        elif init_type == 1:
            # Half-integers
            U = np.random.randint(-2, 3, size=(n * m, rank)) * 0.5
            V = np.random.randint(-2, 3, size=(m * p, rank)) * 0.5
            W = np.random.randint(-2, 3, size=(p * n, rank)) * 0.5
        else:
            # Small random reals
            U = np.random.randn(n * m, rank) * 0.5
            V = np.random.randn(m * p, rank) * 0.5
            W = np.random.randn(p * n, rank) * 0.5
        
        # Local search
        U, V, W, error = local_search(U, V, W, n, m, p, rank, deadline)
        
        if error < best_error:
            best_error = error
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        # Check if we found an exact solution
        if error < 1e-10:
            # Snap to half-integers
            U_snap = np.round(U * 2) / 2
            V_snap = np.round(V * 2) / 2
            W_snap = np.round(W * 2) / 2
            snap_error = compute_error(U_snap, V_snap, W_snap, n, m, p)
            if snap_error < 1e-10:
                return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
        
        restarts_done += 1
    
    # If we found a good float solution but didn't snap, try snapping the best
    if best_error < 1e-6:
        U_snap = np.round(best_U * 2) / 2
        V_snap = np.round(best_V * 2) / 2
        W_snap = np.round(best_W * 2) / 2
        snap_error = compute_error(U_snap, V_snap, W_snap, n, m, p)
        if snap_error < 1e-10:
            return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
    
    return None

def local_search(U, V, W, n, m, p, rank, deadline, max_iters=500):
    """
    Perform local search to minimize the Frobenius norm error.
    Uses simulated annealing style with adaptive step size.
    """
    best_error = compute_error(U, V, W, n, m, p)
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    current_error = best_error
    step_size = 0.5
    temp = 1.0
    cooling = 0.95
    
    for iter in range(max_iters):
        if time.time() > deadline - 12:
            break
        
        # Random perturbation
        mat_idx = np.random.randint(0, 3)
        delta = np.random.randn() * step_size
        
        if mat_idx == 0:
            i, j = np.random.randint(0, n*m), np.random.randint(0, rank)
            old_val = U[i, j]
            U[i, j] += delta
            new_error = compute_error(U, V, W, n, m, p)
            if new_error < current_error or np.random.rand() < np.exp(-(new_error - current_error) / max(temp, 1e-10)):
                current_error = new_error
                if new_error < best_error:
                    best_error = new_error
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            else:
                U[i, j] = old_val
        elif mat_idx == 1:
            i, j = np.random.randint(0, m*p), np.random.randint(0, rank)
            old_val = V[i, j]
            V[i, j] += delta
            new_error = compute_error(U, V, W, n, m, p)
            if new_error < current_error or np.random.rand() < np.exp(-(new_error - current_error) / max(temp, 1e-10)):
                current_error = new_error
                if new_error < best_error:
                    best_error = new_error
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            else:
                V[i, j] = old_val
        else:
            i, j = np.random.randint(0, p*n), np.random.randint(0, rank)
            old_val = W[i, j]
            W[i, j] += delta
            new_error = compute_error(U, V, W, n, m, p)
            if new_error < current_error or np.random.rand() < np.exp(-(new_error - current_error) / max(temp, 1e-10)):
                current_error = new_error
                if new_error < best_error:
                    best_error = new_error
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            else:
                W[i, j] = old_val
        
        # Cool down
        temp *= cooling
        
        # If error is very small, try to snap
        if current_error < 1e-8:
            U_snap = np.round(U * 2) / 2
            V_snap = np.round(V * 2) / 2
            W_snap = np.round(W * 2) / 2
            snap_error = compute_error(U_snap, V_snap, W_snap, n, m, p)
            if snap_error < 1e-10:
                return U_snap, V_snap, W_snap, snap_error
        
        # Early exit if we found a good solution
        if best_error < 1e-12:
            break
    
    return best_U, best_V, best_W, best_error

def compute_error(U, V, W, n, m, p):
    """
    Compute the Frobenius norm squared of the difference between the decomposed tensor
    and the target tensor (all ones).
    
    T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    Target: T[i, j, k] = 1 for all i, j, k
    """
    U3 = U.reshape(n, m, -1)
    V3 = V.reshape(m, p, -1)
    W3 = W.reshape(p, n, -1)
    
    # T[i, j, k] = sum_r U3[i, j, r] * V3[j, k, r] * W3[k, i, r]
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    
    error = np.sum((T - 1.0)**2)
    return error
