import numpy as np
import time
import random

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
    deadline = start_time + 140.0  # Safety margin: 150s limit, keep 10s buffer

    # Try to find a lower rank decomposition
    # Known lower bounds suggest rank is at least ~20 for 3x3x5
    target_ranks = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 35, 40, 44]
    for target_rank in target_ranks:
        if target_rank >= rank_sb:
            continue
        if time.time() > deadline - 5:
            break
        try:
            result = search_rank(n, m, p, target_rank, deadline)
            if result is not None:
                return result
        except Exception:
            pass

    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

def search_rank(n, m, p, rank, deadline):
    # Use structured initialization: start from a scaled version of schoolbook
    # and try to reduce rank
    
    # Initialize with small random half-integers, but with some structure
    np.random.seed(42)  # Reproducible for debugging
    
    # Try multiple random restarts
    best_result = None
    max_restarts = 5
    
    for restart in range(max_restarts):
        if time.time() > deadline - 10:
            break
            
        # Initialize matrices with random values in [-1, 1]
        U = np.random.uniform(-1, 1, size=(n * m, rank))
        V = np.random.uniform(-1, 1, size=(m * p, rank))
        W = np.random.uniform(-1, 1, size=(p * n, rank))
        
        # Snap to half-integers initially
        U = np.round(U * 2) / 2
        V = np.round(V * 2) / 2
        W = np.round(W * 2) / 2
        
        # Local search with gradient-like updates
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_error = compute_error(U, V, W, n, m, p)
        
        # Iterative improvement
        step_size = 0.5
        max_iters = 500
        no_improve_count = 0
        
        for iter in range(max_iters):
            if time.time() > deadline - 15:
                break
                
            # Try perturbing each matrix
            improved = False
            for mat_idx in range(3):
                if time.time() > deadline - 15:
                    break
                    
                # Pick random entries to perturb
                num_perturb = random.randint(1, 3)
                for _ in range(num_perturb):
                    if mat_idx == 0:
                        i, j = np.random.randint(0, n*m), np.random.randint(0, rank)
                        old_val = U[i, j]
                        for delta in [-step_size, step_size]:
                            U[i, j] = old_val + delta
                            new_error = compute_error(U, V, W, n, m, p)
                            if new_error < best_error:
                                best_error = new_error
                                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                                improved = True
                                no_improve_count = 0
                            else:
                                U[i, j] = old_val
                    elif mat_idx == 1:
                        i, j = np.random.randint(0, m*p), np.random.randint(0, rank)
                        old_val = V[i, j]
                        for delta in [-step_size, step_size]:
                            V[i, j] = old_val + delta
                            new_error = compute_error(U, V, W, n, m, p)
                            if new_error < best_error:
                                best_error = new_error
                                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                                improved = True
                                no_improve_count = 0
                            else:
                                V[i, j] = old_val
                    else:
                        i, j = np.random.randint(0, p*n), np.random.randint(0, rank)
                        old_val = W[i, j]
                        for delta in [-step_size, step_size]:
                            W[i, j] = old_val + delta
                            new_error = compute_error(U, V, W, n, m, p)
                            if new_error < best_error:
                                best_error = new_error
                                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                                improved = True
                                no_improve_count = 0
                            else:
                                W[i, j] = old_val
            
            if not improved:
                no_improve_count += 1
                if no_improve_count > 50:
                    break
            else:
                no_improve_count = 0
        
        # Check if we found an exact solution
        if best_error < 1e-10:
            U_snap = np.round(best_U * 2) / 2
            V_snap = np.round(best_V * 2) / 2
            W_snap = np.round(best_W * 2) / 2
            snap_error = compute_error(U_snap, V_snap, W_snap, n, m, p)
            if snap_error < 1e-10:
                return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
    
    return None

def compute_error(U, V, W, n, m, p):
    # Compute sum_{i,j,k} (sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] - 1)^2
    # Using einsum for efficiency
    
    U3 = U.reshape(n, m, -1)
    V3 = V.reshape(m, p, -1)
    W3 = W.reshape(p, n, -1)
    
    # T[i, j, k] = sum_r U3[i, j, r] * V3[j, k, r] * W3[k, i, r]
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    
    error = np.sum((T - 1.0)**2)
    return error
