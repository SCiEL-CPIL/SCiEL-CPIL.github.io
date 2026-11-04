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
    deadline = start_time + 140.0

    # Try to find a lower rank decomposition
    target_ranks = [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 35, 40, 44]
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
    Search for a rank-r decomposition using gradient-based local search.
    """
    best_result = None
    best_error = float('inf')
    
    max_restarts = 50
    for restart in range(max_restarts):
        if time.time() > deadline - 15:
            break
        
        # Initialize with small random values
        init_type = restart % 4
        if init_type == 0:
            U = np.random.randn(n * m, rank) * 0.1
            V = np.random.randn(m * p, rank) * 0.1
            W = np.random.randn(p * n, rank) * 0.1
        elif init_type == 1:
            U = np.random.randn(n * m, rank) * 0.5
            V = np.random.randn(m * p, rank) * 0.5
            W = np.random.randn(p * n, rank) * 0.5
        elif init_type == 2:
            # Structured: try to mimic Strassen-like patterns
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            for r in range(rank):
                for i in range(n):
                    for j in range(m):
                        U[i * m + j, r] = np.random.choice([-1, 0, 1])
                for j in range(m):
                    for k in range(p):
                        V[j * p + k, r] = np.random.choice([-1, 0, 1])
                for k in range(p):
                    for i in range(n):
                        W[k * n + i, r] = np.random.choice([-1, 0, 1])
        else:
            U = np.random.randint(-1, 2, size=(n * m, rank)).astype(float)
            V = np.random.randint(-1, 2, size=(m * p, rank)).astype(float)
            W = np.random.randint(-1, 2, size=(p * n, rank)).astype(float)
        
        # Gradient-based local search
        U, V, W, error = gradient_local_search(U, V, W, n, m, p, rank, deadline)
        
        if error < best_error:
            best_error = error
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        # Check if we found an exact solution
        if error < 1e-10:
            U_snap = np.round(U * 2) / 2
            V_snap = np.round(V * 2) / 2
            W_snap = np.round(W * 2) / 2
            snap_error = compute_error(U_snap, V_snap, W_snap, n, m, p)
            if snap_error < 1e-10:
                return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
    
    # Try snapping the best solution
    if best_error < 1e-6:
        U_snap = np.round(best_U * 2) / 2
        V_snap = np.round(best_V * 2) / 2
        W_snap = np.round(best_W * 2) / 2
        snap_error = compute_error(U_snap, V_snap, W_snap, n, m, p)
        if snap_error < 1e-10:
            return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
    
    return None

def gradient_local_search(U, V, W, n, m, p, rank, deadline, max_iters=2000):
    """
    Gradient-based local search with adaptive step size and second-order correction.
    """
    best_error = compute_error(U, V, W, n, m, p)
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    current_error = best_error
    lr = 0.01
    patience = 50
    no_improve = 0
    
    for it in range(max_iters):
        if time.time() > deadline - 12:
            break
        
        # Compute gradients
        U3 = U.reshape(n, m, -1)
        V3 = V.reshape(m, p, -1)
        W3 = W.reshape(p, n, -1)
        
        T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        residual = T - 1.0
        
        # Gradient for U: dU[i,j,r] = 2 * sum_k residual[i,j,k] * V[j,k,r] * W[k,i,r]
        dU = 2.0 * np.einsum('ijk,jkr,kir->ijr', residual, V3, W3)
        # Gradient for V: dV[j,k,r] = 2 * sum_i residual[i,j,k] * U[i,j,r] * W[k,i,r]
        dV = 2.0 * np.einsum('ijk,ijr,kir->jkr', residual, U3, W3)
        # Gradient for W: dW[k,i,r] = 2 * sum_j residual[i,j,k] * U[i,j,r] * V[j,k,r]
        dW = 2.0 * np.einsum('ijk,ijr,jkr->kir', residual, U3, V3)
        
        # Compute approximate Hessian diagonal for Newton-like correction
        # H_ii ≈ 2 * sum over other indices of (product of other two factors)^2
        H_U = 2.0 * np.einsum('jkr,kir->ijr', V3**2, W3**2)
        H_V = 2.0 * np.einsum('ijr,kir->jkr', U3**2, W3**2)
        H_W = 2.0 * np.einsum('ijr,jkr->kir', U3**2, V3**2)
        
        # Newton step: step = grad / (H_diag + epsilon)
        eps = 1e-6
        newton_U = dU / (H_U + eps)
        newton_V = dV / (H_V + eps)
        newton_W = dW / (H_W + eps)
        
        # Blend gradient and Newton steps
        blend = 0.5 if it % 2 == 0 else 1.0
        step_U = blend * newton_U + (1 - blend) * (dU / (np.sqrt(np.sum(dU**2)) + eps) * lr)
        step_V = blend * newton_V + (1 - blend) * (dV / (np.sqrt(np.sum(dV**2)) + eps) * lr)
        step_W = blend * newton_W + (1 - blend) * (dW / (np.sqrt(np.sum(dW**2)) + eps) * lr)
        
        # Clamp step size to prevent divergence
        max_step = 0.5
        for step in [step_U, step_V, step_W]:
            step_norm = np.sqrt(np.sum(step**2))
            if step_norm > max_step:
                step *= max_step / step_norm
        
        U_new = U - step_U
        V_new = V - step_V
        W_new = W - step_W
        
        new_error = compute_error(U_new, V_new, W_new, n, m, p)
        
        if new_error < current_error:
            U, V, W = U_new, V_new, W_new
            current_error = new_error
            no_improve = 0
            if new_error < best_error:
                best_error = new_error
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                # Decrease learning rate on improvement
                lr *= 0.95
            # Try to snap if error is very small
            if current_error < 1e-8:
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                snap_error = compute_error(U_snap, V_snap, W_snap, n, m, p)
                if snap_error < 1e-10:
                    return U_snap, V_snap, W_snap, snap_error
        else:
            no_improve += 1
            lr *= 0.9
            if no_improve > patience:
                # Random restart within local search
                U = best_U + np.random.randn(n * m, rank) * 0.05
                V = best_V + np.random.randn(m * p, rank) * 0.05
                W = best_W + np.random.randn(p * n, rank) * 0.05
                current_error = compute_error(U, V, W, n, m, p)
                no_improve = 0
                lr = 0.01
        
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
    
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    
    error = np.sum((T - 1.0)**2)
    return error
