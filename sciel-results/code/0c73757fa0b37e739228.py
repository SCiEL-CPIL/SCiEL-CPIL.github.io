"""Rewritten program for 2x5x5 matrix multiplication tensor rank minimization."""
import numpy as np
import random
import time

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i,j,k] = 1 if output (i,k) depends on input (i,j) and (j,k)."""
    T = np.ones((n, m, p))
    return T

def _compute_reconstruction(U, V, W, n, m, p, rank):
    """Compute reconstruction T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]"""
    Ur = U.reshape(n, m, rank)
    Vr = V.reshape(m, p, rank)
    Wr = W.reshape(p, n, rank)
    
    # Use einsum for efficient computation
    recon = np.einsum('ijr,jkr,kir->ijk', Ur, Vr, Wr)
    return recon

def _verify_decomposition(U, V, W, T, tol=1e-10):
    """Check if U, V, W exactly decompose tensor T. Returns error value."""
    n, m, p = T.shape
    rank = U.shape[1]
    recon = _compute_reconstruction(U, V, W, n, m, p, rank)
    return np.max(np.abs(recon - T))

def _flatten_rank_check(n, m, p, target_rank):
    """
    Necessary condition: for each flattening, the matrix rank must be <= target_rank.
    Flattening 1: T_{i,jk} -> matrix (n) x (m*p), rank <= target_rank
    Flattening 2: T_{j,ik} -> matrix (m) x (n*p), rank <= target_rank  
    Flattening 3: T_{k,ij} -> matrix (p) x (n*m), rank <= target_rank
    """
    T = _build_tensor(n, m, p)
    
    # Flattening 1: fix i, matrix is (m*p) x (n) but we need (n) x (m*p)
    T1 = T.reshape(n, m * p)
    r1 = np.linalg.matrix_rank(T1)
    
    # Flattening 2: fix j, matrix is (n*p) x (m) -> (m) x (n*p)
    T2 = np.transpose(T, (1, 0, 2)).reshape(m, n * p)
    r2 = np.linalg.matrix_rank(T2)
    
    # Flattening 3: fix k, matrix is (n*m) x (p) -> (p) x (n*m)
    T3 = np.transpose(T, (2, 0, 1)).reshape(p, n * m)
    r3 = np.linalg.matrix_rank(T3)
    
    return min(target_rank >= r1, target_rank >= r2, target_rank >= r3)

def _optimize_smooth(n, m, p, target_rank, seed=42, deadline=None, max_iter=500):
    """
    Smooth L2 optimization with Adam-like updates.
    Minimizes sum_{i,j,k} (recon[i,j,k] - T[i,j,k])^2
    """
    T = _build_tensor(n, m, p)
    rng = np.random.RandomState(seed)
    
    U = rng.randn(n * m, target_rank) * 0.1
    V = rng.randn(m * p, target_rank) * 0.1
    W = rng.randn(p * n, target_rank) * 0.1
    
    # Adam parameters
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0
    m_U = np.zeros_like(U); v_U = np.zeros_like(U)
    m_V = np.zeros_like(V); v_V = np.zeros_like(V)
    m_W = np.zeros_like(W); v_W = np.zeros_like(W)
    lr = 0.05
    
    best_error = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        t += 1
        
        # Compute reconstruction
        Ur = U.reshape(n, m, target_rank)
        Vr = V.reshape(m, p, target_rank)
        Wr = W.reshape(p, n, target_rank)
        recon = np.einsum('ijr,jkr,kir->ijk', Ur, Vr, Wr)
        
        # L2 error
        diff = recon - T
        l2_error = np.sum(diff ** 2)
        
        if l2_error < 1e-20:
            break
        
        # Gradients (smooth)
        # dL/dU[i*m+j, r] = 2 * sum_k diff[i,j,k] * V[j*p+k, r] * W[k*n+i, r]
        grad_U = np.zeros((n * m, target_rank))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    grad_U[i * m + j, :] += 2.0 * diff[i, j, k] * V[j * p + k, :] * W[k * n + i, :]
        
        grad_V = np.zeros((m * p, target_rank))
        for j in range(m):
            for k in range(p):
                for i in range(n):
                    grad_V[j * p + k, :] += 2.0 * diff[i, j, k] * U[i * m + j, :] * W[k * n + i, :]
        
        grad_W = np.zeros((p * n, target_rank))
        for k in range(p):
            for i in range(n):
                for j in range(m):
                    grad_W[k * n + i, :] += 2.0 * diff[i, j, k] * U[i * m + j, :] * V[j * p + k, :]
        
        # Adam update
        m_U = beta1 * m_U + (1 - beta1) * grad_U
        v_U = beta2 * v_U + (1 - beta2) * grad_U ** 2
        m_U_hat = m_U / (1 - beta1 ** t)
        v_U_hat = v_U / (1 - beta2 ** t)
        U -= lr * m_U_hat / (np.sqrt(v_U_hat) + eps)
        
        m_V = beta1 * m_V + (1 - beta1) * grad_V
        v_V = beta2 * v_V + (1 - beta2) * grad_V ** 2
        m_V_hat = m_V / (1 - beta1 ** t)
        v_V_hat = v_V / (1 - beta2 ** t)
        V -= lr * m_V_hat / (np.sqrt(v_V_hat) + eps)
        
        m_W = beta1 * m_W + (1 - beta1) * grad_W
        v_W = beta2 * v_W + (1 - beta2) * grad_W ** 2
        m_W_hat = m_W / (1 - beta1 ** t)
        v_W_hat = v_W / (1 - beta2 ** t)
        W -= lr * m_W_hat / (np.sqrt(v_W_hat) + eps)
        
        # Check
        max_err = np.max(np.abs(recon - T))
        if max_err < 1e-12:
            return U, V, W
        
        if l2_error < best_error:
            best_error = l2_error
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Final check
    recon = _compute_reconstruction(best_U, best_V, best_W, n, m, p, target_rank)
    if np.max(np.abs(recon - T)) < 1e-10:
        return best_U, best_V, best_W
    return None

def _als_refine(n, m, p, U, V, W, max_iter=100, deadline=None):
    """ALS refinement from a starting point."""
    T = _build_tensor(n, m, p)
    target_rank = U.shape[1]
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Update U
        for i in range(n):
            for j in range(m):
                M = np.zeros((p, target_rank))
                for k in range(p):
                    M[k, :] = V[j * p + k, :] * W[k * n + i, :]
                try:
                    U[i * m + j, :] = np.linalg.lstsq(M.T, T[i, j, :], rcond=None)[0]
                except:
                    pass
        
        # Update V
        for j in range(m):
            for k in range(p):
                M = np.zeros((n, target_rank))
                for i in range(n):
                    M[i, :] = U[i * m + j, :] * W[k * n + i, :]
                try:
                    V[j * p + k, :] = np.linalg.lstsq(M.T, T[:, j, k], rcond=None)[0]
                except:
                    pass
        
        # Update W
        for k in range(p):
            for i in range(n):
                M = np.zeros((m, target_rank))
                for j in range(m):
                    M[j, :] = U[i * m + j, :] * V[j * p + k, :]
                try:
                    W[k * n + i, :] = np.linalg.lstsq(M.T, T[i, :, k], rcond=None)[0]
                except:
                    pass
        
        recon = _compute_reconstruction(U, V, W, n, m, p, target_rank)
        if np.max(np.abs(recon - T)) < 1e-12:
            return U, V, W
    
    recon = _compute_reconstruction(U, V, W, n, m, p, target_rank)
    if np.max(np.abs(recon - T)) < 1e-10:
        return U, V, W
    return None

def construct():
    n, m, p = 2, 5, 5
    deadline = time.time() + 130  # 20s safety margin before 150s limit
    
    # Start with schoolbook
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
    
    best_result = {"U": U, "V": V, "W": W}
    best_rank = rank
    
    # For 2x5x5, try ranks from 49 down to 25
    for target_rank in range(n * m * p - 1, max(n * m * p - 25, 20), -1):
        if time.time() > deadline:
            break
        
        # Quick necessary condition check
        if not _flatten_rank_check(n, m, p, target_rank):
            continue
        
        found = False
        
        # Try smooth optimization with multiple seeds
        for seed in [42, 123, 456, 789, 1000, 2024]:
            if time.time() > deadline:
                break
            
            result = _optimize_smooth(n, m, p, target_rank, seed=seed, deadline=deadline, max_iter=300)
            if result is not None:
                U_arr, V_arr, W_arr = result
                best_result = {"U": U_arr.tolist(), "V": V_arr.tolist(), "W": W_arr.tolist()}
                best_rank = target_rank
                found = True
                break
        
        if found:
            break
        
        # Try ALS from random starts
        rng = np.random.RandomState(42)
        for seed in [42, 123, 456]:
            if time.time() > deadline:
                break
            
            U_init = rng.randn(n * m, target_rank) * 0.1
            V_init = rng.randn(m * p, target_rank) * 0.1
            W_init = rng.randn(p * n, target_rank) * 0.1
            
            result = _als_refine(n, m, p, U_init, V_init, W_init, max_iter=50, deadline=deadline)
            if result is not None:
                U_arr, V_arr, W_arr = result
                best_result = {"U": U_arr.tolist(), "V": V_arr.tolist(), "W": W_arr.tolist()}
                best_rank = target_rank
                break
        
        if best_rank == target_rank:
            break
    
    return best_result
# EVOLVE-BLOCK-END
