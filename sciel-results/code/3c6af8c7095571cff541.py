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
    # recon[i,j,k] = sum_r Ur[i,j,r] * Vr[j,k,r] * Wr[k,i,r]
    recon = np.einsum('ijr,jkr,kir->ijk', Ur, Vr, Wr)
    return recon

def _verify_decomposition(U, V, W, T, tol=1e-10):
    """Check if U, V, W exactly decompose tensor T. Returns error value."""
    n, m, p = T.shape
    rank = U.shape[1]
    recon = _compute_reconstruction(U, V, W, n, m, p, rank)
    return np.max(np.abs(recon - T))

def _try_rank(n, m, p, target_rank, max_iterations=100, seed=42, deadline=None):
    """Try to find a decomposition of rank target_rank using alternating least squares."""
    T = _build_tensor(n, m, p)
    rng = np.random.RandomState(seed)
    
    best_error = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    for iteration in range(max_iterations):
        if deadline is not None and time.time() > deadline:
            break
        
        # Initialize randomly with small values to avoid numerical issues
        U = rng.randn(n * m, target_rank) * 0.1
        V = rng.randn(m * p, target_rank) * 0.1
        W = rng.randn(p * n, target_rank) * 0.1
        
        for inner_iter in range(50):
            if deadline is not None and time.time() > deadline:
                break
            
            # Update U: for each (i,j), solve linear system
            for i in range(n):
                for j in range(m):
                    # M[k,r] = V[j*p+k, r] * W[k*n+i, r]
                    # T[i,j,k] = sum_r U[i*m+j, r] * M[k, r]
                    # So: M.T @ U[i*m+j, :].T = T[i, j, :].T
                    M = np.zeros((p, target_rank))
                    for k in range(p):
                        M[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    
                    try:
                        U[i * m + j, :] = np.linalg.lstsq(M.T, T[i, j, :], rcond=None)[0]
                    except:
                        pass
            
            # Update V: for each (j,k), solve linear system
            for j in range(m):
                for k in range(p):
                    # M[i,r] = U[i*m+j, r] * W[k*n+i, r]
                    # T[i,j,k] = sum_r V[j*p+k, r] * M[i, r]
                    # So: M.T @ V[j*p+k, :].T = T[:, j, k].T
                    M = np.zeros((n, target_rank))
                    for i in range(n):
                        M[i, :] = U[i * m + j, :] * W[k * n + i, :]
                    
                    try:
                        V[j * p + k, :] = np.linalg.lstsq(M.T, T[:, j, k], rcond=None)[0]
                    except:
                        pass
            
            # Update W: for each (k,i), solve linear system
            for k in range(p):
                for i in range(n):
                    # M[j,r] = U[i*m+j, r] * V[j*p+k, r]
                    # T[i,j,k] = sum_r W[k*n+i, r] * M[j, r]
                    # So: M.T @ W[k*n+i, :].T = T[i, :, k].T
                    M = np.zeros((m, target_rank))
                    for j in range(m):
                        M[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    
                    try:
                        W[k * n + i, :] = np.linalg.lstsq(M.T, T[i, :, k], rcond=None)[0]
                    except:
                        pass
            
            # Check error
            curr_error = _verify_decomposition(U, V, W, T, tol=1e-10)
            
            if curr_error < 1e-10:
                return U, V, W
            
            if curr_error < best_error:
                best_error = curr_error
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
    
    if best_error < 1e-8:
        return best_U, best_V, best_W
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
    
    # Try to find lower rank decompositions, starting from rank-1 down
    # For 2x5x5, schoolbook is 50. We know rank is at least max(n*m, m*p, p*n) = 25
    # Try ranks from 49 down to 25
    for target_rank in range(n * m * p - 1, max(n * m * p - 15, 20), -1):
        if time.time() > deadline:
            break
        
        # Try a few different seeds
        for seed in [42, 123, 456, 789]:
            if time.time() > deadline:
                break
            
            result = _try_rank(n, m, p, target_rank, max_iterations=30, seed=seed, deadline=deadline)
            if result is not None:
                U_arr, V_arr, W_arr = result
                best_result = {"U": U_arr.tolist(), "V": V_arr.tolist(), "W": W_arr.tolist()}
                best_rank = target_rank
                break
        
        if best_rank == target_rank:
            break
    
    return best_result
# EVOLVE-BLOCK-END
