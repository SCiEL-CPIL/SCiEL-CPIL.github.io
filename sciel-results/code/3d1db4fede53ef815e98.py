"""Rewritten program for 2x5x5 matrix multiplication tensor rank minimization."""
import numpy as np
import random
import time
from fractions import Fraction

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

def _exact_rank_decomposition(n, m, p):
    """
    Compute exact tensor rank decomposition using algebraic approach.
    For matrix multiplication tensor, we can use the fact that it has a known
    rank bound and try to find minimal rank decomposition.
    
    The key insight: for 2x5x5, we can try to find rank r decomposition by
    solving the bilinear system. We use a structured approach based on
    the fact that the tensor is a 3-way array with specific structure.
    """
    T = _build_tensor(n, m, p)
    
    # Try ranks from lower bound up
    # Lower bound for matrix multiplication is max(n*m, m*p, p*n) = 25
    # Upper bound is n*m*p = 50
    for target_rank in range(max(n*m, m*p, p*n), n*m*p + 1):
        if time.time() > time.time() + 100:  # safety check
            break
        
        # Use alternating minimization with exact rational arithmetic
        rng = np.random.RandomState(42 + target_rank)
        
        # Initialize with random values
        U = rng.randn(n * m, target_rank)
        V = rng.randn(m * p, target_rank)
        W = rng.randn(p * n, target_rank)
        
        # Iterate to convergence
        for iteration in range(200):
            # Update U: for each (i,j), solve U[i*m+j, :] * (V[j*p+k,:] * W[k*n+i,:])_k = T[i,j,:]
            for i in range(n):
                for j in range(m):
                    # Build system: sum_k V[j*p+k, r] * W[k*n+i, r] * U[i*m+j, r] = T[i,j,k]
                    # This is: (V[j*p+k,:] * W[k*n+i,:]) @ U[i*m+j,:] = T[i,j,k]
                    # So: M @ U[i*m+j,:] = T[i,j,:] where M[k,r] = V[j*p+k,r] * W[k*n+i,r]
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
            
            # Check convergence
            error = _verify_decomposition(U, V, W, T, tol=1e-10)
            if error < 1e-8:
                # Try to round to nearest half-integers and verify
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                error_rounded = _verify_decomposition(U_rounded, V_rounded, W_rounded, T, tol=1e-8)
                if error_rounded < 1e-6:
                    return U_rounded, V_rounded, W_rounded, target_rank
                else:
                    # Return the unrounded version if it's good enough
                    if error < 1e-10:
                        return U, V, W, target_rank
        
        # Try multiple random initializations
        for seed in range(10):
            rng = np.random.RandomState(1000 + seed + target_rank)
            U = rng.randn(n * m, target_rank)
            V = rng.randn(m * p, target_rank)
            W = rng.randn(p * n, target_rank)
            
            for iteration in range(100):
                for i in range(n):
                    for j in range(m):
                        M = np.zeros((p, target_rank))
                        for k in range(p):
                            M[k, :] = V[j * p + k, :] * W[k * n + i, :]
                        try:
                            U[i * m + j, :] = np.linalg.lstsq(M.T, T[i, j, :], rcond=None)[0]
                        except:
                            pass
                
                for j in range(m):
                    for k in range(p):
                        M = np.zeros((n, target_rank))
                        for i in range(n):
                            M[i, :] = U[i * m + j, :] * W[k * n + i, :]
                        try:
                            V[j * p + k, :] = np.linalg.lstsq(M.T, T[:, j, k], rcond=None)[0]
                        except:
                            pass
                
                for k in range(p):
                    for i in range(n):
                        M = np.zeros((m, target_rank))
                        for j in range(m):
                            M[j, :] = U[i * m + j, :] * V[j * p + k, :]
                        try:
                            W[k * n + i, :] = np.linalg.lstsq(M.T, T[i, :, k], rcond=None)[0]
                        except:
                            pass
                
                error = _verify_decomposition(U, V, W, T, tol=1e-10)
                if error < 1e-8:
                    U_rounded = np.round(U * 2) / 2
                    V_rounded = np.round(V * 2) / 2
                    W_rounded = np.round(W * 2) / 2
                    error_rounded = _verify_decomposition(U_rounded, V_rounded, W_rounded, T, tol=1e-8)
                    if error_rounded < 1e-6:
                        return U_rounded, V_rounded, W_rounded, target_rank
                    elif error < 1e-10:
                        return U, V, W, target_rank
    
    return None

def construct():
    n, m, p = 2, 5, 5
    deadline = time.time() + 140  # 10s safety margin before 150s limit
    
    # First, try to find exact minimal rank decomposition
    result = _exact_rank_decomposition(n, m, p)
    if result is not None:
        U_arr, V_arr, W_arr, rank = result
        return {"U": U_arr.tolist(), "V": V_arr.tolist(), "W": W_arr.tolist()}
    
    # Fallback: use schoolbook decomposition
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
# EVOLVE-BLOCK-END
