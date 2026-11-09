import time
import numpy as np
import random
from fractions import Fraction

DIMENSIONS = [2, 5, 5]

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i*m+j, j*p+k, k*n+i] = 1."""
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def _verify(U, V, W, T, tol=1e-9):
    """Verify that sum_r U[:,r] V[:,r] W[:,r] == T."""
    n, m, p = T.shape[0]//m, m, p  # recover dims
    # Actually, let's just check directly
    recon = np.zeros_like(T)
    for r in range(U.shape[1]):
        recon += np.einsum('ir,jr,kr->ijk', U[:,r], V[:,r], W[:,r])
    return np.allclose(recon, T, atol=tol)

def _als_decomposition(T, rank, max_iters=50, tol=1e-6, seed=42):
    """ALS decomposition of tensor T into U, V, W."""
    rng = np.random.RandomState(seed)
    n, m, p = T.shape
    
    # Initialize
    U = rng.randn(n, rank)
    V = rng.randn(m, rank)
    W = rng.randn(p, rank)
    
    for it in range(max_iters):
        # Update U: for each i, solve (V^T W^T) U[:,i] = T[i,:,:].ravel()
        for i in range(n):
            # T[i,:,:] reshaped, we need sum_r U[i,r] V[j,r] W[k,r] = T[i,j,k]
            # For fixed i: sum_r U[i,r] * (V[:,r] * W[:,r]).ravel() = T[i,:,:].ravel()
            M = np.einsum('jr,kr->jkr', V, W).reshape(-1, rank)
            rhs = T[i,:,:].ravel()
            # Solve M @ U[i,:] = rhs
            U[i,:] = np.linalg.lstsq(M, rhs, rcond=None)[0]
        
        # Update V: for each j
        for j in range(m):
            M = np.einsum('ir,kr->ikr', U, W).reshape(-1, rank)
            rhs = T[:,j,:].ravel()
            V[j,:] = np.linalg.lstsq(M, rhs, rcond=None)[0]
        
        # Update W: for each k
        for k in range(p):
            M = np.einsum('ir,jr->ijr', U, V).reshape(-1, rank)
            rhs = T[:,:,k].ravel()
            W[k,:] = np.linalg.lstsq(M, rhs, rcond=None)[0]
        
        # Check convergence
        recon = np.einsum('ir,jr,kr->ijk', U, V, W)
        err = np.linalg.norm(recon - T) / (np.linalg.norm(T) + 1e-12)
        if err < tol:
            break
    
    return U, V, W, err

def _round_to_half_int(U, V, W, T, max_attempts=20):
    """Try to round U, V, W to half-integers while maintaining accuracy."""
    best_err = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for attempt in range(max_attempts):
        # Scale to make entries closer to half-integers
        # Try different scales
        scale = 1.0 + 0.1 * attempt
        
        U_r = np.round(U * scale) / scale
        V_r = np.round(V * scale) / scale
        W_r = np.round(W * scale) / scale
        
        # Check if entries are close to half-integers
        # Half-integers are multiples of 0.5
        U_check = np.round(U_r * 2) / 2
        V_check = np.round(V_r * 2) / 2
        W_check = np.round(W_r * 2) / 2
        
        recon = np.einsum('ir,jr,kr->ijk', U_check, V_check, W_check)
        err = np.linalg.norm(recon - T) / (np.linalg.norm(T) + 1e-12)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U_check.copy(), V_check.copy(), W_check.copy()
    
    return best_U, best_V, best_W, best_err

def construct():
    n, m, p = DIMENSIONS
    T = _build_tensor(n, m, p)
    
    # Start with schoolbook decomposition
    rank_schoolbook = n * m * p
    U_school = np.zeros((n*m, rank_schoolbook), dtype=np.float64)
    V_school = np.zeros((m*p, rank_schoolbook), dtype=np.float64)
    W_school = np.zeros((p*n, rank_schoolbook), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1.0
                V_school[j*p+k, col] = 1.0
                W_school[k*n+i, col] = 1.0
                col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_schoolbook
    
    # Try to find lower rank decompositions
    # Known lower bounds: rank >= max(n*m, m*p, p*n) = max(10, 25, 10) = 25
    # Try ranks from 25 up to 49
    
    deadline = time.time() + 140  # Leave 10s safety margin
    
    for rank in range(25, min(50, rank_schoolbook)):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        for seed in range(10):
            if time.time() > deadline:
                break
            
            U, V, W, err = _als_decomposition(T, rank, max_iters=30, tol=1e-8, seed=seed)
            
            if err < 1e-6:
                # Try to round to half-integers
                U_r, V_r, W_r, round_err = _round_to_half_int(U, V, W, T, max_attempts=5)
                
                if round_err < 1e-6:
                    best_U, best_V, best_W = U_r, V_r, W_r
                    best_rank = rank
                    break
                else:
                    # Use unrounded if it's still accurate enough
                    recon = np.einsum('ir,jr,kr->ijk', U, V, W)
                    if np.allclose(recon, T, atol=1e-6):
                        best_U, best_V, best_W = U, V, W
                        best_rank = rank
                        break
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
