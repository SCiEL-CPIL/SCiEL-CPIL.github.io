"""Rewritten program for matrix multiplication tensor decomposition.
Uses a search-based approach to find a rank-minimal decomposition,
falling back to schoolbook if no improvement is found.
"""

import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 8]

def _build_tensor(n, m, p):
    """Build the 3D matrix multiplication tensor T[i*m+j, j*p+k, k*n+i] = 1."""
    tensor = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                tensor[i * m + j, j * p + k, k * n + i] = 1.0
    return tensor

def _verify_decomposition(U, V, W, T, tol=1e-6):
    """Check if U, V, W exactly decompose T: sum_c U[:,c] V[:,c]^T W[:,c]^T = T."""
    n, m, p = T.shape[0] // T.shape[1] // 0 if False else (3, 4, 8)
    # Actually, let's just check the reconstruction
    # T[i*m+j, j*p+k, k*n+i] should equal sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    rank = U.shape[1]
    for c in range(rank):
        pass  # We'll do a vectorized check instead
    # Vectorized: for each c, compute outer product contribution
    # This is slow for large rank, so do it efficiently
    # T_reconstructed = sum_c U[:,c] * V[:,c] * W[:,c]
    # In numpy: einsum
    T_rec = np.einsum('ic,jc,kc->ijk', U, V, W)
    return np.allclose(T_rec, T, atol=tol)

def _schoolbook(n, m, p):
    """Return schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
                c += 1
    return U, V, W, rank

def _try_rank(n, m, p, target_rank, max_iters=5000, seed=42, timeout_s=30.0):
    """
    Attempt to find a rank-`target_rank` decomposition using iterative
    multiplicative update / ALS-like approach. Returns (U, V, W) or None.
    """
    T = _build_tensor(n, m, p)
    U_dim = n * m
    V_dim = m * p
    W_dim = p * n
    
    start_time = time.time()
    rng = np.random.RandomState(seed)
    
    best_err = float('inf')
    best_U = best_V = best_W = None
    
    for it in range(max_iters):
        if time.time() - start_time > timeout_s:
            break
        
        # Initialize randomly
        U = rng.randn(U_dim, target_rank)
        V = rng.randn(V_dim, target_rank)
        W = rng.randn(W_dim, target_rank)
        
        # Iterative refinement: alternate solving for each factor
        # For each c, the contribution is U[:,c] V[:,c]^T W[:,c]^T
        # We want sum_c U[:,c] V[:,c]^T W[:,c]^T = T
        
        # Use a simple gradient-free approach: random restarts with 
        # least-squares refinement per column
        
        # Actually, let's try a different approach: treat it as a 
        # low-rank tensor completion. For each c, we want to find
        # u_c, v_c, w_c such that the sum of outer products equals T.
        
        # Simplified: use random initialization and then do a few
        # rounds of "update one factor" via least squares.
        
        for inner in range(20):
            if time.time() - start_time > timeout_s:
                break
            
            # Update U: for each i*m+j, solve for U[i*m+j, :] given V, W
            # T[i*m+j, :, :] = sum_c U[i*m+j, c] * (V[:, c] * W[:, c])
            # This is linear in U[i*m+j, :] for each (i,j)
            # Build the matrix: for each (i,j), the equation is
            #   T[i*m+j, a, b] = sum_c U[i*m+j, c] * V[a, c] * W[b, c]
            # for all (a, b). This is a linear system in U[i*m+j, :].
            # Let M[a*b, c] = V[a, c] * W[b, c] (Kronecker-like)
            # Then T_flat[i*m+j, a*b] = U[i*m+j, :] @ M
            # So U[i*m+j, :] = pinv(M) @ T_flat[i*m+j, :]
            
            M = np.zeros((V_dim * W_dim, target_rank))
            for a in range(V_dim):
                for b in range(W_dim):
                    M[a * W_dim + b, :] = V[a, :] * W[b, :]
            
            M_pinv = np.linalg.pinv(M)
            
            for ij in range(U_dim):
                T_flat = T[ij, :, :].reshape(-1)
                U[ij, :] = M_pinv @ T_flat
            
            # Similarly update V
            # T[i*m+j, a, b] = sum_c V[a, c] * (U[i*m+j, c] * W[b, c])
            # For fixed a: T[?, a, ?] = sum_c V[a, c] * (U[:, c] * W[b, c])
            # Build M2 for V: M2[ij*W_dim + b, c] = U[ij, c] * W[b, c]
            M2 = np.zeros((U_dim * W_dim, target_rank))
            for ij in range(U_dim):
                for b in range(W_dim):
                    M2[ij * W_dim + b, :] = U[ij, :] * W[b, :]
            M2_pinv = np.linalg.pinv(M2)
            
            for a in range(V_dim):
                T_flat = T[:, a, :].reshape(-1)
                V[a, :] = M2_pinv @ T_flat
            
            # Similarly update W
            # T[i*m+j, a, b] = sum_c W[b, c] * (U[i*m+j, c] * V[a, c])
            # For fixed b: T[:, :, b] = sum_c W[b, c] * (U[:, c] * V[a, c])
            M3 = np.zeros((U_dim * V_dim, target_rank))
            for ij in range(U_dim):
                for a in range(V_dim):
                    M3[ij * V_dim + a, :] = U[ij, :] * V[a, :]
            M3_pinv = np.linalg.pinv(M3)
            
            for b in range(W_dim):
                T_flat = T[:, :, b].reshape(-1)
                W[b, :] = M3_pinv @ T_flat
            
            # Check error
            T_rec = np.einsum('ic,jc,kc->ijk', U, V, W)
            err = np.max(np.abs(T_rec - T))
            if err < best_err:
                best_err = err
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
            
            if err < 1e-6:
                break
        
        if best_err < 1e-6:
            break
    
    if best_err < 1e-6:
        return best_U, best_V, best_W
    return None

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x4 by 4x8 matrix multiplication.
    Returns {'U', 'V', 'W'} with integer/half-integer entries.
    """
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    U0, V0, W0, rank0 = _schoolbook(n, m, p)
    best_rank = rank0
    best_U, best_V, best_W = U0, V0, W0
    
    # Try to find lower ranks using search
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(12, 32, 24) = 32
    # But actual rank is likely 96 (schoolbook) for these dimensions
    
    start_time = time.time()
    deadline = start_time + 100.0  # Leave 20s safety margin
    
    # Try a few target ranks below schoolbook
    for target_rank in [best_rank - 8, best_rank - 4, best_rank - 2]:
        if target_rank < 32:  # Below theoretical bound
            continue
        if time.time() > deadline:
            break
        
        result = _try_rank(n, m, p, target_rank, 
                          max_iters=200, 
                          seed=42, 
                          timeout_s=30.0)
        if result is not None:
            U, V, W = result
            if _verify_decomposition(U, V, W, _build_tensor(n, m, p), tol=1e-6):
                best_rank = target_rank
                best_U, best_V, best_W = U, V, W
                break  # Found improvement, stop searching
    
    # Convert to lists for return (the interface expects lists of lists)
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    # Round to nearest half-integer for cleanliness
    def _to_half_integer(x):
        rounded = round(2 * x) / 2
        return rounded
    
    U_list = [[_to_half_integer(v) for v in row] for row in U_list]
    V_list = [[_to_half_integer(v) for v in row] for row in V_list]
    W_list = [[_to_half_integer(v) for v in row] for row in W_list]
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
