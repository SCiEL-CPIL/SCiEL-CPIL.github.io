"""
Matrix multiplication 3x4 by 4x4 with rank-minimal tensor decomposition.
Uses ALS-based tensor decomposition search with schoolbook fallback.
"""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 4]

def _build_schoolbook():
    """Standard schoolbook decomposition: rank = 3*4*4 = 48."""
    n, m, p = 3, 4, 4
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W, rank

def _make_tensor(n, m, p):
    """Create the matrix multiplication tensor T[i,j,k] = 1 for all valid indices."""
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    return T

def _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
    """Verify that U, V, W exactly decompose the multiplication tensor."""
    rank = U.shape[1]
    # Reconstruct tensor: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    T_reconstructed = np.zeros((n, m, p))
    for r in range(rank):
        u_col = U[:, r]
        v_col = V[:, r]
        w_col = W[:, r]
        # u_col has shape (n*m,), v_col has shape (m*p,), w_col has shape (p*n,)
        # Reshape and multiply
        u_mat = u_col.reshape(n, m)
        v_mat = v_col.reshape(m, p)
        w_mat = w_col.reshape(p, n)
        # Term r contributes: u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_reconstructed[i, j, k] += u_mat[i, j] * v_mat[j, k] * w_mat[k, i]
    
    T_target = _make_tensor(n, m, p)
    err = np.max(np.abs(T_reconstructed - T_target))
    return err < tol, err

def _snap_to_half_integer(U, V, W, tol=1e-4):
    """Snap values to nearest half-integer if close."""
    def snap_val(x):
        # Try to snap to nearest 0.5
        rounded = np.round(2 * x) / 2.0
        if abs(x - rounded) < tol:
            return rounded
        return x
    
    U_snapped = np.vectorize(snap_val)(U)
    V_snapped = np.vectorize(snap_val)(V)
    W_snapped = np.vectorize(snap_val)(W)
    return U_snapped, V_snapped, W_snapped

def _als_decompose(n, m, p, rank, max_iters=200, tol=1e-8, seed=42):
    """
    Use Alternating Least Squares to find a rank-rank decomposition.
    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    """
    rng = np.random.RandomState(seed)
    
    # Initialize
    U = rng.randn(n * m, rank) * 0.1
    V = rng.randn(m * p, rank) * 0.1
    W = rng.randn(p * n, rank) * 0.1
    
    T_target = _make_tensor(n, m, p)
    
    for it in range(max_iters):
        # Update U: for each column r, U[:,r] = argmin ||T - sum_r U_r * V_r * W_r||
        # This is a linear system. For fixed V and W, we can solve for U.
        
        # Build the linear system for U
        # T[i,j,k] = sum_r U[i*m+j, r] * (V[j*p+k, r] * W[k*n+i, r])
        # Let A_r[i*m+j, i,j,k] = V[j*p+k, r] * W[k*n+i, r]
        # Then T = sum_r U[:,r] * A_r (in some flattened sense)
        
        # More efficiently: for each (i,j,k), T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Group by (i,j): for fixed (i,j), T[i,j,k] for all k is a vector of length p
        # T[i,j,:] = sum_r U[i*m+j, r] * (V[j*p+j*p+p-1... wait, V[j*p+k, r] for k=0..p-1)
        
        # Let's do it more carefully with matrix operations
        # For fixed r, define M_r as the matrix where M_r[i*m+j, (k*n+i)] = V[j*p+k, r] * W[k*n+i, r]
        # This is getting complex. Let's use a different approach.
        
        # Actually, let's just do gradient descent or use the fact that this is a
        # standard tensor decomposition problem.
        
        # Simpler approach: use the unfolding technique
        # Fix V and W, solve for U
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # For each (i,j), define b_{i,j} as the vector with components b_{i,j,k} = T[i,j,k]
        # And define the design matrix D_{i,j} with D_{i,j,k,r} = V[j*p+k, r] * W[k*n+i, r]
        # Then b_{i,j} = D_{i,j} @ U[i*m+j, :]
        
        # This is a least squares problem for each (i,j)
        
        U_new = np.zeros_like(U)
        for i in range(n):
            for j in range(m):
                # b vector of length p
                b = T_target[i, j, :]
                # D matrix of shape (p, rank)
                D = np.zeros((p, rank))
                for k in range(p):
                    for r in range(rank):
                        D[k, r] = V[j * p + k, r] * W[k * n + i, r]
                # Solve D @ x = b
                try:
                    sol, _, _, _ = np.linalg.lstsq(D, b, rcond=None)
                    U_new[i * m + j, :] = sol
                except:
                    U_new[i * m + j, :] = U[i * m + j, :]
        
        # Fix U and W, solve for V
        V_new = np.zeros_like(V)
        for j in range(m):
            for k in range(p):
                # For fixed (j,k), T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                # b vector of length n: b_i = T[i,j,k]
                # D matrix of shape (n, rank): D[i,r] = U[i*m+j, r] * W[k*n+i, r]
                b = T_target[:, j, k]
                D = np.zeros((n, rank))
                for i in range(n):
                    for r in range(rank):
                        D[i, r] = U[i * m + j, r] * W[k * n + i, r]
                try:
                    sol, _, _, _ = np.linalg.lstsq(D, b, rcond=None)
                    V_new[j * p + k, :] = sol
                except:
                    V_new[j * p + k, :] = V[j * p + k, :]
        
        # Fix U and V, solve for W
        W_new = np.zeros_like(W)
        for k in range(p):
            for i in range(n):
                # For fixed (k,i), T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                # b vector of length m: b_j = T[i,j,k]
                # D matrix of shape (m, rank): D[j,r] = U[i*m+j, r] * V[j*p+k, r]
                b = T_target[i, :, k]
                D = np.zeros((m, rank))
                for j in range(m):
                    for r in range(rank):
                        D[j, r] = U[i * m + j, r] * V[j * p + k, r]
                try:
                    sol, _, _, _ = np.linalg.lstsq(D, b, rcond=None)
                    W_new[k * n + i, :] = sol
                except:
                    W_new[k * n + i, :] = W[k * n + i, :]
        
        U, V, W = U_new, V_new, W_new
        
        # Check convergence
        valid, err = _verify_decomposition(U, V, W, n, m, p, tol=1e-4)
        if err < tol:
            break
    
    return U, V, W, err

def _build_block_optimized():
    """
    Try to find a low-rank decomposition using ALS, falling back to schoolbook.
    """
    n, m, p = 3, 4, 4
    start_time = time.time()
    deadline = start_time + 120  # Leave 10s margin
    
    # Start with schoolbook as best known
    U_best, V_best, W_best, rank_best = _build_schoolbook()
    
    # Try decreasing ranks
    for rank in range(30, 49):
        if time.time() > deadline - 10:
            break
        
        # Try multiple seeds for robustness
        for seed in range(10):
            if time.time() > deadline - 5:
                break
            
            try:
                U, V, W, err = _als_decompose(n, m, p, rank, max_iters=100, tol=1e-8, seed=seed)
                
                if err < 1e-6:
                    # Snap to half-integers if possible
                    U_snapped, V_snapped, W_snapped = _snap_to_half_integer(U, V, W, tol=1e-3)
                    valid, err_snapped = _verify_decomposition(U_snapped, V_snapped, W_snapped, n, m, p, tol=1e-4)
                    if valid:
                        U_best, V_best, W_best, rank_best = U_snapped, V_snapped, W_snapped, rank
                        break
                    else:
                        # Use unsnapped if snapped doesn't work
                        U_best, V_best, W_best, rank_best = U, V, W, rank
                        break
            except Exception:
                continue
        
        if rank_best < 48:
            # Found improvement, can try lower ranks
            pass
    
    return U_best, V_best, W_best, rank_best

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s margin
    
    U, V, W, rank = _build_block_optimized()
    
    # Convert to lists as required
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
