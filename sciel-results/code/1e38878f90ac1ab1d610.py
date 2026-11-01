import time
import numpy as np
from itertools import product

def construct():
    start = time.time()
    deadline = start + 115  # safety margin before 120s limit
    n, m, p = 2, 3, 5
    
    # Build the tensor T of shape (n*m, m*p, p*n) = (6, 15, 10)
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    def try_rank(r, max_iters=50, seed=42):
        """Try to find a rank-r decomposition using ALS with multiple restarts."""
        for restart in range(3):
            if time.time() > deadline:
                return None
            rng = np.random.RandomState(seed + restart * 1000)
            U = rng.randn(n * m, r)
            V = rng.randn(m * p, r)
            W = rng.randn(p * n, r)
            
            for it in range(max_iters):
                if time.time() > deadline:
                    return None
                
                # Update W: for each c, W[c,:] minimizes ||T[:,:,c] - sum_r W[c,r] * (U[:,r] ⊗ V[:,r])||
                B = np.zeros((n * m * m * p, r))
                for r_idx in range(r):
                    B[:, r_idx] = np.kron(V[:, r_idx], U[:, r_idx])
                
                W_new = np.zeros((p * n, r))
                for c in range(p * n):
                    rhs = T[:, :, c].flatten()
                    W_new[c, :] = np.linalg.lstsq(B, rhs, rcond=None)[0]
                W = W_new
                
                # Update V: for each b, V[b,:] minimizes ||T[:,b,:] - sum_r V[b,r] * (W[:,r] ⊗ U[:,r])||
                C_mat = np.zeros((n * m * p * n, r))
                for r_idx in range(r):
                    C_mat[:, r_idx] = np.kron(W[:, r_idx], U[:, r_idx])
                
                V_new = np.zeros((m * p, r))
                for b in range(m * p):
                    rhs = T[:, b, :].flatten()
                    V_new[b, :] = np.linalg.lstsq(C_mat, rhs, rcond=None)[0]
                V = V_new
                
                # Update U: for each a, U[a,:] minimizes ||T[a,:,:] - sum_r U[a,r] * (W[:,r] ⊗ V[:,r])||
                D = np.zeros((m * p * p * n, r))
                for r_idx in range(r):
                    D[:, r_idx] = np.kron(W[:, r_idx], V[:, r_idx])
                
                U_new = np.zeros((n * m, r))
                for a in range(n * m):
                    rhs = T[a, :, :].flatten()
                    U_new[a, :] = np.linalg.lstsq(D, rhs, rcond=None)[0]
                U = U_new
                
                # Check convergence
                T_recon = np.zeros_like(T)
                for r_idx in range(r):
                    T_recon += np.einsum('a,b,c->abc', U[:, r_idx], V[:, r_idx], W[:, r_idx])
                err = np.max(np.abs(T_recon - T))
                if err < 1e-10:
                    break
            
            # Final check
            T_recon = np.zeros_like(T)
            for r_idx in range(r):
                T_recon += np.einsum('a,b,c->abc', U[:, r_idx], V[:, r_idx], W[:, r_idx])
            err = np.max(np.abs(T_recon - T))
            if err < 1e-8:
                return err, U, V, W
        
        return None
    
    # Try ranks from low to high
    best_result = None
    for r in [10, 12, 14, 15, 18, 20, 25, 30]:
        if time.time() > deadline:
            break
        result = try_rank(r)
        if result is not None:
            err, U, V, W = result
            best_result = (r, U, V, W)
            break
    
    if best_result is None:
        # Fallback to naive rank-30
        r = n * m * p
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        W = np.zeros((p * n, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    r, U, V, W = best_result
    
    # Try to round to rational numbers (integers or half-integers)
    for mat in (U, V, W):
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat[i, j]
                # Try rounding to nearest 0.5
                rounded = np.round(val * 2) / 2
                if abs(val - rounded) < 1e-6:
                    mat[i, j] = rounded
    
    # Verify after rounding
    T_recon = np.zeros_like(T)
    for r_idx in range(r):
        T_recon += np.einsum('a,b,c->abc', U[:, r_idx], V[:, r_idx], W[:, r_idx])
    err = np.max(np.abs(T_recon - T))
    
    if err > 1e-6:
        # Rounding broke it, fall back to naive
        r = n * m * p
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        W = np.zeros((p * n, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
