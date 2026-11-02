import numpy as np
import time
import random

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    R_full = n * m * p  # 84
    
    deadline = time.time() + 140.0
    best_rank = R_full
    best_U = None
    best_V = None
    best_W = None
    
    # Build the full decomposition first
    U_full = np.zeros((nm, R_full), dtype=float)
    V_full = np.zeros((mp, R_full), dtype=float)
    W_full = np.zeros((pn, R_full), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_full[i * m + j, col] = 1.0
                V_full[j * p + k, col] = 1.0
                W_full[k * n + i, col] = 1.0
                col += 1
    
    # Store best as lists initially
    best_U = U_full.tolist()
    best_V = V_full.tolist()
    best_W = W_full.tolist()
    
    def verify(U_mat, V_mat, W_mat):
        """Check if U, V, W form a valid decomposition."""
        # Compute C = sum_r U[:,r] * V[:,r] * W[:,r] in tensor form
        # For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should be 1
        # Efficient: use einsum or matmul
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Reshape U to (n,m,R), V to (m,p,R), W to (p,n,R)
        U_t = U_mat.reshape(n, m, -1)
        V_t = V_mat.reshape(m, p, -1)
        W_t = W_mat.reshape(p, n, -1)
        # T[i,j,k] = sum_r U_t[i,j,r] * V_t[j,k,r] * W_t[k,i,r]
        # Use einsum: 'ijr,jkr,kir->ijk'
        T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
        return np.allclose(T, np.ones((n, m, p)), atol=1e-10)
    
    def try_reduce_rank(U0, V0, W0, target_rank, max_iters=500, seed=None):
        """Try to find a valid decomposition of rank target_rank starting from U0,V0,W0."""
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()
        
        R = target_rank
        # Randomly select R columns from the full decomposition
        full_cols = list(range(U0.shape[1]))
        selected = rng.sample(full_cols, min(R, len(full_cols)))
        
        U = U0[:, selected].copy()
        V = V0[:, selected].copy()
        W = W0[:, selected].copy()
        
        # Also try random initialization
        if rng.random() < 0.3:
            U = np.random.randn(nm, R) * 0.5
            V = np.random.randn(mp, R) * 0.5
            W = np.random.randn(pn, R) * 0.5
        
        # Iterative refinement
        for it in range(max_iters):
            if time.time() > deadline:
                return None, None, None
            
            # Compute residual
            U_t = U.reshape(n, m, R)
            V_t = V.reshape(m, p, R)
            W_t = W.reshape(p, n, R)
            T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
            residual = 1.0 - T
            
            # Check convergence
            if np.max(np.abs(residual)) < 1e-10:
                return U, V, W
            
            # Gauss-Seidel updates: fix two factors, solve for the third
            # Update U: for each (i,j), U[i*m+j, :] should satisfy
            # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all k
            # This is a linear system in U[i*m+j,:]
            # Let A[i,j,k,r] = V[j*p+k,r] * W[k*n+i,r]
            # Then sum_r U[i*m+j,r] * A[i,j,k,r] = 1 for all k
            # A is (p, R), solve A^T * u = 1 (p equations, R unknowns)
            
            # Update U
            for i in range(n):
                for j in range(m):
                    # Build system: for each k, sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
                    A = np.outer(V[j*p:(j+1)*p, :], W[0:n, :].T * 1.0)  # wrong, let me redo
                    # A[k, r] = V[j*p+k, r] * W[k*n+i, r]
                    A = np.zeros((p, R))
                    for k in range(p):
                        A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    # Solve A^T @ u = ones(p)
                    try:
                        u_sol, residuals, rank_a, sv = np.linalg.lstsq(A.T, np.ones(p), rcond=None)
                        U[i*m+j, :] = u_sol
                    except:
                        pass
            
            # Update V
            for j in range(m):
                for k in range(p):
                    # For each i, sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
                    # A[i, r] = U[i*m+j, r] * W[k*n+i, r]
                    A = np.zeros((n, R))
                    for i in range(n):
                        A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    try:
                        v_sol, residuals, rank_a, sv = np.linalg.lstsq(A.T, np.ones(n), rcond=None)
                        V[j*p+k, :] = v_sol
                    except:
                        pass
            
            # Update W
            for k in range(p):
                for i in range(n):
                    # For each j, sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
                    # A[j, r] = U[i*m+j, r] * V[j*p+k, r]
                    A = np.zeros((m, R))
                    for j in range(m):
                        A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    try:
                        w_sol, residuals, rank_a, sv = np.linalg.lstsq(A.T, np.ones(m), rcond=None)
                        W[k*n+i, :] = w_sol
                    except:
                        pass
        
        # Final check
        U_t = U.reshape(n, m, R)
        V_t = V.reshape(m, p, R)
        W_t = W.reshape(p, n, R)
        T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
        if np.allclose(T, np.ones((n, m, p)), atol=1e-10):
            return U, V, W
        return None, None, None
    
    # Try to reduce rank from 84 down
    for target_rank in range(R_full - 1, 20, -1):
        if time.time() > deadline:
            break
        
        # Try a few seeds
        for seed in range(5):
            if time.time() > deadline:
                break
            
            U_new, V_new, W_new = try_reduce_rank(
                U_full, V_full, W_full, target_rank, 
                max_iters=200, seed=seed
            )
            
            if U_new is not None:
                # Verify
                if verify(U_new, V_new, W_new):
                    best_rank = target_rank
                    best_U = U_new.tolist()
                    best_V = V_new.tolist()
                    best_W = W_new.tolist()
                    break  # Found valid at this rank, try lower
        
        if best_rank < R_full:
            # Continue trying to go lower
            continue
    
    return {"U": best_U, "V": best_V, "W": best_W}
