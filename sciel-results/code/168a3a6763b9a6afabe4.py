"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

import numpy as np
import time
from itertools import product

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Build the multiplication tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Lower bound on rank: max(n*m, m*p, p*n)
    lb = max(n * m, m * p, p * n)
    
    best_rank = n * m * p  # schoolbook
    best_U = None
    best_V = None
    best_W = None
    
    def make_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, col] = 1
                    V[b, col] = 1
                    W[c, col] = 1
                    col += 1
        return U, V, W
    
    U_sb, V_sb, W_sb = make_schoolbook()
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = n * m * p
    
    # Try to find lower rank decompositions using ALS
    def als_decompose(T, rank, max_iter=1000, tol=1e-10, seed=42):
        """Alternating least squares for tensor decomposition."""
        rng = np.random.RandomState(seed)
        U = rng.randn(T.shape[0], rank)
        V = rng.randn(T.shape[1], rank)
        W = rng.randn(T.shape[2], rank)
        
        T_flat = T.reshape(-1, 1)
        
        for it in range(max_iter):
            # Update U: for each column r, solve for U[:,r] given V[:,r], W[:,r]
            # T = sum_r U_r ⊗ V_r ⊗ W_r
            # For fixed r, the contribution is U_r ⊗ (V_r ⊗ W_r)
            # We want sum_r U_r ⊗ (V_r ⊗ W_r) = T
            # This is a linear system in {U_r}
            
            # Build the system: T_vec = M @ vec(U)
            # where M is constructed from V and W
            
            # Actually, let's do it properly.
            # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
            
            # For updating U: fix V, W. For each r, we need to find U[:,r].
            # The residual after removing term r is: T - sum_{r' != r} U_{r'} ⊗ V_{r'} ⊗ W_{r'}
            # This should equal U_r ⊗ V_r ⊗ W_r
            # So U_r = (T - sum_{r' != r} ...) contracted with V_r, W_r / ||V_r||^2 ||W_r||^2
            
            # More efficiently: 
            # Let's form the matricization. T reshaped to (n*m, m*p*p*n)
            # T_mat[a, (b,c)] = T[a,b,c]
            # T = sum_r U_r ⊗ (V_r ⊗ W_r)
            # So T_mat = U @ (V ⊗ W)^T where (V ⊗ W) is the Khatri-Rao product
            
            # Khatri-Rao product: KR[:,r] = V[:,r] ⊗ W[:,r]
            KR = np.zeros((T.shape[1] * T.shape[2], rank))
            for r in range(rank):
                KR[:, r] = np.kron(V[:, r], W[:, r])
            
            T_mat = T.reshape(T.shape[0], -1)
            
            # Solve: T_mat = U @ KR^T
            # U = T_mat @ (KR^T)^+ 
            try:
                U_new = T_mat @ np.linalg.pinv(KR.T)
            except:
                break
            
            # Update V
            KR2 = np.zeros((T.shape[0] * T.shape[2], rank))
            for r in range(rank):
                KR2[:, r] = np.kron(U_new[:, r], W[:, r])
            T_mat2 = T.transpose(0, 2, 1).reshape(T.shape[0], -1)  # (a,c,b) -> reshape to (a, c*b)
            # Actually T_mat2[a, (c,b)] = T[a,b,c]
            T_mat2 = T.reshape(T.shape[0], T.shape[1] * T.shape[2]).copy()
            # We need T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            # For V: fix U, W. T_mat_v[b, (a,c)] = sum_r V[b,r] * (U[a,r] * W[c,r])
            # So T_mat_v = V @ (U ⊗ W)^T where the (a,c) index is combined
            KR_v = np.zeros((T.shape[0] * T.shape[2], rank))
            for r in range(rank):
                KR_v[:, r] = np.kron(U_new[:, r], W[:, r])
            T_mat_v = T.transpose(1, 0, 2).reshape(T.shape[1], -1)  # (b, a, c) -> (b, a*c)
            try:
                V_new = T_mat_v @ np.linalg.pinv(KR_v.T)
            except:
                break
            
            # Update W
            KR_w = np.zeros((T.shape[0] * T.shape[1], rank))
            for r in range(rank):
                KR_w[:, r] = np.kron(U_new[:, r], V_new[:, r])
            T_mat_w = T.transpose(2, 0, 1).reshape(T.shape[2], -1)  # (c, a, b) -> (c, a*b)
            try:
                W_new = T_mat_w @ np.linalg.pinv(KR_w.T)
            except:
                break
            
            U, V, W = U_new, V_new, W_new
            
            # Check convergence
            recon = np.zeros_like(T)
            for r in range(rank):
                recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
            err = np.linalg.norm(recon - T) / np.linalg.norm(T)
            if err < tol:
                break
        
        # Final check
        recon = np.zeros_like(T)
        for r in range(rank):
            recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        err = np.linalg.norm(recon - T) / np.linalg.norm(T)
        
        return U, V, W, err
    
    # Try ranks from lower bound up
    deadline = time.time() + 110  # leave 10s margin
    
    for rank in range(lb, best_rank):
        if time.time() > deadline:
            break
        
        found = False
        for seed in range(20):
            if time.time() > deadline:
                break
            U_r, V_r, W_r, err = als_decompose(T, rank, max_iter=500, tol=1e-8, seed=seed)
            if err < 1e-6:
                # Verify more carefully
                recon = np.zeros_like(T)
                for r in range(rank):
                    recon += np.einsum('a,b,c->abc', U_r[:, r], V_r[:, r], W_r[:, r])
                if np.allclose(recon, T, atol=1e-6):
                    # Found a valid decomposition
                    # Try to clean up to integer/half-integer if possible
                    best_rank = rank
                    best_U = U_r
                    best_V = V_r
                    best_W = W_r
                    found = True
                    break
        if found:
            break
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
