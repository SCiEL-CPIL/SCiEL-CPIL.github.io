import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    T = np.ones((n, m, p), dtype=np.float64)
    
    # Schoolbook decomposition
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1.0
                V_sb[j * p + k, col] = 1.0
                W_sb[k * n + i, col] = 1.0
                col += 1
    
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Fast Peeling
    def peel_fast(seed):
        rng = np.random.RandomState(seed)
        residual = T.copy()
        U_cols = []
        V_cols = []
        W_cols = []
        
        max_iters = 200
        n_trials = 500
        
        for it in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            if np.allclose(residual, 0, atol=1e-9):
                break
            
            best_score = -np.inf
            best_u, best_v, best_w = None, None, None
            
            for t in range(n_trials):
                u = rng.randn(n * m)
                v = rng.randn(m * p)
                
                u_mat = u.reshape(n, m)
                v_mat = v.reshape(m, p)
                
                # Compute numerator and denominator for w
                # num[i, k] = sum_j residual[i, j, k] * u[i, j] * v[j, k]
                # den[i, k] = sum_j (u[i, j] * v[j, k])^2
                
                # Vectorized computation
                # u_mat is (n, m), v_mat is (m, p)
                # u_mat[:, :, None] * v_mat[None, :, :] -> (n, m, p)
                # This is too big? n*m*p = 120. It's fine.
                
                prod = u_mat[:, :, None] * v_mat[None, :, :] # (n, m, p)
                num = np.einsum('ijk,ijk->ik', residual, prod)
                den = np.einsum('ijk,ijk->ik', prod, prod)
                
                mask = den > 1e-12
                w_flat = np.zeros(p * n)
                w_mat = np.zeros((n, p))
                w_mat[mask] = num[mask] / den[mask]
                
                # Reconstruct
                recon = np.einsum('ij,jk,ki->ijk', u_mat, v_mat, w_mat)
                diff = residual - recon
                score = -np.sum(diff * diff)
                
                if score > best_score:
                    best_score = score
                    best_u, best_v, best_w = u.copy(), v.copy(), w_flat.copy()
                    # Note: w_flat is not used directly, we use w_mat for reconstruction
                    # But we need to store w_flat in the correct order: W[k*n+i]
                    # w_mat[i, k] corresponds to W[k*n + i]
                    # So we need to transpose w_mat to get w_flat?
                    # w_flat[k*n + i] = w_mat[i, k]
                    # So w_flat = w_mat.T.flatten()
                    best_w = w_mat.T.flatten()
            
            if best_u is None:
                break
            
            # Verify improvement
            recon = np.einsum('ij,jk,ki->ijk', best_u.reshape(n,m), best_v.reshape(m,p), best_w.reshape(n,p))
            new_residual = residual - recon
            if np.sum(new_residual * new_residual) >= np.sum(residual * residual) - 1e-10:
                break
            
            U_cols.append(best_u)
            V_cols.append(best_v)
            W_cols.append(best_w)
            residual = new_residual
        
        if len(U_cols) == 0:
            return None
        
        U = np.column_stack(U_cols)
        V = np.column_stack(V_cols)
        W = np.column_stack(W_cols)
        rank = len(U_cols)
        
        # Final verification
        recon = np.zeros((n, m, p))
        for r in range(rank):
            recon += np.einsum('ij,jk,ki->ijk', U[:,r].reshape(n,m), V[:,r].reshape(m,p), W[:,r].reshape(n,p))
        
        if np.allclose(recon, T, atol=1e-7):
            return U, V, W, rank
        else:
            return None

    # Try multiple seeds
    for seed in range(1000):
        if time.time() > deadline - 10.0:
            break
        result = peel_fast(seed)
        if result is not None:
            U, V, W, rank = result
            if rank < best_rank:
                best_rank = rank
                best_U, best_V, best_W = U, V, W
                print(f"Found rank {rank} with seed {seed}")
                if rank < 100: # Early stop if very good
                    break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
