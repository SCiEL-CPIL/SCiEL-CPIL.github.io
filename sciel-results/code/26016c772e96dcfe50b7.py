def _matrix_hooi(residual, n, m, p, iters=20, rng=None):
    if rng is None:
        rng = np.random.RandomState(42)
        
    U = rng.randn(n, m)
    V = rng.randn(m, p)
    W = rng.randn(p, n)
    
    for _ in range(iters):
        # Update U
        # N_ij = sum_k R_ijk V_jk W_ki
        # D_ij = sum_k (V_jk W_ki)^2
        # We can compute this using einsum
        # V_jk W_ki is a tensor of shape (m, p, n) ? No.
        # V is (m, p). W is (p, n).
        # V_jk W_ki = sum_k V_jk W_ki. This is matrix product V @ W of shape (m, n).
        # But we need the sum over k of R_ijk * (V_jk W_ki).
        # R is (n, m, p).
        # Let's reshape R to (n*m, p).
        # R_reshaped[i*m+j, k] = R[i, j, k]
        # V is (m, p). W is (p, n).
        # We want N_ij = sum_k R[i, j, k] V[j, k] W[k, i].
        
        # Efficient computation:
        # 1. Compute M = V @ W (shape m, n)
        #    M[j, i] = sum_k V[j, k] W[k, i]
        #    But we need sum_k R[i, j, k] * V[j, k] * W[k, i].
        #    This is not just R @ M.
        #    Let's use einsum: 'ijk,jk,ki->ij'
        N_U = np.einsum('ijk,jk,ki->ij', residual, V, W)
        D_U = np.einsum('ijk,jk,ki->ij', (V*W).reshape(m, 1, p) * (V*W).reshape(1, p, n), 1, 1) # Wrong
        # D_ij = sum_k (V_jk W_ki)^2
        # V_jk W_ki is element-wise product of V and W transposed?
        # V is (m, p). W is (p, n).
        # V_jk * W_ki.
        # Let's just compute it directly for small sizes.
        
        # For n=4, m=5, p=5, this is small.
        # D_U[i, j] = 0
        # for k in range(p):
        #   val = V[j, k] * W[k, i]
        #   D_U[i, j] += val * val
        
        # Vectorized:
        # V_exp = V[:, :, np.newaxis] # m, p, 1
        # W_exp = W[np.newaxis, :, :] # 1, p, n
        # Product = V_exp * W_exp # m, p, n
        # D_U = np.einsum('mpn->ni', Product**2) # No, we want i, j.
        # Product[j, k, i] = V[j, k] W[k, i]
        # D_U[i, j] = sum_k Product[j, k, i]^2
        # D_U = np.sum(Product**2, axis=1) # Shape m, n -> transpose to n, m
        D_U = np.sum((V[:, :, np.newaxis] * W[np.newaxis, :, :])**2, axis=1).T # Shape n, m
        
        U = N_U / (D_U + 1e-10)
        
        # Update V
        # N_V[j, k] = sum_i R[i, j, k] U[i, j] W[k, i]
        # D_V[j, k] = sum_i (U[i, j] W[k, i])^2
        N_V = np.einsum('ijk,ij,ki->jk', residual, U, W)
        # U[i, j] W[k, i]
        # U_exp = U[:, :, np.newaxis] # n, m, 1
        # W_exp = W[np.newaxis, np.newaxis, :] # 1, 1, p? No W is p, n.
        # We need U[i, j] * W[k, i].
        # Shape of product: i, j, k.
        # U[i, j] is (n, m). W[k, i] is (p, n).
        # U_exp = U[:, :, np.newaxis] # n, m, 1
        # W_exp = W.T[np.newaxis, :, :] # n, p, 1? No.
        # W[k, i] -> W.T[i, k].
        # U[i, j] W.T[i, k].
        # Product[i, j, k] = U[i, j] * W.T[i, k].
        # D_V[j, k] = sum_i Product[i, j, k]^2.
        Prod_V = U[:, :, np.newaxis] * W.T[np.newaxis, :, :] # n, m, p
        D_V = np.sum(Prod_V**2, axis=0) # m, p
        
        V = N_V / (D_V + 1e-10)
        
        # Update W
        # N_W[k, i] = sum_j R[i, j, k] U[i, j] V[j, k]
        # D_W[k, i] = sum_j (U[i, j] V[j, k])^2
        N_W = np.einsum('ijk,ij,jk->ki', residual, U, V)
        # U[i, j] V[j, k]
        # U_exp = U[:, :, np.newaxis] # n, m, 1
        # V_exp = V[np.newaxis, :, :] # 1, m, p
        # Product[i, j, k] = U[i, j] V[j, k]
        Prod_W = U[:, :, np.newaxis] * V[np.newaxis, :, :] # n, m, p
        D_W = np.sum(Prod_W**2, axis=1) # n, p -> transpose to p, n
        D_W = D_W.T
        
        W = N_W / (D_W + 1e-10)
        
    return U, V, W
