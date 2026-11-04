def _als_decompose(r, n=3, m=3, p=7, iters=50, seed=42):
    np.random.seed(seed)
    U = np.random.randn(n*m, r)
    V = np.random.randn(m*p, r)
    W = np.random.randn(p*n, r)
    
    for it in range(iters):
        # Solve for U given V, W
        for i in range(n):
            for j in range(m):
                # Set up linear system for U[i*m+j, :]
                A = np.zeros((m*p*p*n, r))
                b = np.zeros(m*p*p*n)
                idx = 0
                for j_prime in range(m):
                    for k in range(p):
                        for k_prime in range(p):
                            for i_prime in range(n):
                                for r_idx in range(r):
                                    A[idx, r_idx] = V[j_prime*p+k, r_idx] * W[k_prime*n+i_prime, r_idx]
                                b[idx] = 1.0 if (i==i_prime and k==k_prime and j==j_prime) else 0.0
                                idx += 1
                u, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                U[i*m+j, :] = u
        
        # Similarly for V and W...
        # For V: fix U, W. Unknowns V[j'*p+k, r].
        # Equations: Σ_r U[i*m+j, r] * V[j'*p+k, r] * W[k'*n+i', r] = δ_{i,i'}δ_{k,k'}δ_{j,j'}
        # For fixed (j',k), unknowns are V[j'*p+k, r].
        # Equations indexed by (i, j, k', i'):
        # Σ_r U[i*m+j, r] * V[j'*p+k, r] * W[k'*n+i', r] = δ_{i,i'}δ_{k,k'}δ_{j,j'}
        
        for j_prime in range(m):
            for k in range(p):
                A = np.zeros((n*m*p*n, r))
                b = np.zeros(n*m*p*n)
                idx = 0
                for i in range(n):
                    for j in range(m):
                        for k_prime in range(p):
                            for i_prime in range(n):
                                for r_idx in range(r):
                                    A[idx, r_idx] = U[i*m+j, r_idx] * W[k_prime*n+i_prime, r_idx]
                                b[idx] = 1.0 if (i==i_prime and k==k_prime and j==j_prime) else 0.0
                                idx += 1
                v, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                V[j_prime*p+k, :] = v
        
        # For W: fix U, V. Unknowns W[k'*n+i', r].
        # Equations: Σ_r U[i*m+j, r] * V[j'*p+k, r] * W[k'*n+i', r] = δ_{i,i'}δ_{k,k'}δ_{j,j'}
        # For fixed (k',i'), unknowns are W[k'*n+i', r].
        # Equations indexed by (i, j, j', k):
        # Σ_r U[i*m+j, r] * V[j'*p+k, r] * W[k'*n+i', r] = δ_{i,i'}δ_{k,k'}δ_{j,j'}
        
        for k_prime in range(p):
            for i_prime in range(n):
                A = np.zeros((n*m*m*p, r))
                b = np.zeros(n*m*m*p)
                idx = 0
                for i in range(n):
                    for j in range(m):
                        for j_prime in range(m):
                            for k in range(p):
                                for r_idx in range(r):
                                    A[idx, r_idx] = U[i*m+j, r_idx] * V[j_prime*p+k, r_idx]
                                b[idx] = 1.0 if (i==i_prime and k==k_prime and j==j_prime) else 0.0
                                idx += 1
                w, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                W[k_prime*n+i_prime, :] = w
    
    # Compute final loss
    loss = 0.0
    for i in range(n):
        for j in range(m):
            for j_prime in range(m):
                for k in range(p):
                    for k_prime in range(p):
                        for i_prime in range(n):
                            target = 1.0 if (i==i_prime and k==k_prime and j==j_prime) else 0.0
                            rec = np.sum(U[i*m+j, :] * V[j_prime*p+k, :] * W[k_prime*n+i_prime, :])
                            loss += (rec - target) ** 2
    
    return U, V, W, loss
