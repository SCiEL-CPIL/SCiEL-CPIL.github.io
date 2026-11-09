    def optimize_rank(r):
        if time.time() > deadline:
            return None
        
        # Initialize
        U = np.random.randn(n * m, r) * 0.1
        V = np.random.randn(m * p, r) * 0.1
        W = np.random.randn(p * n, r) * 0.1
        
        # Precompute target vectors b_w, b_u, b_v
        # b_w[w] is a vector of length n*m*m*p
        b_w = np.zeros((p * n, n * m * m * p))
        b_u = np.zeros((n * m, p * n * m * p))
        b_v = np.zeros((m * p, p * n * n * m))
        
        for tw, tu, tv in target_indices:
            b_w[tw, tu * (m * p) + tv] = 1.0
            b_u[tu, tw * (m * p) + tv] = 1.0
            b_v[tv, tw * (n * m) + tu] = 1.0
            
        # Precompute indices for flattening
        # A_W shape (n*m, m*p, r) -> (n*m*m*p, r)
        # A_U shape (p*n, m*p, r) -> (p*n*m*p, r)
        # A_V shape (p*n, n*m, r) -> (p*n*n*m, r)

        for step in range(300):
            if time.time() > deadline:
                return None
            
            # Update W
            # A_W = U (n*m, r) and V (m*p, r)
            # A_W[u, v, t] = U[u, t] * V[v, t]
            A_W = np.einsum('ut,vt->uvt', U, V).reshape(-1, r)
            # Solve W[w, :] = lstsq(A_W, b_w[w, :])
            # This is a batched least squares.
            # We can do it in a loop or use batched solver if available.
            # Loop is fine for 15 rows.
            for w in range(p * n):
                W[w, :] = np.linalg.lstsq(A_W, b_w[w, :], rcond=None)[0]
                
            # Update U
            A_U = np.einsum('wt,vt->wvt', W, V).reshape(-1, r)
            for u in range(n * m):
                U[u, :] = np.linalg.lstsq(A_U, b_u[u, :], rcond=None)[0]
                
            # Update V
            A_V = np.einsum('wt,ut->wut', W, U).reshape(-1, r)
            for v in range(m * p):
                V[v, :] = np.linalg.lstsq(A_V, b_v[v, :], rcond=None)[0]
                
            # Check convergence
            if step % 20 == 0:
                T_calc = np.einsum('wt,ut,vt->wuv', W, U, V)
                # Target T is sparse.
                # Check error on non-zeros
                err = 0
                for tw, tu, tv in target_indices:
                    err += (T_calc[tw, tu, tv] - 1.0)**2
                # Check error on zeros (sample)
                # For now, just check non-zeros. If they are close to 1, and we assume structure, 
                # the zeros might be small.
                # But we need exactness.
                if err < 1e-8:
                    # Try rounding
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    if verify(U_r, V_r, W_r_r): # Typo in previous thought, verify takes 3 args
                        return U_r, V_r, W_r
                    U_r = np.round(U)
                    V_r = np.round(V)
                    W_r = np.round(W)
                    if verify(U_r, V_r, W_r):
                        return U_r, V_r, W_r
        return None
