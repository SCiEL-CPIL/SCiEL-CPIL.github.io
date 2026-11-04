    def solve_rank_fast(n, m, p, rank, max_iters=20000, tol=1e-10):
        np.random.seed(42)
        U = np.random.randn(n * m, rank) * 0.5
        V = np.random.randn(m * p, rank) * 0.5
        W = np.random.randn(p * n, rank) * 0.5
        
        ones_p = np.ones(p)
        ones_n = np.ones(n)
        ones_m = np.ones(m)
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
                
            # Update U
            U_new = np.zeros((n * m, rank))
            for i in range(n):
                for j in range(m):
                    # A (p, rank)
                    rows_w = [k * n + i for k in range(p)]
                    A = V[j * p : j * p + p, :] * W[rows_w, :]
                    # U = A^T (A A^T)^-1 1
                    AT_A = A.T @ A # rank x rank? No, we want min norm.
                    # Min norm solution to A u = b is u = A^T (A A^T)^-1 b
                    AA_T = A @ A.T # p x p
                    try:
                        U_new[i * m + j, :] = A.T @ np.linalg.solve(AA_T, ones_p)
                    except:
                        pass
            U = U_new

            # Update V
            V_new = np.zeros((m * p, rank))
            for j in range(m):
                for k in range(p):
                    # B (n, rank)
                    rows_u = [i * m + j for i in range(n)]
                    rows_w = [k * n + i for i in range(n)]
                    B = U[rows_u, :] * W[rows_w, :]
                    BB_T = B @ B.T # n x n
                    try:
                        V_new[j * p + k, :] = B.T @ np.linalg.solve(BB_T, ones_n)
                    except:
                        pass
            V = V_new

            # Update W
            W_new = np.zeros((p * n, rank))
            for k in range(p):
                for i in range(n):
                    # C (m, rank)
                    rows_u = [i * m + j for j in range(m)]
                    rows_v = [j * p + k for j in range(m)]
                    C = U[rows_u, :] * V[rows_v, :]
                    CC_T = C @ C.T # m x m
                    try:
                        W_new[k * n + i, :] = C.T @ np.linalg.solve(CC_T, ones_m)
                    except:
                        pass
            W = W_new
            
            if it % 100 == 0:
                res = compute_residuals(U.tolist(), V.tolist(), W.tolist(), rank)
                if np.max(np.abs(res)) < tol:
                    break

        U_snap = snap_to_half_int(U.tolist())
        V_snap = snap_to_half_int(V.tolist())
        W_snap = snap_to_half_int(W.tolist())
        
        if verify(U_snap, V_snap, W_snap):
            return {"U": U_snap, "V": V_snap, "W": W_snap}
        return None
