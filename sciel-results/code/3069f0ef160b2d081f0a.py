<<<<<<< SEARCH
def _try_als_rank(R, T, deadline, n_seeds=5, max_iters=100):
    """Try ALS for a given rank with rational polish."""
    n, m, p = DIMENSIONS
    U_rows, V_rows, W_rows = n * m, m * p, p * n

    seeds = [42, 123, 7, 999, 2024]

    for seed in seeds[:n_seeds]:
        if time.time() > deadline:
            break
        np.random.seed(seed)

        inits = []
        inits.append((np.random.randn(U_rows, R) * 0.3,
                      np.random.randn(V_rows, R) * 0.3,
                      np.random.randn(W_rows, R) * 0.3))
        inits.append((np.random.randn(U_rows, R) * 1.0,
                      np.random.randn(V_rows, R) * 1.0,
                      np.random.randn(W_rows, R) * 1.0))

        for U0, V0, W0 in inits:
            if time.time() > deadline:
                break
            U, V, W = U0.copy(), V0.copy(), W0.copy()
            prev_err = np.inf
            for it in range(max_iters):
                if time.time() > deadline:
                    break
                # Update U
                A = np.einsum('vr,wr->vwr', V, W).reshape(-1, R)
                for u in range(U_rows):
                    b = T[u].reshape(-1).astype(np.float64)
                    sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    U[u] = sol
                
                # Update V
                A = np.einsum('ur,wr->uwr', U, W).reshape(-1, R)
                for v in range(V_rows):
                    b = T[:, v, :].reshape(-1).astype(np.float64)
                    sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    V[v] = sol
                
                # Update W
                A = np.einsum('ur,vr->uvr', U, V).reshape(-1, R)
                for w in range(W_rows):
                    b = T[:, :, w].reshape(-1).astype(np.float64)
                    sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    W[w] = sol
                
                result = np.zeros((U_rows, V_rows, W_rows), dtype=np.float64)
                for r in range(R):
                    result += np.einsum('u,v,w->uvw', U[:, r], V[:, r], W[:, r])
                err = np.linalg.norm(result - T.astype(np.float64))
                
                if abs(prev_err - err) < 1e-14:
                    break
                prev_err = err

            if _check_decomposition(U, V, W, T, tol=1e-7):
                # Try to snap to rationals
                for denom in [1, 2]:
                    U_s = np.round(U * denom) / denom
                    V_s = np.round(V * denom) / denom
                    W_s = np.round(W * denom) / denom
                    if _check_exact(U_s, V_s, W_s, T):
                        return U_s.tolist(), V_s.tolist(), W_s.tolist()
                return U.tolist(), V.tolist(), W.tolist()
    return None
=======
def _try_als_rank(R, T, deadline, n_seeds=5, max_iters=200, init_from=None):
    """Try ALS for a given rank with rational polish. Optionally initialize from a higher-rank solution."""
    n, m, p = DIMENSIONS
    U_rows, V_rows, W_rows = n * m, m * p, p * n
    T_float = T.astype(np.float64)

    seeds = [42, 123, 7, 999, 2024]

    for seed in seeds[:n_seeds]:
        if time.time() > deadline:
            break
        np.random.seed(seed)

        inits = []
        inits.append((np.random.randn(U_rows, R) * 0.3,
                      np.random.randn(V_rows, R) * 0.3,
                      np.random.randn(W_rows, R) * 0.3))
        inits.append((np.random.randn(U_rows, R) * 1.0,
                      np.random.randn(V_rows, R) * 1.0,
                      np.random.randn(W_rows, R) * 1.0))
        
        # If we have a higher-rank init, try truncating it
        if init_from is not None:
            U_init, V_init, W_init = init_from
            # SVD-truncate to rank R
            U_svd, _, _ = np.linalg.svd(U_init, full_matrices=False)
            V_svd, _, _ = np.linalg.svd(V_init, full_matrices=False)
            W_svd, _, _ = np.linalg.svd(W_init, full_matrices=False)
            inits.append((U_svd[:, :R], V_svd[:, :R], W_svd[:, :R]))

        for U0, V0, W0 in inits:
            if time.time() > deadline:
                break
            U, V, W = U0.copy(), V0.copy(), W0.copy()
            prev_err = np.inf
            for it in range(max_iters):
                if time.time() > deadline:
                    break
                # Update U
                A = np.einsum('vr,wr->vwr', V, W).reshape(-1, R)
                for u in range(U_rows):
                    b = T_float[u].reshape(-1).astype(np.float64)
                    sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    U[u] = sol
                
                # Update V
                A = np.einsum('ur,wr->uwr', U, W).reshape(-1, R)
                for v in range(V_rows):
                    b = T_float[:, v, :].reshape(-1).astype(np.float64)
                    sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    V[v] = sol
                
                # Update W
                A = np.einsum('ur,vr->uvr', U, V).reshape(-1, R)
                for w in range(W_rows):
                    b = T_float[:, :, w].reshape(-1).astype(np.float64)
                    sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    W[w] = sol
                
                result = np.zeros((U_rows, V_rows, W_rows), dtype=np.float64)
                for r in range(R):
                    result += np.einsum('u,v,w->uvw', U[:, r], V[:, r], W[:, r])
                err = np.linalg.norm(result - T_float)
                
                if abs(prev_err - err) < 1e-14:
                    break
                prev_err = err

            if _check_decomposition(U, V, W, T, tol=1e-7):
                # Try to snap to rationals
                for denom in [1, 2]:
                    U_s = np.round(U * denom) / denom
                    V_s = np.round(V * denom) / denom
                    W_s = np.round(W * denom) / denom
                    if _check_exact(U_s, V_s, W_s, T):
                        return U_s.tolist(), V_s.tolist(), W_s.tolist()
                return U.tolist(), V.tolist(), W.tolist()
    return None
>>>>>>> REPLACE

