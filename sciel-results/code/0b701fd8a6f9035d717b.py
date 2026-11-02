def als_decompose(target, n, m, p, rank, n_iter=200, n_restarts=50):
    best = None
    best_err = np.inf
    rng = np.random.RandomState(42)
    for restart in range(n_restarts):
        U = rng.randn(n*m, rank)
        V = rng.randn(m*p, rank)
        W = rng.randn(p*n, rank)
        for it in range(n_iter):
            # Update U
            WV = np.kron(W.T, V.T).T  # (m*p*p*n, rank), column r = w_r ⊗ v_r
            # Actually Khatri-Rao: (W⊙V)[(b,c), r] = W[c,r] V[b,r]
            # Let's build it properly
            # WV_kr[b*(p*n)+c, r] = V[b,r] * W[c,r]
            WV_kr = (V[:, :, None] * W[None, :, :]).reshape(m*p*p*n, rank)
            T1 = target.reshape(n*m, m*p*p*n)
            U = T1 @ np.linalg.pinv(WV_kr)
            # Update V
            # T_(2)[b, (a,c)] = T[a,b,c]
            T2 = target.transpose(1,0,2).reshape(m*p, n*m*p*n)
            UW_kr = (U[:, :, None] * W[None, :, :]).reshape(n*m*p*n, rank)
            V = T2 @ np.linalg.pinv(UW_kr)
            # Update W
            T3 = target.transpose(2,0,1).reshape(p*n, n*m*m*p)
            UV_kr = (U[:, :, None] * V[None, :, :]).reshape(n*m*m*p, rank)
            W = T3 @ np.linalg.pinv(UV_kr)
            # Compute error
            T_comp = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.linalg.norm(T_comp - target)
            if err < best_err:
                best_err = err
                best = (U.copy(), V.copy(), W.copy())
            if err < 1e-8:
                break
        if best_err < 1e-8:
            break
    return best, best_err
