def _try_rank(rank, max_iters, n_restarts, deadline, best):
    n, m, p = DIMENSIONS
    best_rank, best_U, best_V, best_W = best
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        # random init
        U = np.random.randn(n*m, rank)
        V = np.random.randn(m*p, rank)
        W = np.random.randn(p*n, rank)
        # normalize to avoid blowup
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            U, V, W = _als_step(U, V, W, 'U')
            U, V, W = _als_step(U, V, W, 'V')
            U, V, W = _als_step(U, V, W, 'W')
            rsq = _residual_sq(U, V, W)
            if rsq < 1e-12:
                # exact-ish, try rounding
                Ur = np.round(U*2)/2
                Vr = np.round(V*2)/2
                Wr = np.round(W*2)/2
                rsq_r = _residual_sq(Ur, Vr, Wr)
                if rsq_r < 1e-9:
                    if rank < best_rank:
                        best = (rank, Ur, Vr, Wr)
                        best_rank = rank
                break
    return best
