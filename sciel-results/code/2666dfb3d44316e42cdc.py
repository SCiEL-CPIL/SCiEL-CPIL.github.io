def construct():
    import time
    import numpy as np
    _t0 = time.time()
    _DEADLINE = 140.0  # safety margin before 150s kill

    n, m, p = 3, 4, 6

    def build_tensor():
        T = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i * m + j, j * p + k, k * n + i] = 1.0
        return T

    T = build_tensor()

    def verify_fast(U, V, W, T, tol=1e-10):
        """Fast verification using matrix reshaping: T_mat = U @ diag(W) @ V^T style."""
        # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # Reshape: for fixed c, T[:,:,c] = sum_r U[:,r]*V[:,r]^T * W[r,c]
        # = (U * W[None,:]) @ V.T  ... but W is (p*n, rank), not (rank, p*n)
        # Actually: T[:,:,c] = U @ (W[c,:] * V).T  where W[c,:] is row c of W
        # Let's just do: recon = np.einsum('ar,br,cr->abc', U, V, W)
        recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.max(np.abs(recon - T)) < tol

    def verify_exact(U, V, W, T, tol=1e-8):
        """Verification with tolerance for floating point."""
        recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.max(np.abs(recon - T)) < tol

    # Build Strassen+schoolbook decomposition (rank 66) as baseline
    def build_strassen_decomposition():
        terms = []

        def add_strassen_block(i1, i2, j1, j2, k1, k2):
            a = (i1, j1); b = (i1, j2)
            c = (i2, j1); d = (i2, j2)
            e = (j1, k1); f = (j1, k2)
            g = (j2, k1); h = (j2, k2)
            C11 = (i1, k1); C12 = (i1, k2)
            C21 = (i2, k1); C22 = (i2, k2)

            terms.append(({a: 1, d: 1}, {e: 1, h: 1}, {C11: 1, C22: 1}))
            terms.append(({c: 1, d: 1}, {e: 1}, {C21: 1, C22: -1}))
            terms.append(({a: 1}, {f: 1, h: -1}, {C12: 1, C22: 1}))
            terms.append(({d: 1}, {g: 1, e: -1}, {C11: 1, C21: 1}))
            terms.append(({a: 1, b: 1}, {h: 1}, {C11: -1, C12: 1}))
            terms.append(({c: 1, a: -1}, {e: 1, f: 1}, {C22: 1}))
            terms.append(({b: 1, d: -1}, {g: 1, h: 1}, {C11: 1}))

        def add_schoolbook_term(i, j, k):
            terms.append(({(i, j): 1}, {(j, k): 1}, {(i, k): 1}))

        # Row 0: schoolbook (12 terms)
        for j in range(m):
            for k in range(p):
                add_schoolbook_term(0, j, k)

        # Rows 1-2: Strassen blocks for each pair of k's
        for (k1, k2) in [(0, 1), (2, 3), (4, 5)]:
            add_strassen_block(1, 2, 0, 1, k1, k2)
            add_strassen_block(1, 2, 2, 3, k1, k2)

        rank = len(terms)
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))

        for col, (u_dict, v_dict, w_dict) in enumerate(terms):
            for (i, j), coeff in u_dict.items():
                U[i * m + j, col] = coeff
            for (j, k), coeff in v_dict.items():
                V[j * p + k, col] = coeff
            for (i, k), coeff in w_dict.items():
                W[k * n + i, col] = coeff

        return U, V, W

    U0, V0, W0 = build_strassen_decomposition()
    best_rank = U0.shape[1]
    best_U, best_V, best_W = U0.copy(), V0.copy(), W0.copy()

    if not verify_exact(U0, V0, W0, T):
        # Fallback to schoolbook
        rank_fb = n * m * p
        U0 = np.zeros((n * m, rank_fb))
        V0 = np.zeros((m * p, rank_fb))
        W0 = np.zeros((p * n, rank_fb))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U0[i * m + j, idx] = 1
                    V0[j * p + k, idx] = 1
                    W0[k * n + i, idx] = 1
                    idx += 1
        best_rank = rank_fb
        best_U, best_V, best_W = U0, V0, W0

    # Try to find a lower-rank decomposition using alternating minimization
    # with the Strassen decomposition as initialization.
    # Key insight: use the structure of the problem to guide initialization.

    def try_am_init_from_strassen(target_rank, n_iter=500, n_restarts=3, seed=42):
        """Alternating minimization initialized from Strassen decomposition."""
        dim_a, dim_b, dim_c = T.shape
        rng = np.random.RandomState(seed)
        best_err = np.inf
        best_U, best_V, best_W = None, None, None

        for restart in range(n_restarts):
            # Initialize from Strassen + noise
            U = np.zeros((dim_a, target_rank))
            V = np.zeros((dim_b, target_rank))
            W = np.zeros((dim_c, target_rank))

            if target_rank <= U0.shape[1]:
                U[:, :target_rank] = U0[:, :target_rank].copy()
                V[:, :target_rank] = V0[:, :target_rank].copy()
                W[:, :target_rank] = W0[:, :target_rank].copy()
            else:
                U[:, :U0.shape[1]] = U0
                V[:, :U0.shape[1]] = V0
                W[:, :W0.shape[1]] = W0

            # Add small noise to break symmetry
            noise_scale = 0.01
            U += noise_scale * rng.randn(dim_a, target_rank)
            V += noise_scale * rng.randn(dim_b, target_rank)
            W += noise_scale * rng.randn(dim_c, target_rank)

            err = np.inf
            for it in range(n_iter):
                # Update U columns
                for i in range(target_rank):
                    tvw = np.einsum('abc,b,c->a', T, V[:, i], W[:, i])
                    vv = np.dot(V[:, i], V[:, i])
                    ww = np.dot(W[:, i], W[:, i])
                    if vv * ww > 1e-15:
                        U[:, i] = tvw / (vv * ww)

                # Update V columns
                for i in range(target_rank):
                    tvw = np.einsum('abc,a,c->b', T, U[:, i], W[:, i])
                    uu = np.dot(U[:, i], U[:, i])
                    ww = np.dot(W[:, i], W[:, i])
                    if uu * ww > 1e-15:
                        V[:, i] = tvw / (uu * ww)

                # Update W columns
                for i in range(target_rank):
                    tvw = np.einsum('abc,a,b->c', T, U[:, i], V[:, i])
                    uu = np.dot(U[:, i], U[:, i])
                    vv = np.dot(V[:, i], V[:, i])
                    if uu * vv > 1e-15:
                        W[:, i] = tvw / (uu * vv)

                # Compute error using fast einsum
                recon = np.einsum('ar,br,cr->abc', U, V, W)
                err = np.linalg.norm(recon - T) / np.linalg.norm(T)

                if err < best_err:
                    best_err = err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

                if err < 1e-12:
                    break

                if time.time() - _t0 > _DEADLINE - 5:
                    return best_U, best_V, best_W, target_rank, best_err

        return best_U, best_V, best_W, target_rank, best_err

    # Try ranks from 65 down to 50
    found_lower = False
    for target_rank in range(65, 49, -1):
        if time.time() - _t0 > _DEADLINE - 10:
            break

        U_try, V_try, W_try, r_try, err = try_am_init_from_strassen(target_rank)

        if err is not None and err < 1e-10:
            # Found valid decomposition
            best_rank = r_try
            best_U, best_V, best_W = U_try, V_try, W_try
            found_lower = True
            # Continue trying lower ranks
            continue
        else:
            # If we couldn't get close, stop
            if err is not None and err > 0.01:
                break
            # If close but not exact, try one more time with more iterations
            U_try2, V_try2, W_try2, r_try2, err2 = try_am_init_from_strassen(
                target_rank, n_iter=1000, n_restarts=5, seed=999)
            if err2 is not None and err2 < 1e-10:
                best_rank = r_try2
                best_U, best_V, best_W = U_try2, V_try2, W_try2
                found_lower = True
                continue
            break

    # If we found a lower rank, try to round to half-integers for exactness
    if found_lower and best_rank < 66:
        def try_round(M, denom):
            """Round to multiples of 1/denom."""
            return np.round(M * denom) / denom

        for denom in [2, 4, 8]:
            U_r = try_round(best_U, denom)
            V_r = try_round(best_V, denom)
            W_r = try_round(best_W, denom)
            if verify_exact(U_r, V_r, W_r, T, tol=1e-10):
                best_U, best_V, best_W = U_r, V_r, W_r
                break

    # Final verification
    if not verify_exact(best_U, best_V, best_W, T, tol=1e-8):
        best_U, best_V, best_W = U0.copy(), V0.copy(), W0.copy()
        best_rank = U0.shape[1]

    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()

    assert len(U_list) == n * m
    assert len(V_list) == m * p
    assert len(W_list) == p * n

    return {"U": U_list, "V": V_list, "W": W_list}
