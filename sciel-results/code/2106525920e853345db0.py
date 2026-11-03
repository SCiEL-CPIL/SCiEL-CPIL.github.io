<<<<<<< SEARCH
def _try_better_decomposition(deadline):
    """
    Try to find a decomposition with rank < 84 using ALS search.
    Uses vectorized ALS updates for speed, then attempts rank reduction
    by zeroing columns and re-solving.
    """
    import numpy as np
    import time

    n, m, p = 3, 5, 6
    nm = n * m  # 15
    mp = m * p  # 30
    pn = p * n  # 18
    T = _build_target_tensor(n, m, p)

    # Start from block decomposition
    U0, V0, W0 = _block_decomposition_356()
    r0 = U0.shape[1]  # 84

    best_result = None
    best_rank = r0

    # Try to reduce rank by dropping columns and re-solving
    for target_rank in range(r0 - 1, max(r0 - 8, 75), -1):
        if time.time() > deadline - 10:
            break

        # Initialize from the top target_rank columns of the block decomposition
        # But first, let's try to get a better initialization via SVD
        T_mat = T.reshape(nm, mp * pn)
        try:
            U_svd, s, Vt_svd = np.linalg.svd(T_mat, full_matrices=False)
            # Use top target_rank singular vectors for U
            U = U_svd[:, :target_rank].copy()
            # For V and W, use random initialization scaled
            np.random.seed(42 + target_rank)
            V = np.random.randn(mp, target_rank) * 0.1
            W = np.random.randn(pn, target_rank) * 0.1
        except Exception:
            U = U0[:, :target_rank].copy()
            V = V0[:, :target_rank].copy()
            W = W0[:, :target_rank].copy()

        # Run ALS iterations with vectorized updates
        max_iter = 30
        tol = 1e-8
        prev_err = float('inf')

        for it in range(max_iter):
            if time.time() > deadline - 5:
                break

            # Update U: Z_flat[b*pn+c, k] = V[b,k] * W[c,k]
            # Vectorized: Z_flat = np.einsum('bk,ck->bck', V, W).reshape(mp*pn, target_rank)
            Z_flat = np.einsum('bk,ck->bck', V, W, optimize=True).reshape(mp * pn, target_rank)
            # U[a,:] = lstsq(Z_flat, T[a,:,:].flatten())
            # Vectorized: U = lstsq(Z_flat, T.reshape(nm, -1).T).T
            T_flat = T.reshape(nm, mp * pn)
            try:
                U = np.linalg.lstsq(Z_flat, T_flat.T, rcond=None)[0].T
            except Exception:
                pass

            # Update V: Z_flat_V[a*pn+c, k] = U[a,k] * W[c,k]
            Z_flat_V = np.einsum('ak,ck->ack', U, W, optimize=True).reshape(nm * pn, target_rank)
            T_flat_V = T.transpose(1, 0, 2).reshape(mp, nm * pn)
            try:
                V = np.linalg.lstsq(Z_flat_V, T_flat_V.T, rcond=None)[0].T
            except Exception:
                pass

            # Update W: Z_flat_W[a*mp+b, k] = U[a,k] * V[b,k]
            Z_flat_W = np.einsum('ak,bk->abk', U, V, optimize=True).reshape(nm * mp, target_rank)
            T_flat_W = T.transpose(2, 0, 1).reshape(pn, nm * mp)
            try:
                W = np.linalg.lstsq(Z_flat_W, T_flat_W.T, rcond=None)[0].T
            except Exception:
                pass

            # Compute error
            T_comp = np.einsum('ak,bk,ck->abc', U, V, W, optimize=True)
            err = np.linalg.norm(T_comp - T)
            if abs(prev_err - err) < tol:
                break
            prev_err = err

        # Verify the result
        if _verify_decomp(n, m, p, U, V, W):
            if target_rank < best_rank:
                best_rank = target_rank
                best_result = (U.copy(), V.copy(), W.copy())
                break

    if best_result is not None:
        return best_result
    return None
=======
def _als_refine(U, V, W, T, max_iter=80, tol=1e-12, deadline=None):
    """
    ALS refinement with tight tolerance. U:(nm,r), V:(mp,r), W:(pn,r), T:(nm,mp,pn).
    Returns (U, V, W, final_error).
    """
    nm, mp, pn = T.shape
    r = U.shape[1]
    U = U.copy()
    V = V.copy()
    W = W.copy()
    T_flat_U = T.reshape(nm, mp * pn)
    T_flat_V = T.transpose(1, 0, 2).reshape(mp, nm * pn)
    T_flat_W = T.transpose(2, 0, 1).reshape(pn, nm * mp)
    prev_err = float('inf')
    final_err = float('inf')
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline - 3:
            break
        # Update U
        ZU = np.einsum('bk,ck->bck', V, W, optimize=True).reshape(mp * pn, r)
        try:
            U = np.linalg.lstsq(ZU, T_flat_U.T, rcond=None)[0].T
        except Exception:
            break
        # Update V
        ZV = np.einsum('ak,ck->ack', U, W, optimize=True).reshape(nm * pn, r)
        try:
            V = np.linalg.lstsq(ZV, T_flat_V.T, rcond=None)[0].T
        except Exception:
            break
        # Update W
        ZW = np.einsum('ak,bk->abk', U, V, optimize=True).reshape(nm * mp, r)
        try:
            W = np.linalg.lstsq(ZW, T_flat_W.T, rcond=None)[0].T
        except Exception:
            break
        T_comp = np.einsum('ak,bk,ck->abc', U, V, W, optimize=True)
        final_err = np.linalg.norm(T_comp - T)
        if abs(prev_err - final_err) < tol:
            break
        prev_err = final_err
    return U, V, W, final_err

def _try_better_decomposition(deadline):
    """
    Try to find a decomposition with rank < 84 using SVD-initialized ALS.
    Uses rank-revealing SVD to get good initializations for all three factors,
    then refines with ALS. Tries multiple target ranks and restarts.
    """
    n, m, p = 3, 5, 6
    nm, mp, pn = n * m, m * p, p * n  # 15, 30, 18
    T = _build_target_tensor(n, m, p)
    U0, V0, W0 = _block_decomposition_356()
    r0 = U0.shape[1]  # 84

    best_result = None
    best_rank = r0

    # Precompute SVDs for initialization
    T_mat_U = T.reshape(nm, mp * pn)
    T_mat_V = T.transpose(1, 0, 2).reshape(mp, nm * pn)
    T_mat_W = T.transpose(2, 0, 1).reshape(pn, nm * mp)
    U_svd, sU, _ = np.linalg.svd(T_mat_U, full_matrices=False)
    V_svd, sV, _ = np.linalg.svd(T_mat_V, full_matrices=False)
    W_svd, sW, _ = np.linalg.svd(T_mat_W, full_matrices=False)

    # Try target ranks from r0-1 down, with multiple restarts
    for target_rank in range(r0 - 1, max(r0 - 15, 70), -1):
        if time.time() > deadline - 15:
            break

        # Number of restarts based on remaining time
        remaining = deadline - time.time()
        n_restarts = max(1, min(3, int(remaining / 20)))

        found = False
        for restart in range(n_restarts):
            if time.time() > deadline - 5:
                break

            # SVD-based initialization with random sign flips for diversity
            U = U_svd[:, :target_rank].copy()
            V = V_svd[:, :target_rank].copy()
            W = W_svd[:, :target_rank].copy()
            if restart > 0:
                np.random.seed(1000 + target_rank * 7 + restart)
                signs = np.random.choice([-1, 1], size=target_rank)
                U *= signs
                V *= signs
                W *= signs

            U, V, W, err = _als_refine(U, V, W, T, max_iter=60, tol=1e-12, deadline=deadline)

            if err < 1e-9:
                # Verify exactly
                if _verify_decomp(n, m, p, U, V, W):
                    if target_rank < best_rank:
                        best_rank = target_rank
                        best_result = (U.copy(), V.copy(), W.copy())
                        found = True
                        break

        if found:
            break

    if best_result is not None:
        return best_result
    return None
>>>>>>> REPLACE

Now let me also update the `construct` function to use a tighter deadline and add a final polish pass:

