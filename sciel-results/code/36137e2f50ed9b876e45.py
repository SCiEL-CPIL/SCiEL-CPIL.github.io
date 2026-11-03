<<<<<<< SEARCH
def _try_better_decomposition(deadline):
    """
    Try to find a decomposition with rank < 84 using sparse ALS search.
    """
    import numpy as np
    import time

    n, m, p = 3, 5, 6
    T = _build_target_tensor(n, m, p)
    nm = n * m  # 15
    mp = m * p  # 30
    pn = p * n  # 18

    # Start from block decomposition
    U0, V0, W0 = _block_decomposition_356()
    r0 = U0.shape[1]  # 84

    best_result = None
    best_rank = r0

    # Try ranks 83 down to 80
    for target_rank in range(83, 79, -1):
        if time.time() > deadline - 10:
            break

        # Multiple restarts
        for restart in range(5):
            if time.time() > deadline - 5:
                break

            np.random.seed(42 + restart)
            
            if restart == 0:
                U = U0[:, :target_rank].copy()
                V = V0[:, :target_rank].copy()
                W = W0[:, :target_rank].copy()
            else:
                # Random initialization with small noise from SVD
                T_mat = T.reshape(nm, mp * pn)
                try:
                    U_svd, s, Vt_svd = np.linalg.svd(T_mat, full_matrices=False)
                    U = U_svd[:, :target_rank] * np.random.randn(nm, target_rank)
                    V = Vt_svd[:target_rank, :].T * np.random.randn(mp, target_rank)
                    W = np.random.randn(pn, target_rank)
                except Exception:
                    U = np.random.randn(nm, target_rank)
                    V = np.random.randn(mp, target_rank)
                    W = np.random.randn(pn, target_rank)

            # Run ALS iterations
            max_iter = 100
            tol = 1e-8
            prev_err = float('inf')

            for it in range(max_iter):
                if time.time() > deadline - 2:
                    break

                # Update U
                Z_flat = np.zeros((mp * pn, target_rank))
                for b in range(mp):
                    for c in range(pn):
                        Z_flat[b * pn + c, :] = V[b, :] * W[c, :]

                U_new = np.zeros((nm, target_rank))
                for a in range(nm):
                    rhs = T[a, :, :].flatten()
                    try:
                        U_new[a, :] = np.linalg.lstsq(Z_flat, rhs, rcond=None)[0]
                    except Exception:
                        U_new[a, :] = U[a, :]

                # Update V
                Z_flat_V = np.zeros((nm * pn, target_rank))
                for a in range(nm):
                    for c in range(pn):
                        Z_flat_V[a * pn + c, :] = U[a, :] * W[c, :]

                V_new = np.zeros((mp, target_rank))
                for b in range(mp):
                    rhs = T[:, b, :].flatten()
                    try:
                        V_new[b, :] = np.linalg.lstsq(Z_flat_V, rhs, rcond=None)[0]
                    except Exception:
                        V_new[b, :] = V[b, :]

                # Update W
                Z_flat_W = np.zeros((nm * mp, target_rank))
                for a in range(nm):
                    for b in range(mp):
                        Z_flat_W[a * mp + b, :] = U[a, :] * V[b, :]

                W_new = np.zeros((pn, target_rank))
                for c in range(pn):
                    rhs = T[:, :, c].flatten()
                    try:
                        W_new[c, :] = np.linalg.lstsq(Z_flat_W, rhs, rcond=None)[0]
                    except Exception:
                        W_new[c, :] = W[c, :]

                U, V, W = U_new, V_new, W_new

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
                    break  # Found a valid decomposition at this rank, try even lower

        if best_result is not None and best_rank < target_rank:
            break

    if best_result is not None:
        return best_result
    return None
=======
def _als_update_vectorized(T, U, V, W, nm, mp, pn, r, deadline):
    """
    Perform one full ALS cycle (update U, V, W) using vectorized operations.
    Returns updated (U, V, W) and error norm.
    """
    # Update U: for each a, solve Z_VW @ U[a,:] = T[a,:,:].flatten()
    # Z_VW[b*pn+c, k] = V[b,k] * W[c,k]
    Z_VW = np.einsum('bk,ck->bck', V, W)  # (mp, pn, r)
    Z_VW_flat = Z_VW.reshape(mp * pn, r)
    
    U_new = np.zeros((nm, r))
    for a in range(nm):
        rhs = T[a, :, :].flatten()
        try:
            U_new[a, :] = np.linalg.lstsq(Z_VW_flat, rhs, rcond=None)[0]
        except Exception:
            U_new[a, :] = U[a, :]

    # Update V: for each b, solve Z_UW @ V[b,:] = T[:,b,:].flatten()
    Z_UW = np.einsum('ak,ck->ack', U, W)  # (nm, pn, r)
    Z_UW_flat = Z_UW.reshape(nm * pn, r)
    
    V_new = np.zeros((mp, r))
    for b in range(mp):
        rhs = T[:, b, :].flatten()
        try:
            V_new[b, :] = np.linalg.lstsq(Z_UW_flat, rhs, rcond=None)[0]
        except Exception:
            V_new[b, :] = V[b, :]

    # Update W: for each c, solve Z_UV @ W[c,:] = T[:,:,c].flatten()
    Z_UV = np.einsum('ak,bk->abk', U, V)  # (nm, mp, r)
    Z_UV_flat = Z_UV.reshape(nm * mp, r)
    
    W_new = np.zeros((pn, r))
    for c in range(pn):
        rhs = T[:, :, c].flatten()
        try:
            W_new[c, :] = np.linalg.lstsq(Z_UV_flat, rhs, rcond=None)[0]
        except Exception:
            W_new[c, :] = W[c, :]

    T_comp = np.einsum('ak,bk,ck->abc', U_new, V_new, W_new, optimize=True)
    err = np.linalg.norm(T_comp - T)
    return U_new, V_new, W_new, err

def _try_better_decomposition(deadline):
    """
    Try to find a decomposition with rank < 84 using vectorized ALS search.
    """
    import numpy as np
    import time

    n, m, p = 3, 5, 6
    T = _build_target_tensor(n, m, p)
    nm = n * m  # 15
    mp = m * p  # 30
    pn = p * n  # 18

    # Start from block decomposition
    U0, V0, W0 = _block_decomposition_356()
    r0 = U0.shape[1]  # 84

    best_result = None
    best_rank = r0

    # Try ranks 83 down to 78
    for target_rank in range(83, 77, -1):
        if time.time() > deadline - 15:
            break

        # Multiple restarts
        max_restarts = 8
        for restart in range(max_restarts):
            if time.time() > deadline - 10:
                break

            if restart == 0:
                # Truncate the block decomposition
                U = U0[:, :target_rank].copy()
                V = V0[:, :target_rank].copy()
                W = W0[:, :target_rank].copy()
            elif restart == 1:
                # Perturbed truncation
                np.random.seed(100 + restart)
                U = U0[:, :target_rank].copy() + 0.01 * np.random.randn(nm, target_rank)
                V = V0[:, :target_rank].copy() + 0.01 * np.random.randn(mp, target_rank)
                W = W0[:, :target_rank].copy() + 0.01 * np.random.randn(pn, target_rank)
            elif restart == 2:
                # SVD-based initialization
                T_mat = T.reshape(nm, mp * pn)
                try:
                    U_svd, s, Vt_svd = np.linalg.svd(T_mat, full_matrices=False)
                    np.random.seed(200 + restart)
                    U = U_svd[:, :min(target_rank, nm)]
                    if target_rank > nm:
                        U = np.hstack([U, np.random.randn(nm, target_rank - nm)])
                    V = Vt_svd[:min(target_rank, mp * pn), :].T
                    if V.shape[0] < mp:
                        V = np.vstack([V, np.random.randn(mp - V.shape[0], target_rank)])
                    V = V[:mp, :]
                    np.random.seed(300 + restart)
                    W = np.random.randn(pn, target_rank)
                except Exception:
                    np.random.seed(400 + restart)
                    U = np.random.randn(nm, target_rank)
                    V = np.random.randn(mp, target_rank)
                    W = np.random.randn(pn, target_rank)
            else:
                # Random initialization
                np.random.seed(500 + restart)
                U = np.random.randn(nm, target_rank)
                V = np.random.randn(mp, target_rank)
                W = np.random.randn(pn, target_rank)
                # Scale to match tensor magnitude
                U = U / np.linalg.norm(U) * np.sqrt(nm)
                V = V / np.linalg.norm(V) * np.sqrt(mp)
                W = W / np.linalg.norm(W) * np.sqrt(pn)

            # Run ALS iterations
            max_iter = 200
            tol = 1e-10
            prev_err = float('inf')
            stall_count = 0

            for it in range(max_iter):
                if time.time() > deadline - 5:
                    break

                U, V, W, err = _als_update_vectorized(T, U, V, W, nm, mp, pn, target_rank, deadline)

                if abs(prev_err - err) < tol:
                    stall_count += 1
                    if stall_count > 10:
                        break
                else:
                    stall_count = 0
                prev_err = err

                # Early exit if error is very small
                if err < 1e-7:
                    break

            # Verify the result
            if _verify_decomp(n, m, p, U, V, W):
                if target_rank < best_rank:
                    best_rank = target_rank
                    best_result = (U.copy(), V.copy(), W.copy())
                    break  # Found a valid decomposition at this rank, try even lower

        if best_result is not None and best_rank < target_rank:
            break

    if best_result is not None:
        return best_result
    return None
>>>>>>> REPLACE

Now let me also improve the `construct` function to use the deadline more effectively and add a final verification with tighter tolerance:

