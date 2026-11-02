def _als(U, V, W, T, R, max_iters, deadline):
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    Um, Vm, Wm = U.shape[0], V.shape[0], W.shape[0]
    for it in range(max_iters):
        if time.time() > deadline:
            break
        # Update U
        A = (V[:, None, :] * W[None, :, :]).reshape(Vm*Wm, R)  # (Vm*Wm, R)
        for u in range(Um):
            U[u] = np.linalg.lstsq(A, T[u].reshape(-1), rcond=None)[0]
        # Update V
        A = (U[:, None, :] * W[None, :, :]).reshape(Um*Wm, R)
        for v in range(Vm):
            V[v] = np.linalg.lstsq(A, T[:, v, :].reshape(-1), rcond=None)[0]
        # Update W
        A = (U[:, None, :] * V[None, :, :]).reshape(Um*Vm, R)
        for w in range(Wm):
            W[w] = np.linalg.lstsq(A, T[:, :, w].reshape(-1), rcond=None)[0]
    return U, V, W
