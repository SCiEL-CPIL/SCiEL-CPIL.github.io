def _als(T, r, iters, seed):
    A, B, C = T.shape
    rng = np.random.default_rng(seed)
    U = rng.normal(size=(A, r)) * 0.7
    V = rng.normal(size=(B, r)) * 0.7
    W = rng.normal(size=(C, r)) * 0.7
    Tr_a = T.reshape(A, B * C)
    Tr_b = T.transpose(1, 0, 2).reshape(B, A * C)
    Tr_c = T.transpose(2, 0, 1).reshape(C, A * B)
    res = None
    for it in range(iters):
        M = (V[:, :, None] * W[None, :, :]).reshape(r, B * C)
        U = np.linalg.lstsq(M.T, Tr_a, rcond=None)[0]
        N = (U[:, :, None] * W[None, :, :]).reshape(r, A * C)
        V = np.linalg.lstsq(N.T, Tr_b, rcond=None)[0]
        P = (U[:, :, None] * V[None, :, :]).reshape(r, A * B)
        W = np.linalg.lstsq(P.T, Tr_c, rcond=None)[0]
        approx = np.einsum('at,bt,ct->abc', U, V, W)
        res = float(np.linalg.norm(approx - T))
        if res < 1e-11:
            break
    return U, V, W, res
