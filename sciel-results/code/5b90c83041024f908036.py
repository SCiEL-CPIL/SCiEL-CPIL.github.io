def _verify(U, V, W, n, m, p):
    r = U.shape[1]
    # Reshape to tensors
    # U: (n, m, r)
    # V: (m, p, r)
    # W: (p, n, r)
    U_t = U.reshape(n, m, r)
    V_t = V.reshape(m, p, r)
    W_t = W.reshape(p, n, r)
    
    # Compute T[i, j, jp, k] = sum_c W[k, i, c] U[i, j, c] V[jp, k, c]
    # This is not a standard einsum.
    # Let's just loop.
    pass
