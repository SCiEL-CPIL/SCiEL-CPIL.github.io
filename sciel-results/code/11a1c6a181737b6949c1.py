def residual(params):
    U = params[:9*R].reshape(3, 3, R) # i, j, c
    V = params[9*R:18*R].reshape(3, 3, R) # j, k, c
    W = params[18*R:].reshape(3, 3, R) # k, i, c
    T = np.einsum('ijc,jkc,kic->ijk', U, V, W)
    return (T - 1).flatten()
