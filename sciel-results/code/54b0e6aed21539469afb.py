def construct():
    start = time.time()
    deadline = start + 135
    n, m, p = DIMENSIONS
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    T1 = T.reshape(n*m, m*p*p*n)
    T2 = T.reshape(m*p, n*m*p*n)
    T3 = T.reshape(p*n, n*m*m*p)
    lb = max(np.linalg.matrix_rank(T1), np.linalg.matrix_rank(T2), np.linalg.matrix_rank(T3))
    best = None
    for R in range(lb, lb + 50):
        if time.time() > deadline - 20:
            break
        d = _als_decomp(T, R, deadline=deadline - 20)
        if d is not None:
            U, V, W = d
            best = (R, U, V, W)
            break
    if best is None:
        R = n*m*p
        U = np.zeros((n*m, R)); V = np.zeros((m*p, R)); W = np.zeros((p*n, R))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, c] = 1; V[j*p+k, c] = 1; W[k*n+i, c] = 1; c += 1
    else:
        R, U, V, W = best
        U = np.round(U*2)/2; V = np.round(V*2)/2; W = np.round(W*2)/2
        if not np.allclose(np.einsum('ar,br,cr->abc', U, V, W), T, atol=1e-6):
            R = n*m*p
            U = np.zeros((n*m, R)); V = np.zeros((m*p, R)); W = np.zeros((p*n, R))
            c = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i*m+j, c] = 1; V[j*p+k, c] = 1; W[k*n+i, c] = 1; c += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
