def verify(U, V, W):
    U3 = np.array(U, dtype=float).reshape(n, m, -1)
    V3 = np.array(V, dtype=float).reshape(m, p, -1)
    W3 = np.array(W, dtype=float).reshape(p, n, -1)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = float(np.dot(U3[i, j, :], V3[j, k, :] * W3[k, i, :]))
                if abs(s - 1.0) > 1e-9:
                    return False
    return True
