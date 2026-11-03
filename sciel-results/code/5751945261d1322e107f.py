import numpy as np

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    n_m = n * m
    m_p = m * p
    p_n = p * n

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n_m)]
        V = [[0] * rank for _ in range(m_p)]
        W = [[0] * rank for _ in range(p_n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Build target tensor
    T = np.zeros((n_m, m_p, p_n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    # Greedy CP decomposition
    residual = T.copy()
    U_cols, V_cols, W_cols = [], [], []
    np.random.seed(42)

    for r in range(150):
        if np.linalg.norm(residual) < 1e-9:
            break
        u = np.random.randn(n_m)
        v = np.random.randn(m_p)
        w = np.random.randn(p_n)
        for _ in range(300):
            vw = np.outer(v, w)
            u = np.einsum('abc,bc->a', residual, vw) / (np.dot(vw.ravel(), vw.ravel()) + 1e-15)
            uw = np.outer(u, w)
            v = np.einsum('abc,ac->b', residual, uw) / (np.dot(uw.ravel(), uw.ravel()) + 1e-15)
            uv = np.outer(u, v)
            w = np.einsum('abc,ab->c', residual, uv) / (np.dot(uv.ravel(), uv.ravel()) + 1e-15)
        residual -= np.einsum('a,b,c->abc', u, v, w)
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)

    U = np.column_stack(U_cols)
    V = np.column_stack(V_cols)
    W = np.column_stack(W_cols)

    # Round to half-integers
    U_r = np.round(U * 2) / 2
    V_r = np.round(V * 2) / 2
    W_r = np.round(W * 2) / 2

    # Verify exactness
    T_recon = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
    if np.allclose(T_recon, T, atol=1e-9):
        return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
    else:
        return schoolbook()
