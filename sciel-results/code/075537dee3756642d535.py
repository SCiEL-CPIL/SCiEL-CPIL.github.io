import numpy as np

DIMENSIONS = [2, 3, 6]


def _strassen_block():
    """Return (U2, V2, W2) rank-7 for 2x2x2, or None if not found."""
    u = np.array([
        [1, 0, 0, 1],   # a+d
        [0, 0, 1, 1],   # c+d
        [1, -1, 0, 0],  # a-b
        [0, 0, 1, 1],   # c+d
        [1, 1, 0, 0],   # a+b
        [0, 0, 1, -1],  # c-a
        [0, 1, 0, -1],  # b-d
    ], dtype=float)
    v = np.array([
        [1, 0, 0, 1],   # e+h
        [1, 0, 0, 1],   # e+h
        [1, 1, 0, 0],   # e+f
        [0, 0, 1, 1],   # g+h
        [0, 0, 0, 1],   # h
        [1, 1, 0, 0],   # e+f
        [0, 0, 1, 1],   # g+h
    ], dtype=float)
    alphas = np.zeros((2, 2, 7))
    for i in range(2):
        for k2 in range(2):
            M = np.zeros((4, 7))
            for t in range(7):
                ui = [u[t, 0 * 2 + i], u[t, 1 * 2 + i]]
                vk = [v[t, 0 * 2 + k2], v[t, 1 * 2 + k2]]
                M[:, t] = np.outer(ui, vk)
            target = np.eye(2).flatten()
            sol, _, _, _ = np.linalg.lstsq(M, target, rcond=None)
            if np.linalg.norm(M @ sol - target) > 1e-8:
                return None
            alphas[i, k2, :] = sol
    U2 = np.zeros((4, 7)); V2 = np.zeros((4, 7)); W2 = np.zeros((4, 7))
    for t in range(7):
        for i in range(2):
            for j in range(2):
                U2[j * 2 + i, t] = u[t, j * 2 + i]
        for j2 in range(2):
            for k2 in range(2):
                V2[k2 * 2 + j2, t] = v[t, k2 * 2 + j2]
        for i in range(2):
            for k2 in range(2):
                W2[k2 * 2 + i, t] = alphas[i, k2, t]
    return U2, V2, W2


def construct():
    n, m, p = DIMENSIONS
    block = _strassen_block()
    if block is not None:
        U2, V2, W2 = block
        U, V, W = [], [], []
        for blk in range(3):
            co = blk * 2
            for t in range(7):
                u_row = [0.0] * (n * m)
                v_row = [0.0] * (m * p)
                w_row = [0.0] * (p * n)
                for i in range(2):
                    for j in range(2):
                        u_row[i * m + j] = U2[j * 2 + i, t]
                for j2 in range(2):
                    for k2 in range(2):
                        v_row[j2 * p + (co + k2)] = V2[k2 * 2 + j2, t]
                for i in range(2):
                    for k2 in range(2):
                        w_row[(co + k2) * n + i] = W2[k2 * 2 + i, t]
                U.append(u_row); V.append(v_row); W.append(w_row)
        for i in range(2):
            for k in range(6):
                u_row = [0.0] * (n * m)
                v_row = [0.0] * (m * p)
                w_row = [0.0] * (p * n)
                u_row[i * m + 2] = 1.0
                v_row[2 * p + k] = 1.0
                w_row[k * n + i] = 1.0
                U.append(u_row); V.append(v_row); W.append(w_row)
        rank = len(U)
        Ua = np.array(U, dtype=float); Va = np.array(V, dtype=float); Wa = np.array(W, dtype=float)
        ok = True
        for _ in range(5):
            A = np.random.rand(n, m); B = np.random.rand(m, p)
            C_true = A @ B
            C_rec = np.zeros((n, p))
            a_flat = A.reshape(n * m); b_flat = B.reshape(m * p)
            for t in range(rank):
                sc = (Ua[:, t] @ a_flat) * (Va[:, t] @ b_flat)
                C_rec += (Wa[:, t].reshape(p, n) * sc).T
            if not np.allclose(C_rec, C_true, atol=1e-8):
                ok = False
                break
        if ok:
            return {"U": U, "V": V, "W": W}
    # Fallback: schoolbook rank 36
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}
