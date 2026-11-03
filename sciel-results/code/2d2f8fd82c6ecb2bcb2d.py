import numpy as np
import time

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    nm = n * m
    mp = m * p
    pn = p * n

    # Build target tensor
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    valid_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
                valid_indices.append((a, b, c))

    # Try ranks from 70 to 84
    for rank in range(70, 85):
        if time.time() > deadline:
            break

        for restart in range(3):
            if time.time() > deadline:
                break

            np.random.seed(rank * 100 + restart)
            U = np.random.randn(nm, rank)
            V = np.random.randn(mp, rank)
            W = np.random.randn(pn, rank)

            for it in range(80):
                if time.time() > deadline:
                    break

                # Update U
                for a in range(nm):
                    M = np.zeros((mp * pn, rank))
                    for b in range(mp):
                        for c in range(pn):
                            M[b * pn + c, :] = V[b, :] * W[c, :]
                    rhs = T[a].ravel()
                    U[a, :] = np.linalg.lstsq(M.T, rhs, rcond=None)[0]

                # Update V
                for b in range(mp):
                    M = np.zeros((nm * pn, rank))
                    for a in range(nm):
                        for c in range(pn):
                            M[a * pn + c, :] = U[a, :] * W[c, :]
                    rhs = T[:, b, :].ravel()
                    V[b, :] = np.linalg.lstsq(M.T, rhs, rcond=None)[0]

                # Update W
                for c in range(pn):
                    M = np.zeros((nm * mp, rank))
                    for a in range(nm):
                        for b in range(mp):
                            M[a * mp + b, :] = U[a, :] * V[b, :]
                    rhs = T[:, :, c].ravel()
                    W[c, :] = np.linalg.lstsq(M.T, rhs, rcond=None)[0]

            # Check error
            err = 0.0
            for a, b, c in valid_indices:
                val = np.dot(U[a, :], V[b, :] * W[c, :])
                err += (val - 1.0) ** 2

            # Sample invalid entries
            np.random.seed(42)
            for _ in range(50):
                a = np.random.randint(nm)
                b = np.random.randint(mp)
                c = np.random.randint(pn)
                if T[a, b, c] == 0:
                    val = np.dot(U[a, :], V[b, :] * W[c, :])
                    err += val ** 2

            if err < 1e-6:
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2

                err2 = 0.0
                for a, b, c in valid_indices:
                    val = np.dot(U_r[a, :], V_r[b, :] * W_r[c, :])
                    err2 += (val - 1.0) ** 2
                np.random.seed(43)
                for _ in range(100):
                    a = np.random.randint(nm)
                    b = np.random.randint(mp)
                    c = np.random.randint(pn)
                    if T[a, b, c] == 0:
                        val = np.dot(U_r[a, :], V_r[b, :] * W_r[c, :])
                        err2 += val ** 2

                if err2 < 1e-6:
                    return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                else:
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
