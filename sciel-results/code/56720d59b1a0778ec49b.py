import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = 2, 4, 4
    nm, mp, pn = n * m, m * p, p * n  # 8, 16, 8

    # Build the tensor T: T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1

    # Matrix-rank lower bounds
    T1 = T.reshape(nm, mp * pn)
    T2 = T.transpose(1, 0, 2).reshape(mp, nm * pn)
    T3 = T.transpose(2, 0, 1).reshape(pn, nm * mp)
    rank_lb = max(np.linalg.matrix_rank(T1), np.linalg.matrix_rank(T2), np.linalg.matrix_rank(T3))

    best_result = None

    for rank in range(rank_lb, 33):
        if time.time() > deadline:
            break
        found = False
        for trial in range(300):
            if time.time() > deadline:
                break
            np.random.seed(trial * 1000 + rank * 100)
            U = np.random.randn(nm, rank)
            V = np.random.randn(mp, rank)

            # Linear system for W given U, V
            A = np.zeros((nm * mp * pn, pn * rank))
            for a in range(nm):
                for b in range(mp):
                    for c in range(pn):
                        eq = a * mp * pn + b * pn + c
                        for cc in range(rank):
                            A[eq, c * rank + cc] = U[a, cc] * V[b, cc]

            b_vec = T.flatten()
            try:
                W_flat, _, _, _ = np.linalg.lstsq(A, b_vec, rcond=None)
                W = W_flat.reshape(pn, rank)

                # Verify reconstruction
                T_recon = np.zeros((nm, mp, pn))
                for cc in range(rank):
                    T_recon += np.einsum('a,b,c->abc', U[:, cc], V[:, cc], W[:, cc])
                max_err = np.max(np.abs(T_recon - T))

                if max_err < 1e-9:
                    # Try half-integer rounding
                    Uh = np.round(U * 2) / 2
                    Vh = np.round(V * 2) / 2
                    Wh = np.round(W * 2) / 2
                    T_recon_h = np.zeros((nm, mp, pn))
                    for cc in range(rank):
                        T_recon_h += np.einsum('a,b,c->abc', Uh[:, cc], Vh[:, cc], Wh[:, cc])
                    max_err_h = np.max(np.abs(T_recon_h - T))

                    if max_err_h < 1e-9:
                        result = {"U": Uh.tolist(), "V": Vh.tolist(), "W": Wh.tolist()}
                    else:
                        result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

                    best_result = (rank, result)
                    found = True
                    break
            except Exception:
                pass

        if found:
            break

    if best_result is not None:
        return best_result[1]

    # Fallback: schoolbook rank 32
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
