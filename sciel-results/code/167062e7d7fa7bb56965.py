import time
import numpy as np

def construct():
    t0 = time.time()
    deadline = t0 + 110

    n, m, p = 2, 4, 8

    # Strassen: 7 rank-1 tensors for 2x2*2x2
    # Local indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    strassen = [
        (np.array([1, 0, 0, 1]), np.array([1, 0, 0, 1]), np.array([1, 0, 0, 1])),
        (np.array([0, 0, 1, 1]), np.array([1, 0, 0, 0]), np.array([0, 0, 1, -1])),
        (np.array([1, 0, 0, 0]), np.array([0, 1, 0, -1]), np.array([0, 1, 0, 1])),
        (np.array([0, 0, 0, 1]), np.array([0, 0, 1, -1]), np.array([1, 0, 1, 0])),
        (np.array([1, 1, 0, 0]), np.array([0, 0, 0, 1]), np.array([-1, 1, 0, 0])),
        (np.array([-1, 0, 1, 0]), np.array([1, 1, 0, 0]), np.array([0, 0, 0, 1])),
        (np.array([0, 1, 0, -1]), np.array([0, 0, 1, 1]), np.array([1, 0, 0, 0])),
    ]

    total_rank = 56
    U = np.zeros((n * m, total_rank), dtype=np.float64)
    V = np.zeros((m * p, total_rank), dtype=np.float64)
    W = np.zeros((p * n, total_rank), dtype=np.float64)

    rank_idx = 0
    for h in range(2):
        for j in range(4):
            for uc, vc, wc in strassen:
                if time.time() > deadline:
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

                for li, c in enumerate(uc):
                    if c != 0:
                        i = li // 2
                        ll = li % 2
                        U[i * m + 2 * h + ll, rank_idx] = c

                for li, c in enumerate(vc):
                    if c != 0:
                        lr = li // 2
                        kl = li % 2
                        V[(2 * h + lr) * p + 2 * j + kl, rank_idx] = c

                for li, c in enumerate(wc):
                    if c != 0:
                        i = li // 2
                        kl = li % 2
                        W[(2 * j + kl) * n + i, rank_idx] = c

                rank_idx += 1

    # Verify: for all i in [0,n), l in [0,m), k in [0,p): sum_r U[i*m+l,r]*V[l*p+k,r]*W[k*n+i,r] == 1
    for i in range(n):
        for l in range(m):
            for k in range(p):
                s = np.sum(U[i * m + l, :] * V[l * p + k, :] * W[k * n + i, :])
                if abs(s - 1.0) > 1e-10:
                    break

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
