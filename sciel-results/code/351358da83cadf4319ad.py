import numpy as np
import time

def construct():
    n, m, p = 2, 3, 6
    deadline = time.time() + 140

    # Block decomposition: split B (3x6) into B1 (3x3) and B2 (3x3)
    # A*B = A*B1 + A*B2, each is 2x3x3 multiplication
    n2, m2, p2 = 2, 3, 3
    T2 = np.zeros((n2*m2, m2*p2, p2*n2))
    for i in range(n2):
        for j in range(m2):
            for k in range(p2):
                T2[i*m2+j, j*p2+k, k*n2+i] = 1.0

    def try_decompose(r, seed=42, max_time=20):
        t0 = time.time()
        rng = np.random.RandomState(seed)
        U = rng.randn(n2*m2, r)
        V = rng.randn(m2*p2, r)
        W = rng.randn(p2*n2, r)

        best_err = np.inf
        for it in range(20000):
            if time.time() - t0 > max_time:
                break
            err = 0.0
            for a in range(n2*m2):
                for b in range(m2*p2):
                    for c in range(p2*n2):
                        val = np.dot(U[a,:], V[b,:] * W[c,:])
                        err += (T2[a,b,c] - val)**2
            if err < best_err:
                best_err = err
                if err < 1e-10:
                    return U.copy(), V.copy(), W.copy()
            U += 0.05 * rng.randn(n2*m2, r)
            V += 0.05 * rng.randn(m2*p2, r)
            W += 0.05 * rng.randn(p2*n2, r)
        return None

    best_2x3x3 = None
    for r in [17, 16, 15]:
        if time.time() > deadline - 40:
            break
        res = try_decompose(r, seed=42, max_time=15)
        if res is not None:
            best_2x3x3 = (r, res)
            break

    if best_2x3x3 is not None:
        r, (U1, V1, W1) = best_2x3x3
        total_rank = 2 * r
        U = np.zeros((n*m, total_rank))
        V = np.zeros((m*p, total_rank))
        W = np.zeros((p*n, total_rank))
        U[:, 0:r] = U1
        V[0:9, 0:r] = V1
        W[0:6, 0:r] = W1
        U[:, r:2*r] = U1
        V[9:18, r:2*r] = V1
        W[6:12, r:2*r] = W1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
