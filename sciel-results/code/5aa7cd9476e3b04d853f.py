import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    schoolbook_rank = n * m * p

    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    U_sb = np.zeros((n * m, schoolbook_rank))
    V_sb = np.zeros((m * p, schoolbook_rank))
    W_sb = np.zeros((p * n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    best_result = {'U': U_sb.tolist(), 'V': V_sb.tolist(), 'W': W_sb.tolist()}
    best_rank = schoolbook_rank

    np.random.seed(42)

    def verify(U, V, W, tol=1e-6):
        recon = np.einsum('ar,br,cr->abc', U, V, W)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if abs(recon[i * m + j, j * p + k, k * n + i] - 1.0) > tol:
                        return False
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    nonzero = any(i * m + j == a and j * p + k == b and k * n + i == c
                                  for i in range(n) for j in range(m) for k in range(p))
                    if not nonzero and abs(recon[a, b, c]) > tol:
                        return False
        return True

    def solve_rank(r, n_restarts=2, max_iters=15):
        for _ in range(n_restarts):
            if time.time() > deadline - 5:
                return None
            U = np.random.randn(n * m, r) * 0.1
            V = np.random.randn(m * p, r) * 0.1
            W = np.random.randn(p * n, r) * 0.1
            for it in range(max_iters):
                if time.time() > deadline - 5:
                    return None
                for c in range(p * n):
                    M = (U[:, None, :] * V[None, :, :]).reshape(n * m * m * p, r)
                    W[c] = np.linalg.lstsq(M, T[:, :, c].ravel(), rcond=None)[0]
                for a in range(n * m):
                    M = (V[:, None, :] * W[None, :, :]).reshape(m * p * p * n, r)
                    U[a] = np.linalg.lstsq(M, T[a, :, :].ravel(), rcond=None)[0]
                for b in range(m * p):
                    M = (U[:, None, :] * W[None, :, :]).reshape(n * m * p * n, r)
                    V[b] = np.linalg.lstsq(M, T[:, b, :].ravel(), rcond=None)[0]
                if np.max(np.abs(np.einsum('ar,br,cr->abc', U, V, W) - T)) < 1e-8:
                    if verify(U, V, W):
                        return U, V, W
        return None

    for r in range(21, schoolbook_rank):
        if time.time() > deadline - 10:
            break
        res = solve_rank(r)
        if res:
            best_rank = r
            best_result = {'U': res[0].tolist(), 'V': res[1].tolist(), 'W': res[2].tolist()}
            break

    return best_result
