"""Matrix multiplication 2x4, 4x6 rank-minimal decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 4, 6]

def _t0(i, m, j): return i * m + j
def _t1(j, p, k): return j * p + k
def _t2(k, n, i): return k * n + i

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 115.0

    N_t = n * m   # 8
    M_t = m * p   # 24
    P_t = p * n   # 12

    # Build target tensor
    target = np.zeros((P_t, M_t, N_t), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[_t2(k, n, i), _t1(j, p, k), _t0(i, m, j)] = 1.0

    def residual(U, V, W):
        r = U.shape[1]
        approx = np.zeros_like(target)
        for c in range(r):
            approx += np.einsum('C,B,A->CBA', W[:, c], V[:, c], U[:, c])
        return np.linalg.norm(approx - target)

    def schoolbook():
        r = n * m * p
        U = np.zeros((N_t, r))
        V = np.zeros((M_t, r))
        W = np.zeros((P_t, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[_t0(i, m, j), col] = 1.0
                    V[_t1(j, p, k), col] = 1.0
                    W[_t2(k, n, i), col] = 1.0
                    col += 1
        return U, V, W

    U_sb, V_sb, W_sb = schoolbook()
    best_rank = n * m * p
    best_U, best_V, best_W = U_sb, V_sb, W_sb

    def als_for_rank(r, n_trials=10, max_iter=500):
        best_res = float('inf')
        best = None
        for trial in range(n_trials):
            if time.time() > deadline - 18.0:
                break
            seed = 42 + trial * 13
            rng = np.random.RandomState(seed)
            U = rng.randn(N_t, r) * 0.3
            V = rng.randn(M_t, r) * 0.3
            W = rng.randn(P_t, r) * 0.3
            for it in range(max_iter):
                if time.time() > deadline - 12.0:
                    break
                # Solve U
                phi = W[:, None, :] * V[None, :, :]  # (P_t, M_t, r)
                A_U = phi.reshape(P_t * M_t, r)
                for A in range(N_t):
                    U[:, A] = np.linalg.lstsq(A_U, target[:, :, A].reshape(-1), rcond=None)[0]
                # Solve V
                psi = W[:, None, :] * U[None, :, :]  # (P_t, N_t, r)
                A_V = psi.reshape(P_t * N_t, r)
                for B in range(M_t):
                    V[:, B] = np.linalg.lstsq(A_V, target[:, B, :].reshape(-1), rcond=None)[0]
                # Solve W
                omega = V[:, None, :] * U[None, :, :]  # (M_t, N_t, r)
                A_W = omega.reshape(M_t * N_t, r)
                for C in range(P_t):
                    W[:, C] = np.linalg.lstsq(A_W, target[C, :, :].reshape(-1), rcond=None)[0]
                cur_res = residual(U, V, W)
                if cur_res < best_res:
                    best_res = cur_res
                    best = (U.copy(), V.copy(), W.copy())
                if cur_res < 1e-7:
                    break
            if best_res < 1e-7:
                break
        if best_res < 1e-7:
            return best
        return None

    lower = max(n * m, m * p, p * n)
    for r in range(lower, best_rank + 1):
        if time.time() > deadline - 30.0:
            break
        if r >= best_rank:
            break
        res = als_for_rank(r)
        if res is not None:
            best_U, best_V, best_W = res
            best_rank = r
            break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
