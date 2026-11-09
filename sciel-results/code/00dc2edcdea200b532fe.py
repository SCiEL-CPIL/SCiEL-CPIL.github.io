"""Tensor decomposition for 2x4 * 4x7 matrix multiplication (rank-minimal search)."""
import numpy as np
import time

DIMENSIONS = [2, 4, 7]
DEADLINE = 140.0  # safety margin under 150s wall-clock


def _verify(U, V, W):
    """Verify: for all i,j,k: sum_t U[i*m+j,t]*V[j*p+k,t]*W[k*n+i,t] == 1."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    r = U.shape[1]
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    tmp = np.einsum('ijr,jkr->ijk', U3, V3)   # (n,m,p)
    T = np.einsum('ijk,kit->ijk', tmp, W3)     # (n,m,p)
    return bool(np.allclose(T, np.ones((n, m, p)), atol=1e-6))


def _schoolbook():
    n, m, p = DIMENSIONS
    r = n * m * p
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def _try_rank(r, seed, max_iters=40):
    """Random integer U,V; solve linear system for W; verify."""
    n, m, p = DIMENSIONS
    rng = np.random.RandomState(seed)
    num_eq = n * m * p
    num_unk = p * n * r
    for it in range(max_iters):
        U = rng.randint(-1, 2, size=(n * m, r))
        V = rng.randint(-1, 2, size=(m * p, r))
        M = np.zeros((num_eq, num_unk))
        b = np.ones(num_eq)
        eq = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for t in range(r):
                        coeff = U[i * m + j, t] * V[j * p + k, t]
                        if coeff != 0:
                            M[eq, (k * n + i) * r + t] = coeff
                    eq += 1
        try:
            W_flat, *_ = np.linalg.lstsq(M, b, rcond=None)
        except Exception:
            continue
        W = W_flat.reshape(p * n, r)
        if _verify(U, V, W):
            return U, V, W
    return None


def construct():
    start = time.time()
    best = _schoolbook()
    best_rank = len(best["U"][0])  # 56

    # Search for lower rank via random structured U,V + linear solve for W
    for r in range(30, best_rank):
        if time.time() - start > DEADLINE - 15:
            break
        seed = 0
        while time.time() - start < DEADLINE - 15:
            res = _try_rank(r, seed=seed, max_iters=20)
            if res is not None:
                U, V, W = res
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            seed += 1
    return best


if __name__ == "__main__":
    out = construct()
    print("rank:", len(out["U"][0]))
    print("verify:", _verify(out["U"], out["V"], out["W"]))
