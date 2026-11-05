"""Exact rank-minimal tensor decomposition for 3x4 * 4x7 matrix multiplication.

For a fixed rank r, fit the multiplication tensor by alternating least-squares
on a large overdetermined set of random samples, then VERIFY exactness by
checking the decomposed product equals A@B on a fixed set of generating pairs.
Sweep r upward; return the smallest r that verifies exactly.
"""
import time
import random
import numpy as np

DIMENSIONS = [3, 4, 7]
n, m, p = DIMENSIONS
D = n * m * p          # 84 tensor coefficients
DEADLINE = 115.0
START = time.time()


def _time_left():
    return DEADLINE - (time.time() - START)


def _time_up():
    return _time_left() <= 0.0


def _build_samples():
    rng = random.Random(12345)
    mat = np.empty((400, D), dtype=float)
    for t in range(400):
        A = np.array([[rng.randint(-3, 3) for _ in range(m)] for _ in range(n)], dtype=float)
        B = np.array([[rng.randint(-3, 3) for _ in range(p)] for _ in range(m)], dtype=float)
        v = np.empty(D, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    v[k * n + i] = A[i, j] * B[j, k]
        mat[t] = v
    return mat


def _verify_exact(U, V, W, gen_pairs, tol):
    r = U.shape[1]
    for (A, B) in gen_pairs:
        C = A @ B
        Cdec = np.zeros((n, p), dtype=float)
        for c in range(r):
            Uc = U[:, c].reshape(n, m)
            Vc = V[:, c].reshape(m, p)
            Wc = W[:, c].reshape(p, n)
            Cdec += (Uc @ Vc) * Wc.T
        if not np.allclose(Cdec, C, atol=tol, rtol=1e-9):
            return False
    return True


def _fit_rank(r, samples, gen_pairs, seed, max_iter=300):
    rng = np.random.RandomState(seed)
    S = samples.shape[0]
    U = rng.randn(n * m, r)
    V = rng.randn(m * p, r)
    W = rng.randn(p * n, r)

    # Precompute index maps for vectorized A_d construction.
    # comp index rr = s*n*p + k*n + i
    ks = np.repeat(np.arange(p), n)          # (p*n,)
    is_ = np.tile(np.arange(n), p)           # (p*n,)
    comp = np.arange(n * p)                  # (n*p,)

    for it in range(max_iter):
        if _time_up():
            break

        # ---- Update U (fix V,W): A_d[s*n*p + k*n + i, c*m + j] = W[k*n+i,c]*V[j*p+k,c] ----
        # W[k*n+i, c] shape (p*n, r); V[j*p+k, c]: for each k, V[::p, c] is (m, r)
        # Build A_d of shape (S, p*n, r*m)
        # term = W[k*n+i, c] * V[j*p+k, c]  -> for fixed k: W_k (n, r) outer V_k (m, r) -> (n, m, r)
        # We'll build per-k then stack.
        A_d = np.empty((S, n * p, r * m), dtype=float)
        b = np.empty((S, n * p), dtype=float)
        for s in range(S):
            b[s] = samples[s].reshape(p, n).T.reshape(-1)  # order: k*n+i -> reshape (p,n) then .T gives (n,p)? 
            # careful: we need index (k*n+i) -> element [k, i]. samples[s] is indexed by k*n+i.
            # So b[s, k*n+i] = samples[s, k*n+i]. Just use samples[s] directly in comp order.
        # Redo b in comp order (k*n+i):
        b = samples.reshape(S, n * p).copy()

        for k in range(p):
            Wk = W[k * n:(k + 1) * n, :]        # (n, r)
            Vk = V[k::p, :]                      # (m, r)  rows j*p+k for j=0..m-1
            # A_d[s, k*n+i, c*m+j] = Wk[i,c] * Vk[j,c]
            # shape (n, m, r) -> rearrange to (n, r*m) with axis (i, c*m+j)
            term = Wk[:, None, :] * Vk[None, :, :]   # (n, m, r)
            term = term.transpose(0, 2, 1).reshape(n, r * m)  # (n, r*m)
            for s in range(S):
                A_d[s, k * n:(k + 1) * n, :] = term
        A_d = A_d.reshape(S * n * p, r * m)
        Uf, *_ = np.linalg.lstsq(A_d, b.reshape(-1), rcond=None)
        U = Uf.reshape(n * m, r)

        # ---- Update V (fix U,W): A_d[s, k*n+i, c*m+j] = U[i*m+j,c]*W[k*n+i,c] ----
        A_d = np.empty((S, n * p, r * m), dtype=float)
        for k in range(p):
            Wk = W[k * n:(k + 1) * n, :]        # (n, r)
            Uk = U[:, :]                          # (n*m, r)
            # A_d[s, k*n+i, c*m+j] = U[i*m+j, c] * Wk[i, c]
            # for fixed i: U[i*m:(i+1)*m, c] (m, r) * Wk[i, c] (r,) -> (m, r)
            term = np.empty((n, m, r), dtype=float)
            for i in range(n):
                Ui = U[i * m:(i + 1) * m, :]     # (m, r)
                term[i] = Ui * Wk[i][None, :]     # (m, r)
            term = term.transpose(0, 2, 1).reshape(n, r * m)
            for s in range(S):
                A_d[s, k * n:(k + 1) * n, :] = term
        A_d = A_d.reshape(S * n * p, r * m)
        Vf, *_ = np.linalg.lstsq(A_d, b.reshape(-1), rcond=None)
        V = Vf.reshape(m * p, r)

        # ---- Update W (fix U,V): A_d[s, k*n+i, c] = sum_j U[i*m+j,c]*V[j*p+k,c] = (U_c@V_c)[i,k] ----
        A_d = np.empty((S, n * p, r), dtype=float)
        for k in range(p):
            Vk = V[k::p, :]                      # (m, r)
            # (U_c @ V_c)[i,k] = sum_j U[i*m+j,c]*Vk[j,c]  -> U (n*m, r) @ Vk.T (r, m) -> (n*m, m) -> per i: (n, r)
            prod = np.einsum('imr,jr->ijr', U, Vk)  # (n*m, m, r)? 
            # simpler: for each c: (U[:,c].reshape(n,m) @ Vk[:,c]) -> (n,)
            term = np.empty((n, r), dtype=float)
            for c in range(r):
                Uc = U[:, c].reshape(n, m)
                term[:, c] = Uc @ Vk[:, c]
            for s in range(S):
                A_d[s, k * n:(k + 1) * n, :] = term
        A_d = A_d.reshape(S * n * p, r)
        Wf, *_ = np.linalg.lstsq(A_d, b.reshape(-1), rcond=None)
        W = Wf.reshape(p * n, r)

        if _verify_exact(U, V, W, gen_pairs, tol=1e-8):
            if _verify_exact(U, V, W, gen_pairs, tol=1e-11):
                return (U, V, W)
    return None


def _to_output(U, V, W):
    def clean(arr):
        out = []
        for row in arr:
            r = []
            for x in row:
                hx = round(x * 2) / 2
                if abs(x - hx) < 1e-6:
                    x = hx
                r.append(x)
            out.append(r)
        return out
    return {"U": clean(U), "V": clean(V), "W": clean(W)}


# EVOLVE-BLOCK-START
def construct():
    global START
    START = time.time()

    rng = random.Random(777)
    gen_pairs = []
    for t in range(60):
        A = np.array([[rng.randint(-4, 4) for _ in range(m)] for _ in range(n)], dtype=float)
        B = np.array([[rng.randint(-4, 4) for _ in range(p)] for _ in range(m)], dtype=float)
        gen_pairs.append((A, B))

    samples = _build_samples()

    r_min = 24
    r_max = n * m * p

    best_result = None
    seed = 0
    for r in range(r_min, r_max + 1):
        if _time_up():
            break
        found = None
        while _time_left() > 3.0:
            found = _fit_rank(r, samples, gen_pairs, seed=seed, max_iter=200)
            if found is not None:
                if _verify_exact(found[0], found[1], found[2], gen_pairs, tol=1e-11):
                    best_result = _to_output(*found)
                    break
            seed += 1
        if best_result is not None:
            break
        if _time_left() < 10.0:
            break

    if best_result is None:
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    return best_result
# EVOLVE-BLOCK-END
