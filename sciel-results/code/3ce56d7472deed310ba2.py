import numpy as np

DIMENSIONS = [2, 2, 15]

def _build_tensor():
    n, m, p = DIMENSIONS
    T = np.zeros((n*m, m*p, n*p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    return T

def _als_decompose(T, rank, iters=60, seed=0, tol=1e-12):
    rng = np.random.default_rng(seed)
    n1, n2, n3 = T.shape
    U = rng.standard_normal((n1, rank))
    V = rng.standard_normal((n2, rank))
    W = rng.standard_normal((n3, rank))
    best = None
    for it in range(iters):
        # solve U
        VtW = V.T @ W
        Tmat = np.einsum('ijk,kr->ijr', T, V)  # for each r: sum_k T[i,j,k] V[k,r]
        # U[i,r] = (Tmat[i,r]) / VtW[r,r]
        for r in range(rank):
            U[:, r] = Tmat[:, r] / (VtW[r, r] + 1e-15)
        # solve V
        UtW = U.T @ W
        Tmat = np.einsum('ijk,ir->jkr', T, U)
        for r in range(rank):
            V[:, r] = Tmat[:, r] / (UtW[r, r] + 1e-15)
        # solve W
        UtV = U.T @ V
        Tmat = np.einsum('ijk,jr->ikr', T, V)
        for r in range(rank):
            W[:, r] = Tmat[:, r] / (UtV[r, r] + 1e-15)
        # residual
        recon = np.einsum('ir,jr,kr->ijk', U, V, W)
        res = np.linalg.norm(recon - T)
        if best is None or res < best[0]:
            best = (res, U.copy(), V.copy(), W.copy())
    return best

def _to_fraction_list(M, max_den=1000000):
    from fractions import Fraction
    out = []
    for row in M:
        r = []
        for v in row:
            # try to snap to a simple fraction
            fr = Fraction(v).limit_denominator(max_den)
            r.append(fr)
        out.append(r)
    return out

def _verify(U, V, W, T, tol=1e-9):
    recon = np.einsum('ir,jr,kr->ijk', U, V, W)
    return np.linalg.norm(recon - T) < tol

def construct():
    import time
    t0 = time.time()
    deadline = 110.0
    T = _build_tensor()
    n1, n2, n3 = T.shape
    best = None
    # try ranks from a lower bound up
    for rank in range(40, 61):
        if time.time() > deadline - 20:
            break
        for seed in range(30):
            if time.time() > deadline - 20:
                break
            res, U, V, W = _als_decompose(T, rank, iters=40, seed=seed)
            if best is None or res < best[0]:
                best = (res, rank, U, V, W)
            if res < 1e-8:
                break
        if best is not None and best[0] < 1e-8:
            break
    if best is None or best[0] > 1e-6:
        # fallback schoolbook
        n, m, p = DIMENSIONS
        rank = n*m*p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    res, rank, U, V, W = best
    # snap to fractions
    Uf = _to_fraction_list(U)
    Vf = _to_fraction_list(V)
    Wf = _to_fraction_list(W)
    return {"U": Uf, "V": Vf, "W": Wf}
