import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [3, 3, 4]
T0 = time.time()
DEADLINE = T0 + 110  # safety margin under the 120s/150s evaluator limit

def _build_T():
    n, m, p = DIMENSIONS
    T = np.zeros((n*m, m*p, p*n))
    for a in range(n):
        for b in range(m):
            for c in range(p):
                i = a*m + b
                j = b*p + c
                k = c*n + a
                T[i, j, k] = 1.0
    return T

def _schoolbook():
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

def _als(T, R, iters=100, seed=0, init=None):
    """ALS tensor decomposition. init: optional (U,V,W) numpy arrays to warm-start."""
    rng = np.random.default_rng(seed)
    n, m, p = T.shape
    if init is not None:
        U, V, W = [a.copy() for a in init]
    else:
        U = rng.normal(size=(n, R))
        V = rng.normal(size=(m, R))
        W = rng.normal(size=(p, R))

    for it in range(iters):
        for r in range(R):
            v_r = V[:, r]; w_r = W[:, r]
            prod = np.outer(v_r, w_r)
            numer = np.tensordot(T, prod, axes=([1, 2], [0, 1]))
            denom = np.sum(prod**2)
            U[:, r] = numer / denom if denom > 1e-14 else 0.0
        for r in range(R):
            u_r = U[:, r]; w_r = W[:, r]
            prod = np.outer(u_r, w_r)
            numer = np.tensordot(T, prod, axes=([0, 2], [0, 1]))
            denom = np.sum(prod**2)
            V[:, r] = numer / denom if denom > 1e-14 else 0.0
        for r in range(R):
            u_r = U[:, r]; v_r = V[:, r]
            prod = np.outer(u_r, v_r)
            numer = np.tensordot(T, prod, axes=([0, 1], [0, 1]))
            denom = np.sum(prod**2)
            W[:, r] = numer / denom if denom > 1e-14 else 0.0

    recon = np.einsum('ir,jr,kr->ijk', U, V, W)
    res = np.linalg.norm(T - recon)
    return U, V, W, res

def _als_polish(T, U, V, W, iters=200):
    """High-precision polish: more ALS iterations with tighter zero threshold."""
    n, m, p = T.shape
    R = U.shape[1]
    U, V, W = U.copy(), V.copy(), W.copy()
    for it in range(iters):
        for r in range(R):
            prod = np.outer(V[:, r], W[:, r])
            numer = np.tensordot(T, prod, axes=([1, 2], [0, 1]))
            denom = np.sum(prod**2)
            U[:, r] = numer / denom if denom > 1e-16 else 0.0
        for r in range(R):
            prod = np.outer(U[:, r], W[:, r])
            numer = np.tensordot(T, prod, axes=([0, 2], [0, 1]))
            denom = np.sum(prod**2)
            V[:, r] = numer / denom if denom > 1e-16 else 0.0
        for r in range(R):
            prod = np.outer(U[:, r], V[:, r])
            numer = np.tensordot(T, prod, axes=([0, 1], [0, 1]))
            denom = np.sum(prod**2)
            W[:, r] = numer / denom if denom > 1e-16 else 0.0
    recon = np.einsum('ir,jr,kr->ijk', U, V, W)
    res = np.linalg.norm(T - recon)
    return U, V, W, res

def _exact_verify(U, V, W, T_int):
    n, m, p = T_int.shape
    R = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for r in range(R):
                    s += Fraction(U[i][r]) * Fraction(V[j][r]) * Fraction(W[k][r])
                if s != T_int[i, j, k]:
                    return False
    return True

def _to_frac_list(arr, denom):
    out = []
    for row in arr:
        r = []
        for x in row:
            r.append(round(x * denom) / denom)
        out.append(r)
    return out

def _try_exactify(U, V, W, T_int):
    """Try rational rounding with several denominators; return (Uf,Vf,Wf) or None."""
    for denom in [1, 2, 4, 8, 16, 32]:
        Uf = _to_frac_list(U, denom)
        Vf = _to_frac_list(V, denom)
        Wf = _to_frac_list(W, denom)
        if _exact_verify(Uf, Vf, Wf, T_int):
            return (Uf, Vf, Wf)
    return None

def _fmm3_init(T, R, seed=0):
    """Structured FMM-3-inspired warm start: seed U,V,W with a near-sparse
    pattern derived from the 3x3 Winograd structure, then zero-pad to R."""
    rng = np.random.default_rng(seed)
    n, m, p = T.shape
    U = rng.normal(size=(n, R)) * 0.1
    V = rng.normal(size=(m, R)) * 0.1
    W = rng.normal(size=(p, R)) * 0.1
    # Inject a schoolbook-like sparse core to help convergence
    core = min(R, n*m*p)
    for c in range(core):
        i = (c // (m*p)) % n
        j = (c // p) % m
        k = c % p
        U[i, c] += 1.0
        V[j, c] += 1.0
        W[k, c] += 1.0
    return U, V, W

def construct():
    T = _build_T()
    T_int = T.astype(int)
    n, m, p = DIMENSIONS
    fallback = _schoolbook()

    # Search from a low bound up to (but not including) schoolbook rank.
    # Known: 3x3x3 rank <= 23 (FMM-3). For 3x3x4, schoolbook is 36.
    # We attempt ranks 19..35 with multiple seeds, FMM warm-starts, and polish.
    best = None
    for R in range(19, 36):
        if time.time() > DEADLINE:
            break
        found = None
        # Alternate between random init and FMM-3-inspired init
        for seed in range(0, 40):
            if time.time() > DEADLINE:
                break
            init = _fmm3_init(T, R, seed=seed) if (seed % 2 == 0) else None
            U, V, W, res = _als(T, R, iters=80, seed=seed, init=init)
            if res < 1e-5:
                # Polish at high precision
                U, V, W, res = _als_polish(T, U, V, W, iters=300)
                if res < 1e-7:
                    found = _try_exactify(U, V, W, T_int)
                    if found:
                        break
        if found:
            best = found
            break

    if best is not None:
        Uf, Vf, Wf = best
        return {"U": Uf, "V": Vf, "W": Wf}

    return fallback
