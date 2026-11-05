import numpy as np
import time

_START = time.time()
# safety margin: the evaluator kills at 120 s; keep ~15 s headroom
_DEADLINE = _START + 105.0

_N = 24
_NGRID = 2048
# Precompute the evaluation basis: Z[i,k] = exp(i*2pi*i/NGRID)^k
_ANG = 2.0 * np.pi * np.arange(_NGRID) / _NGRID
_K = np.arange(_N)
Z = np.exp(1j * np.outer(_ANG, _K))  # shape (NGRID, N)


def _supnorm(c):
    # c: (N,) real array of +1/-1; returns max over |z|=1 of |P(z)|
    return float(np.max(np.abs(Z @ c)))


def _golay_seed():
    # Golay-Rudin-Shapiro recursion, truncated to N coefficients
    p, q = [1], [1]
    while len(p) < _N:
        p, q = p + q, p + [-c for c in q]
    return np.array(p[:_N], dtype=np.float64)


def _local_search(c, rng):
    # Greedy single-bit-flip hill descent. Returns (coeffs, best_sup).
    c = c.copy()
    best = _supnorm(c)
    n = len(c)
    while True:
        if time.time() >= _DEADLINE:
            break
        # flip effect for each position: new = Z@c - 2*Z@e_k
        base = Z @ c
        flip = 2.0 * Z  # columns of 2*Z
        cand = base[None, :] - flip  # shape (n, NGRID)
        m = np.max(np.abs(cand), axis=1)  # (n,)
        j = int(np.argmin(m))
        if m[j] >= best:
            break
        c[j] = -c[j]
        best = float(m[j])
    return c, best


def construct():
    rng = np.random.default_rng(12345)
    c0 = _golay_seed()
    best_c, best_val = _local_search(c0, rng)

    it = 0
    while time.time() < _DEADLINE:
        it += 1
        # random restart: mutate the best found so far (or a fresh Golay seed)
        if it % 5 == 0:
            start = _golay_seed().copy()
        else:
            start = best_c.copy()
        # random jumps to escape local minima
        for _ in range(4):
            k = int(rng.integers(0, _N))
            start[k] = -start[k]
        c, val = _local_search(start, rng)
        if val < best_val - 1e-12:
            best_val = val
            best_c = c.copy()
        if time.time() >= _DEADLINE:
            break

    return [int(x) for x in best_c]
