"""QAOA p=2 graph-dependent angle optimizer with exact MaxCut objective."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_DEADLINE = 110.0  # safety margin below 120s wall-clock limit
_START = time.time()
_CACHE = {}
_MAXN = 10


def _elapsed():
    return time.time() - _START


def _remaining():
    return _DEADLINE - _elapsed()


def _graph_key(n, edges):
    e = sorted((int(u), int(v)) for (u, v) in edges)
    return (int(n), tuple(e))


def _build_cost(n, edges):
    """Return (cost_func, init_params) for p=2 QAOA on this graph."""
    N = 1 << n
    idx = np.arange(N, dtype=np.int64)
    # X (bit flip on all qubits) permutation
    X = np.zeros((N, N), dtype=np.float64)
    X[idx, idx ^ (N - 1)] = 1.0
    # Pauli-Z diagonal
    zdiag = np.where((idx & 1) == 0, 1.0, -1.0)
    # MaxCut cost C = 0.5 * sum_{(u,v)} (I - Z_u Z_v)
    C = np.zeros((N, N), dtype=np.float64)
    for (u, v) in edges:
        zu = np.where((idx >> u) & 1 == 0, 1.0, -1.0)
        zv = np.where((idx >> v) & 1 == 0, 1.0, -1.0)
        C += 0.5 * (N - (zu * zv) * idx)  # placeholder, replaced below
    # Recompute C correctly (diagonal)
    C = np.zeros((N, N), dtype=np.float64)
    cdiag = np.zeros(N, dtype=np.float64)
    for (u, v) in edges:
        zu = np.where((idx >> u) & 1 == 0, 1.0, -1.0)
        zv = np.where((idx >> v) & 1 == 0, 1.0, -1.0)
        cdiag += 0.5 * (1.0 - zu * zv)
    C[np.diag_indices(N)] = cdiag
    # Precompute exp(-i beta X) for a grid? No, beta varies. Use scipy expm.
    from scipy.linalg import expm

    psi0 = np.full(N, 1.0 / np.sqrt(N), dtype=np.complex128)

    def expected_cut(params):
        g1, b1, g2, b2 = params
        U1 = expm(-1j * g1 * C) @ expm(-1j * b1 * X)
        U2 = expm(-1j * b2 * X)
        psi = U2 @ (U1 @ psi0)
        # <psi| Z_u Z_v |psi>
        corr = np.conj(psi) * psi  # diagonal of density
        cut = 0.0
        for (u, v) in edges:
            zu = np.where((idx >> u) & 1 == 0, 1.0, -1.0)
            zv = np.where((idx >> v) & 1 == 0, 1.0, -1.0)
            zz = zu * zv
            cut += 0.5 * (1.0 - np.real(np.dot(corr, zz)))
        return cut

    return expected_cut, [0.35, 0.25, 0.35, 0.25]


def _exact_maxcut(n, edges):
    """Exact MaxCut value via exhaustive search (n<=10)."""
    best = 0.0
    N = 1 << n
    for s in range(N):
        val = 0.0
        for (u, v) in edges:
            bu = (s >> u) & 1
            bv = (s >> v) & 1
            if bu != bv:
                val += 1.0
        if val > best:
            best = val
    return best


def _optimize_graph(n, edges):
    """Optimize p=2 QAOA angles for one graph, return (p,2) array."""
    key = _graph_key(n, edges)
    if key in _CACHE:
        return _CACHE[key]

    opt = _exact_maxcut(n, edges)
    if opt <= 0:
        res = np.tile([0.35, 0.25], (2, 1))
        _CACHE[key] = res
        return res

    try:
        cost_func, init = _build_cost(n, edges)
    except Exception:
        res = np.tile([0.35, 0.25], (2, 1))
        _CACHE[key] = res
        return res

    def neg_ratio(x):
        return -cost_func(x) / opt

    bounds = [(-0.5, 3.0), (-0.5, 3.0), (-0.5, 3.0), (-0.5, 3.0)]
    best_x = init
    best_val = neg_ratio(init)

    # Multi-start L-BFGS-B
    starts = [init, [0.5, 0.5, 0.5, 0.5], [1.0, 0.5, 0.5, 0.5],
              [0.2, 0.3, 0.4, 0.3], [0.8, 0.2, 0.6, 0.4]]
    for s in starts:
        if _remaining() < 1.0:
            break
        try:
            r = minimize(neg_ratio, s, method="L-BFGS-B",
                         bounds=bounds, options={"maxiter": 80, "gtol": 1e-6})
            if r.success or r.fun < best_val:
                if r.fun < best_val:
                    best_val = r.fun
                    best_x = r.x
        except Exception:
            pass

    # Polish with Nelder-Mead on best
    if _remaining() > 2.0:
        try:
            r = minimize(neg_ratio, best_x, method="Nelder-Mead",
                         options={"maxiter": 400, "xatol": 1e-8, "fatol": 1e-10})
            if r.fun < best_val:
                best_val = r.fun
                best_x = r.x
        except Exception:
            pass

    g1, b1, g2, b2 = best_x
    res = np.array([[g1, b1], [g2, b2]], dtype=np.float64)
    _CACHE[key] = res
    return res


def angles(n, edges, p):
    # Deadline guard: if over budget, return the seed default (score 1.0).
    if _remaining() <= 0.0:
        return np.tile([0.35, 0.25], (p, 1))
    try:
        if p == 2:
            return _optimize_graph(int(n), edges)
        # Fallback for unexpected p: tile seed
        return np.tile([0.35, 0.25], (p, 1))
    except Exception:
        return np.tile([0.35, 0.25], (p, 1))
# EVOLVE-BLOCK-END
