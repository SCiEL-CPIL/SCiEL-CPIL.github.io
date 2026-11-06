"""Two-layer QAOA angle design: per-graph numerical optimization of (gamma, beta)."""
# EVOLVE-BLOCK-START
import time
import math
import itertools
import numpy as np
from scipy.optimize import minimize

_DEADLINE = 120.0
_START = time.time()

# Precompute the 2^p-dim phase basis for the MaxCut cost operator.
# C = sum_{(i,j) in E} (I - Z_i Z_j)/2  =>  exp(-i gamma C) applies phase exp(-i gamma * c)
# where c = (# edges whose endpoints differ) / 2.
_PHASE_CACHE = {}

def _phase_vector(n, edges):
    key = (n, tuple(sorted(edges)))
    if key in _PHASE_CACHE:
        return _PHASE_CACHE[key]
    m = 1 << n
    c = np.empty(m, dtype=np.float64)
    for s in range(m):
        cnt = 0
        for (u, v) in edges:
            if ((s >> u) & 1) != ((s >> v) & 1):
                cnt += 1
        c[s] = 0.5 * cnt
    _PHASE_CACHE[key] = c
    return c

# Precompute the Kronecker product of n single-qubit X-rotations for each n.
_ROT_CACHE = {}

def _rot_matrix(n, beta):
    key = (n, round(float(beta), 12))
    if key in _ROT_CACHE:
        return _ROT_CACHE[key]
    cb = math.cos(beta)
    sb = math.sin(beta)
    R = np.array([[cb, -1j * sb], [-1j * sb, cb]], dtype=np.complex128)
    M = R
    for _ in range(n - 1):
        M = np.kron(M, R)
    if len(_ROT_CACHE) < 200000:
        _ROT_CACHE[key] = M
    return M

_INIT_STATE_CACHE = {}

def _init_state(n):
    if n not in _INIT_STATE_CACHE:
        _INIT_STATE_CACHE[n] = np.full(1 << n, 1.0 / math.sqrt(1 << n), dtype=np.complex128)
    return _INIT_STATE_CACHE[n]

def _expected_cut(n, edges, gamma, beta):
    """Mean cut value = E[ # edges crossing the cut ] for QAOA angles (gamma, beta), p=2."""
    c = _phase_vector(n, edges)
    psi = _init_state(n)
    # Layer 1: cost
    psi = psi * np.exp(-1j * gamma * c)
    psi = psi * _rot_matrix(n, beta)
    # Layer 2: cost
    psi = psi * np.exp(-1j * gamma * c)
    psi = psi * _rot_matrix(n, beta)
    prob = np.abs(psi) ** 2
    return float(np.dot(prob, c))

def _exact_maxcut(n, edges):
    m = 1 << n
    best = 0
    for s in range(m):
        cnt = 0
        for (u, v) in edges:
            if ((s >> u) & 1) != ((s >> v) & 1):
                cnt += 1
        if cnt > best:
            best = cnt
    return best

def _starts(n, edges):
    """Return a list of (gamma, beta) starting points."""
    starts = [(0.35, 0.25)]  # known-good seed point
    # Graph-structure-driven starts: scale by edge density / average degree.
    m_edges = len(edges)
    density = (2.0 * m_edges) / (n * (n - 1)) if n > 1 else 0.0
    avg_deg = (2.0 * m_edges) / n if n > 0 else 0.0
    # Heuristic: gamma ~ pi * (1 - density), beta ~ pi * density / 2
    starts.append((math.pi * max(0.0, 1.0 - density), math.pi * density / 2.0))
    starts.append((math.pi * 0.25, math.pi * 0.125))
    starts.append((math.pi * 0.5, math.pi * 0.25))
    # Coarse grid
    for g in (0.2, 0.4, 0.6, 0.8, 1.0):
        for b in (0.1, 0.2, 0.3, 0.4, 0.5):
            starts.append((g, b))
    # Deduplicate
    seen = set()
    out = []
    for s in starts:
        k = (round(s[0], 5), round(s[1], 5))
        if k not in seen:
            seen.add(k)
            out.append(s)
    return out

def _optimize_one(n, edges):
    """Optimize (gamma, beta) for a single graph; return the best (gamma, beta, value)."""
    starts = _starts(n, edges)
    best_val = -1.0
    best_pt = (0.35, 0.25)

    def neg(x):
        return -_expected_cut(n, edges, x[0], x[1])

    budget = max(0.0, _DEADLINE - (time.time() - _START) - 5.0)
    per_start = budget / max(1, len(starts))

    for s in starts:
        if time.time() - _START > _DEADLINE - 5.0:
            break
        v = _expected_cut(n, edges, s[0], s[1])
        if v > best_val:
            best_val = v
            best_pt = (s[0], s[1])
        # Local refinement from this start
        try:
            res = minimize(neg, np.array(s, dtype=np.float64), method='Nelder-Mead',
                           options={'maxiter': int(max(50, per_start * 200)),
                                    'xatol': 1e-5, 'fatol': 1e-7})
            if res.x is not None:
                v2 = _expected_cut(n, edges, float(res.x[0]), float(res.x[1]))
                if v2 > best_val:
                    best_val = v2
                    best_pt = (float(res.x[0]), float(res.x[1]))
        except Exception:
            pass

    # Fine grid refinement around the best point found.
    if time.time() - _START < _DEADLINE - 5.0:
        g0, b0 = best_pt
        for dg in (-0.1, -0.05, 0.0, 0.05, 0.1):
            for db in (-0.05, -0.025, 0.0, 0.025, 0.05):
                if time.time() - _START > _DEADLINE - 5.0:
                    break
                v = _expected_cut(n, edges, g0 + dg, b0 + db)
                if v > best_val:
                    best_val = v
                    best_pt = (g0 + dg, b0 + db)

    return best_pt, best_val

_RESULT_CACHE = {}

def angles(n, edges, p):
    """Return a (p, 2) array of (gamma, beta) in radians for the given graph."""
    edges_t = tuple(sorted((int(u), int(v)) for (u, v) in edges))
    key = (int(n), edges_t, int(p))
    if key in _RESULT_CACHE:
        return _RESULT_CACHE[key]

    # Safety: if we are already close to the deadline, return the known-good point.
    if time.time() - _START > _DEADLINE - 10.0:
        res = np.tile([0.35, 0.25], (int(p), 1), dtype=np.float64)
        _RESULT_CACHE[key] = res
        return res

    (g, b), _ = _optimize_one(n, edges_t)
    res = np.tile([g, b], (int(p), 1), dtype=np.float64)
    _RESULT_CACHE[key] = res
    return res
# EVOLVE-BLOCK-END
