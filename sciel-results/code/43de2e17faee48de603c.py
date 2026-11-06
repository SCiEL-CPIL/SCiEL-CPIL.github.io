"""QAOA p=2 angle design: multi-start global optimization with graph-aware scaling."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_T0 = time.time()
_DEADLINE = _T0 + 135.0  # 135s out of 150s limit, leaving 15s safety margin


def _remaining():
    return _DEADLINE - time.time()


def _build_graph(n, edges):
    adj = np.zeros((n, n), dtype=float)
    for u, v in edges:
        if u == v:
            continue
        adj[u, v] = 1.0
        adj[v, u] = 1.0
    return adj


def _expected_cut(n, edges, g1, b1, g2, b2, cache):
    """Vectorized expected MaxCut value for p=2 QAOA."""
    m = len(edges)
    if m == 0:
        return 0.0

    key = (n, tuple(edges))
    if key not in cache:
        c = 1 << n
        u = np.empty(m, dtype=np.int64)
        v = np.empty(m, dtype=np.int64)
        for k, (a, b) in enumerate(edges):
            u[k] = a
            v[k] = b
        bits = np.arange(c, dtype=np.int64)
        zu = np.where((bits >> u) & 1, -1.0, 1.0)
        zv = np.where((bits >> v) & 1, -1.0, 1.0)
        zz = zu * zv
        cache[key] = (c, zz)
    c, zz = cache[key]

    cos1, sin1 = np.cos(b1), np.sin(b1)
    cos2, sin2 = np.cos(b2), np.sin(b2)

    phase1 = np.exp(0.5j * g1 * zz)
    p = (cos1 + 1j * sin1 * zz) * phase1
    phase2 = np.exp(0.5j * g2 * zz)
    p = p * phase2

    prob = np.abs(p) ** 2
    corr = np.sum(prob * zz, axis=1)
    return float(np.sum(0.5 * (1.0 - corr)))


def _exhaustive_maxcut(n, edges):
    best = 0
    for mask in range(1 << n):
        cut = 0
        for u, v in edges:
            if ((mask >> u) ^ (mask >> v)) & 1:
                cut += 1
        if cut > best:
            best = cut
    return best


def _objective(params, n, edges, cache):
    return -_expected_cut(n, edges, params[0], params[1], params[2], params[3], cache)


def angles(n, edges, p):
    adj = _build_graph(n, edges)
    m = len(edges)
    cache = {}
    
    # Compute graph properties for scaling
    deg = adj.sum(axis=1)
    L = np.diag(deg) - adj
    try:
        w = np.linalg.eigvalsh(L)
        lmax = float(w[-1]) if len(w) else 0.0
    except Exception:
        lmax = 0.0

    density = 2.0 * m / (n * (n - 1)) if n > 1 else 0.0

    # Graph-aware parameter ranges
    # For sparse graphs, smaller angles; for dense, larger
    if lmax > 1e-8:
        gamma_max = min(1.5, 0.5 * lmax)
    else:
        gamma_max = 0.5
    beta_max = 1.0

    # Multi-start strategy with diverse initial points
    # Generate initial points using different strategies
    
    initial_points = []
    
    # Strategy 1: Uniform grid sampling (coarse)
    n_grid = 6
    gamma_vals = np.linspace(0.05, gamma_max, n_grid)
    beta_vals = np.linspace(0.05, beta_max, n_grid)
    
    for g1 in gamma_vals:
        for b1 in beta_vals:
            for g2 in gamma_vals:
                for b2 in beta_vals:
                    initial_points.append([g1, b1, g2, b2])
    
    # Strategy 2: Random sampling
    rng = np.random.default_rng(42)
    for _ in range(30):
        g1 = rng.uniform(0.05, gamma_max)
        b1 = rng.uniform(0.05, beta_max)
        g2 = rng.uniform(0.05, gamma_max)
        b2 = rng.uniform(0.05, beta_max)
        initial_points.append([g1, b1, g2, b2])
    
    # Strategy 3: Heuristic starting points based on graph properties
    # For small graphs, try symmetric and asymmetric splits
    for ratio in [0.3, 0.5, 0.7]:
        g1 = gamma_max * ratio
        g2 = gamma_max * (1 - ratio)
        b1 = beta_max * 0.5
        b2 = beta_max * 0.5
        initial_points.append([g1, b1, g2, b2])
    
    # Evaluate all initial points and keep top candidates
    scored_points = []
    for point in initial_points:
        if _remaining() < 15.0:
            break
        try:
            val = -_objective(point, n, edges, cache)
            scored_points.append((val, point))
        except Exception:
            continue
    
    # Sort by value and keep top K for refinement
    scored_points.sort(key=lambda x: x[0], reverse=True)
    top_k = scored_points[:min(8, len(scored_points))]
    
    best_val = top_k[0][0] if top_k else -np.inf
    best_x = top_k[0][1] if top_k else np.array([0.1, 0.1, 0.1, 0.1])
    
    # Refine top candidates with Nelder-Mead
    for val, point in top_k:
        if _remaining() < 20.0:
            break
        try:
            res = minimize(
                _objective, point, args=(n, edges, cache),
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-12},
            )
            if res.success and -res.fun > best_val:
                best_val = -res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    # Final high-precision refinement
    if _remaining() > 15.0:
        try:
            res = minimize(
                _objective, best_x, args=(n, edges, cache),
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-11, 'fatol': 1e-14},
            )
            if res.success and -res.fun > best_val:
                best_val = -res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    out = np.zeros((p, 2), dtype=float)
    out[0, 0] = best_x[0]
    out[0, 1] = best_x[1]
    if p > 1:
        out[1, 0] = best_x[2]
        out[1, 1] = best_x[3]
    return out
# EVOLVE-BLOCK-END
