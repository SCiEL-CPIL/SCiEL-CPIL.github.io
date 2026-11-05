"""Graph-dependent two-layer QAOA angle design (fresh design)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from scipy.optimize import minimize


# ---------- Exact MaxCut (brute force) ----------
def _exact_maxcut(n, edges):
    if not edges:
        return 0.0
    best = 0.0
    for mask in range(1 << n):
        c = 0
        for u, v in edges:
            if (mask >> u) & 1 != (mask >> v) & 1:
                c += 1
        if c > best:
            best = c
    return float(best)


# ---------- QAOA expected cut ----------
def _expected_cut(n, edges, gammas, betas):
    if not edges:
        return 0.0
    x = np.ones(n)
    for g, b in zip(gammas, betas):
        x = np.cos(g) * x + np.sin(g) * np.array([np.cos(b) * x[u] * x[v] for u, v in edges])
        x = np.cos(b) * x + np.sin(b) * np.array([x[u] + x[v] for u, v in edges])
    return 0.25 * sum((1.0 - x[u] * x[v]) for u, v in edges)


def _ratio(n, edges, exact, gammas, betas):
    if exact <= 0:
        return 1.0
    return _expected_cut(n, edges, gammas, betas) / exact


# ---------- Coarse grid search over the 4 parameters ----------
def _grid_search(n, edges, exact, time_budget):
    start = time.time()
    best_val = -1.0
    best = (0.35, 0.25, 0.35, 0.25)
    # gamma in [0, pi/2], beta in [0, pi]
    g_range = np.linspace(0.0, np.pi / 2, 21)
    b_range = np.linspace(0.0, np.pi, 21)
    for g1, g2, b1, b2 in itertools.product(g_range, g_range, b_range, b_range):
        v = _ratio(n, edges, exact, (g1, g2), (b1, b2))
        if v > best_val:
            best_val = v
            best = (g1, g2, b1, b2)
        if time.time() - start > time_budget:
            break
    return best, best_val


# ---------- Local refinement with scipy ----------
def _refine(n, edges, exact, x0, time_budget):
    start = time.time()
    best_val = _ratio(n, edges, exact, (x0[0], x0[1]), (x0[2], x0[3]))
    best = list(x0)

    def neg_ratio(x):
        if time.time() - start > time_budget:
            return best_val
        return -_ratio(n, edges, exact, (x[0], x[1]), (x[2], x[3]))

    bounds = [(0.0, np.pi / 2), (0.0, np.pi / 2), (0.0, np.pi), (0.0, np.pi)]
    # Try a few restarts
    restarts = [list(x0)]
    # Add a few perturbations
    for _ in range(3):
        r = [min(max(x, 0.0), hi) for x, hi in zip(x0 + 0.1 * np.array([0.5, 0.5, 0.5, 0.5]),
                                                      [np.pi / 2, np.pi / 2, np.pi, np.pi])]
        restarts.append(r)
    for r in restarts:
        if time.time() - start > time_budget:
            break
        try:
            res = minimize(neg_ratio, r, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 200, 'maxfun': 400, 'disp': False})
            if res.fun < -best_val:
                best_val = -res.fun
                best = list(res.x)
        except Exception:
            pass
    return best, best_val


# ---------- Main entry point ----------
def angles(n, edges, p):
    deadline = 120.0 - 10.0  # 10s safety margin
    start = time.time()

    # Compute exact optimum (fast for n<=10)
    exact = _exact_maxcut(n, edges)

    # Fallback defaults
    default = np.tile([0.35, 0.25], (p, 1))
    if exact <= 0:
        return default

    # Coarse grid search
    budget1 = min(30.0, max(5.0, deadline - (time.time() - start)))
    x0, v0 = _grid_search(n, edges, exact, budget1)
    if time.time() - start > deadline:
        return np.array([[x0[0], x0[2]], [x0[1], x0[3]]])

    # Local refinement
    budget2 = min(60.0, max(5.0, deadline - (time.time() - start)))
    best, best_val = _refine(n, edges, exact, x0, budget2)

    return np.array([[best[0], best[2]], [best[1], best[3]]])
# EVOLVE-BLOCK-END
