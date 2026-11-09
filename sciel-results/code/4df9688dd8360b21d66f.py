# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution


def _compute_edge_contribs(n=10, demands=None):
    """
    For each demand, compute which edges are used if routed clockwise (CW)
    vs counter-clockwise (CCW).
    Returns (cw, ccw) arrays of shape (D, n).
    """
    if demands is None:
        demands = [
            (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1),
            (0, 4, 5), (2, 8, 2), (5, 9, 3), (1, 4, 2),
            (6, 0, 4), (3, 8, 1), (7, 2, 3), (4, 9, 2),
            (8, 1, 5), (0, 6, 1), (5, 2, 2),
        ]
    D = len(demands)
    cw = np.zeros((D, n))
    ccw = np.zeros((D, n))
    for i, (s, t, w) in enumerate(demands):
        # CW: edges s, s+1, ..., t-1 (mod n)
        if s <= t:
            edges = range(s, t)
        else:
            edges = list(range(s, n)) + list(range(0, t))
        for e in edges:
            cw[i, e] = w
        # CCW: all other edges
        for e in range(n):
            if cw[i, e] == 0:
                ccw[i, e] = w
    return cw, ccw


def _max_load(routing, cw, ccw):
    """
    routing: array of 0/1, 0=CW, 1=CCW
    Returns the maximum edge load.
    """
    loads = np.zeros(cw.shape[1])
    for i in range(len(routing)):
        if routing[i] >= 0.5:
            loads += ccw[i]
        else:
            loads += cw[i]
    return np.max(loads)


def _smooth_max_load(x, cw, ccw, beta=20.0):
    """
    Smooth surrogate of max load using log-sum-exp.
    x: continuous vector in [0,1]^D
    Returns a scalar.
    """
    D, n = cw.shape
    # Interpolate: for each demand, load on edge e = x_i * ccw[i,e] + (1-x_i) * cw[i,e]
    # Shape: (D, n)
    contrib = x[:, None] * ccw + (1.0 - x[:, None]) * cw
    # Sum over demands: (n,)
    loads = contrib.sum(axis=0)
    # Soft max: (1/beta) * log(sum(exp(beta * loads)))
    m = np.max(loads)
    if m > 1e10:
        return m
    s = np.exp(beta * (loads - m))
    return m + np.log(np.sum(s)) / beta


def ring_routing() -> np.ndarray:
    t_start = time.time()
    deadline = t_start + 590.0  # safety margin

    n = 10
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1),
        (0, 4, 5), (2, 8, 2), (5, 9, 3), (1, 4, 2),
        (6, 0, 4), (3, 8, 1), (7, 2, 3), (4, 9, 2),
        (8, 1, 5), (0, 6, 1), (5, 2, 2),
    ]
    D = len(demands)
    cw, ccw = _compute_edge_contribs(n, demands)

    # Initial best: all CW
    best_routing = np.zeros(D)
    best_load = _max_load(best_routing, cw, ccw)

    # Also try all CCW
    alt = np.ones(D)
    alt_load = _max_load(alt, cw, ccw)
    if alt_load < best_load:
        best_routing = alt
        best_load = alt_load

    # Differential evolution on continuous domain [0,1]^D
    bounds = [(0.0, 1.0)] * D

    def objective(x):
        return _smooth_max_load(x, cw, ccw, beta=25.0)

    # Run DE
    try:
        result = differential_evolution(
            objective,
            bounds=bounds,
            maxiter=300,
            popsize=30,
            tol=1e-8,
            seed=42,
            polish=True,
            callback=None,
            workers=-1,
        )
    except Exception:
        result = None

    if result is not None and result.x is not None:
        # Round to binary
        x_rounded = np.where(result.x >= 0.5, 1.0, 0.0)
        load = _max_load(x_rounded, cw, ccw)
        if load < best_load:
            best_routing = x_rounded
            best_load = load

    # Local search: try flipping each bit, and pairs
    improved = True
    while improved:
        improved = False
        # Single-bit flips
        for i in range(D):
            candidate = best_routing.copy()
            candidate[i] = 1.0 - candidate[i]
            load = _max_load(candidate, cw, ccw)
            if load < best_load - 1e-12:
                best_routing = candidate
                best_load = load
                improved = True
        # Pair flips
        for i in range(D):
            if time.time() > deadline:
                break
            for j in range(i + 1, D):
                candidate = best_routing.copy()
                candidate[i] = 1.0 - candidate[i]
                candidate[j] = 1.0 - candidate[j]
                load = _max_load(candidate, cw, ccw)
                if load < best_load - 1e-12:
                    best_routing = candidate
                    best_load = load
                    improved = True
            if time.time() > deadline:
                break

    # Second DE run with different seed if time permits
    if time.time() < deadline - 60:
        try:
            result2 = differential_evolution(
                objective,
                bounds=bounds,
                maxiter=200,
                popsize=20,
                tol=1e-8,
                seed=123,
                polish=True,
            )
            if result2.x is not None:
                x_rounded = np.where(result2.x >= 0.5, 1.0, 0.0)
                load = _max_load(x_rounded, cw, ccw)
                if load < best_load:
                    best_routing = x_rounded
                    best_load = load
                # Local search again
                improved = True
                while improved:
                    improved = False
                    for i in range(D):
                        if time.time() > deadline:
                            break
                        candidate = best_routing.copy()
                        candidate[i] = 1.0 - candidate[i]
                        load = _max_load(candidate, cw, ccw)
                        if load < best_load - 1e-12:
                            best_routing = candidate
                            best_load = load
                            improved = True
                    if time.time() > deadline:
                        break
        except Exception:
            pass

    return best_routing


# EVOLVE-BLOCK-END
