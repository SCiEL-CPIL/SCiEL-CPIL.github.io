"""Maximize product of ordered-pair distances for 8 planar points at fixed diameter.

Strategy: represent the 8 points as a centrally symmetric set defined by 4 free
points and their negations (8-D search space). Use differential_evolution for
global search over the 8 free coordinates, plus hand-crafted symmetric seeds
(two concentric squares). Refine each candidate with a multi-stage local
polish (L-BFGS-B -> Powell -> SLSQP -> Nelder-Mead)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from math import cos, sin, pi, log, exp
from scipy.optimize import minimize, differential_evolution


def _score(points):
    """Delta / n^n where Delta is the product of ordered-pair distances after
    scaling the set to diameter 2. The regular octagon scores exactly 1."""
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    if n < 2:
        return 0.0
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    diam = np.max(dists)
    if diam < 1e-12:
        return 0.0
    scale = 2.0 / diam
    dists = dists * scale
    mask = ~np.eye(n, dtype=bool)
    if np.any(dists[mask] < 1e-12):
        return 0.0
    log_prod = np.sum(np.log(dists[mask]))
    return float(np.exp(log_prod) / (n ** n))


def _sym_points(flat8):
    """8 points = 4 free points and their negations, from 8 free coords."""
    p = flat8.reshape(4, 2)
    return np.vstack([p, -p])


def _neg_log_score_8d(flat8):
    """Negative log-score for the centrally-symmetric 8-point config.
    Minimizing this maximizes the product of ordered-pair distances at fixed diameter.
    (The constant 8*log(8) is irrelevant for optimization but kept for clarity.)"""
    pts = _sym_points(flat8)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    diam = np.max(dists)
    if diam < 1e-12:
        return 1e10
    scale = 2.0 / diam
    dists = dists * scale
    mask = ~np.eye(8, dtype=bool)
    if np.any(dists[mask] < 1e-12):
        return 1e10
    log_prod = np.sum(np.log(dists[mask]))
    return 8.0 * log(8.0) - log_prod


def _polish_8d(flat8, deadline):
    """Multi-stage local refinement in the 8-D symmetric space. Returns best flat8."""
    best = flat8.copy()
    best_val = _neg_log_score_8d(best)

    def _obj(x):
        return _neg_log_score_8d(x)

    stages = [
        ('L-BFGS-B', {'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}),
        ('Powell', {'maxiter': 400, 'ftol': 1e-14, 'xtol': 1e-14}),
        ('SLSQP', {'maxiter': 300, 'ftol': 1e-14}),
        ('Nelder-Mead', {'maxiter': 20000, 'fatol': 1e-15, 'xatol': 1e-15, 'adaptive': True}),
    ]
    for method, opts in stages:
        if time.time() > deadline - 3.0:
            break
        try:
            res = minimize(_obj, best, method=method, options=opts)
            if res.fun < best_val - 1e-15:
                best_val = res.fun
                best = res.x.copy()
        except Exception:
            pass
    return best


def _two_squares_flat8(r1, theta1, r2, theta2):
    """Two concentric squares (centrally symmetric) -> 8 free coords (4 points + negs).
    Square 1: radius r1, angles theta1 + {0, pi/2, pi, 3pi/2}.
    Square 2: radius r2, angles theta2 + {0, pi/2, pi, 3pi/2}.
    Return the 8 flat coords of the 4 unique points (one per square pair)."""
    a1 = theta1 + np.array([0.0, pi / 2, pi, 3 * pi / 2])
    a2 = theta2 + np.array([0.0, pi / 2, pi, 3 * pi / 2])
    p1 = np.column_stack([r1 * np.cos(a1), r1 * np.sin(a1)])
    p2 = np.column_stack([r2 * np.cos(a2), r2 * np.sin(a2)])
    # 4 unique points: take square1 point0, square2 point0, and their ...
    # Actually the 8 points are p1[0..3] and p2[0..3]; negation pairs:
    # p1[2] = -p1[0], p1[3] = -p1[1], p2[2] = -p2[0], p2[3] = -p2[1].
    # So 4 unique points = [p1[0], p1[1], p2[0], p2[1]].
    unique = np.vstack([p1[:2], p2[:2]])
    return unique.flatten()


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150 s kill

    best_flat = None
    best_score = 0.0

    def _consider(flat8):
        nonlocal best_flat, best_score
        pts = _sym_points(flat8)
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_flat = flat8.copy()

    # ---- Hand-crafted symmetric seeds (two concentric squares) ----
    seeds = []
    for r1 in [1.0, 1.2, 0.9, 1.3]:
        for r2 in [0.6, 0.8, 1.0, 0.5]:
            for t1 in [0.0, pi / 4]:
                for t2 in [0.0, pi / 4, pi / 8]:
                    seeds.append(_two_squares_flat8(r1, t1, r2, t2))
    # Regular octagon (centrally symmetric): 4 points at 45,90,135,180 deg? 
    # Regular octagon vertices at angle 2pi*k/8, k=0..7; negation pairs are k and k+4.
    # 4 unique = k=0,1,2,3 at radius 1.
    reg_angles = np.array([0.0, pi / 4, pi / 2, 3 * pi / 4])
    seeds.append(np.column_stack([np.cos(reg_angles), np.sin(reg_angles)]).flatten())

    for s in seeds:
        _consider(s)

    # ---- Global search: differential evolution on 8 free coordinates ----
    bounds = [(-2.0, 2.0)] * 8
    try:
        if time.time() < deadline - 20.0:
            de_res = differential_evolution(
                _neg_log_score_8d,
                bounds,
                maxiter=200,
                popsize=30,
                seed=42,
                tol=1e-10,
                polish=False,
                init='sobol',
                mutation=(0.3, 1.0),
                recombination=0.8,
            )
            if de_res.fun < 1e6:
                _consider(de_res.x)
    except Exception:
        pass

    # ---- Refinement: polish every promising candidate in the 8-D symmetric space ----
    candidates = [s for s in seeds]
    if best_flat is not None:
        candidates.append(best_flat)
    # Add a few random perturbations of the best for basin diversity
    rng = np.random.default_rng(0)
    if best_flat is not None:
        for scale in [0.1, 0.3, 0.6]:
            candidates.append(best_flat + rng.normal(0, scale, size=8))

    for cand in candidates:
        if time.time() > deadline - 5.0:
            break
        try:
            polished = _polish_8d(cand, deadline)
            _consider(polished)
        except Exception:
            pass

    # ---- Final high-precision polish of the overall best ----
    if best_flat is not None and time.time() < deadline - 2.0:
        try:
            final = _polish_8d(best_flat, deadline)
            _consider(final)
        except Exception:
            pass

    # ---- Build the final 8-point list (4 points + negations) ----
    pts = _sym_points(best_flat)
    return [[float(p[0]), float(p[1])] for p in pts]
# EVOLVE-BLOCK-END
