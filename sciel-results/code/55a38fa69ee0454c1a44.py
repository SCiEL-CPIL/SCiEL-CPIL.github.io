import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    time_limit = 145.0  # Leave 5s margin for safety

    # Precompute indices for all triples (286 triples)
    triples = list(combinations(range(N), 3))
    i_idx = np.array([t[0] for t in triples])
    j_idx = np.array([t[1] for t in triples])
    k_idx = np.array([t[2] for t in triples])

    def hull_area(points):
        """Convex hull area, or 0.0 on failure."""
        try:
            if len(np.unique(np.round(points, 12), axis=0)) < 3:
                return 0.0
            h = ConvexHull(points)
            return h.volume
        except Exception:
            return 0.0

    def min_triangle_area(points):
        """Minimum area over all triples of points."""
        x = points[:, 0]
        y = points[:, 1]
        v1x = x[j_idx] - x[i_idx]
        v1y = y[j_idx] - y[i_idx]
        v2x = x[k_idx] - x[i_idx]
        v2y = y[k_idx] - y[i_idx]
        areas = 0.5 * np.abs(v1x * v2y - v1y * v2x)
        return float(np.min(areas))

    def score(points):
        """min triangle area / convex hull area. 0 on degenerate."""
        ha = hull_area(points)
        if ha < 1e-9:
            return 0.0
        return min_triangle_area(points) / ha

    def neg_score_flat(xf):
        """Hard objective: negative score for minimization."""
        pts = xf.reshape(-1, 2)
        s = score(pts)
        return -s

    def _regular_polygon(n, radius=1.0, center=(0.0, 0.0), phase=0.0):
        if n < 2:
            return np.array([[center[0], center[1]]] * n)
        angles = phase + np.linspace(0, 2 * np.pi, n, endpoint=False)
        xs = center[0] + radius * np.cos(angles)
        ys = center[1] + radius * np.sin(angles)
        return np.column_stack([xs, ys])

    def _fibonacci_spiral(n, radius=1.0, center=(0.0, 0.0)):
        """Golden-angle spiral: near-uniform disk distribution."""
        golden = np.pi * (3.0 - np.sqrt(5.0))
        idx = np.arange(n)
        r = radius * np.sqrt((idx + 0.5) / n)
        theta = golden * idx
        xs = center[0] + r * np.cos(theta)
        ys = center[1] + r * np.sin(theta)
        return np.column_stack([xs, ys])

    def _make_initial_configurations():
        """Diverse initial configurations emphasizing hull structure."""
        configs = []

        # Regular polygons
        for n in range(3, 14):
            configs.append(_regular_polygon(n, radius=1.0))

        # Concentric ring families: outer ring + inner ring
        ring_specs = [
            (12, 1, 0.0),
            (11, 2, 0.3),
            (10, 3, 0.35),
            (9, 4, 0.4),
            (8, 5, 0.45),
            (7, 6, 0.5),
            (6, 7, 0.55),
            (5, 8, 0.6),
            (4, 9, 0.65),
            (3, 10, 0.7),
        ]
        for outer_n, inner_n, inner_r in ring_specs:
            outer = _regular_polygon(outer_n, radius=1.0)
            if inner_n > 0:
                inner = _regular_polygon(inner_n, radius=inner_r, phase=np.pi / outer_n)
                configs.append(np.vstack([outer, inner]))
            else:
                configs.append(outer)

        # Fibonacci spiral variants
        for rad in [0.9, 1.0, 1.1]:
            configs.append(_fibonacci_spiral(N, radius=rad))

        # Fibonacci with a couple of hull points pushed out
        base = _fibonacci_spiral(N, radius=1.0)
        configs.append(base)

        # Mixed: outer ring + Fibonacci interior
        for outer_n in [7, 8, 9, 10]:
            inner_n = N - outer_n
            outer = _regular_polygon(outer_n, radius=1.0)
            inner = _fibonacci_spiral(inner_n, radius=0.5)
            configs.append(np.vstack([outer, inner]))

        # Jittered regular 13-gon (multiple seeds)
        rng = np.random.default_rng(7)
        for _ in range(3):
            cfg = _regular_polygon(13, radius=1.0) + rng.normal(0, 0.05, (N, 2))
            configs.append(cfg)

        return configs

    best_points = None
    best_score = -float('inf')

    def update_best(pts, s):
        nonlocal best_points, best_score
        if s > best_score:
            best_score = s
            best_points = pts.copy()

    def polish(pts, remaining_budget, maxiter=400):
        """Local polish using SLSQP (hard objective) then Nelder-Mead."""
        if time.time() > start_time + remaining_budget:
            return pts
        cur = pts.copy()
        # SLSQP on hard objective
        try:
            res = opt.minimize(
                neg_score_flat,
                cur.flatten(),
                method='SLSQP',
                options={'maxiter': int(max(20, remaining_budget * 1.5)),
                         'ftol': 1e-12, 'eps': 1e-8}
            )
            cand = res.x.reshape(-1, 2)
            cs = score(cand)
            if cs > score(cur):
                cur = cand
        except Exception:
            pass
        # Nelder-Mead fine polish
        if time.time() < start_time + remaining_budget - 1:
            try:
                res = opt.minimize(
                    neg_score_flat,
                    cur.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': int(max(50, remaining_budget * 2)),
                             'adaptive': True, 'xatol': 1e-10, 'fatol': 1e-12}
                )
                cand = res.x.reshape(-1, 2)
                cs = score(cand)
                if cs > score(cur):
                    cur = cand
            except Exception:
                pass
        return cur

    configs = _make_initial_configurations()

    # Pass 1: evaluate + polish each config
    for i, cfg in enumerate(configs):
        if time.time() > start_time + time_limit - 20:
            break
        s = score(cfg)
        update_best(cfg, s)
        # polish
        cfg = polish(cfg, remaining_budget=time_limit - 10)
        s = score(cfg)
        update_best(cfg, s)

    # Pass 2: adaptive re-seeding around the current best to escape local minima
    rng = np.random.default_rng(42)
    n_adaptive = 10
    for _ in range(n_adaptive):
        if time.time() > start_time + time_limit - 15:
            break
        if best_points is None:
            break
        # Perturb the best config with a moderate random kick
        sigma = rng.uniform(0.05, 0.25)
        cand = best_points + rng.normal(0, sigma, (N, 2))
        s = score(cand)
        update_best(cand, s)
        cand = polish(cand, remaining_budget=time_limit - 10)
        s = score(cand)
        update_best(cand, s)

    # Pass 3: a few fresh random Fibonacci / ring restarts
    for _ in range(6):
        if time.time() > start_time + time_limit - 10:
            break
        kind = rng.integers(0, 3)
        if kind == 0:
            cand = _fibonacci_spiral(N, radius=rng.uniform(0.85, 1.15))
        elif kind == 1:
            outer_n = int(rng.integers(5, 11))
            inner_n = N - outer_n
            outer = _regular_polygon(outer_n, radius=1.0)
            inner = _fibonacci_spiral(inner_n, radius=rng.uniform(0.3, 0.6))
            cand = np.vstack([outer, inner])
        else:
            cand = _regular_polygon(13, radius=1.0) + rng.normal(0, 0.1, (N, 2))
        s = score(cand)
        update_best(cand, s)
        cand = polish(cand, remaining_budget=time_limit - 8)
        s = score(cand)
        update_best(cand, s)

    # Final high-precision polish
    if best_points is not None and time.time() < start_time + time_limit - 3:
        try:
            res = opt.minimize(
                neg_score_flat,
                best_points.flatten(),
                method='SLSQP',
                options={'maxiter': 200, 'ftol': 1e-14, 'eps': 1e-9}
            )
            cand = res.x.reshape(-1, 2)
            cs = score(cand)
            if cs > best_score:
                best_points = cand
                best_score = cs
        except Exception:
            pass
        if time.time() < start_time + time_limit - 1:
            try:
                res = opt.minimize(
                    neg_score_flat,
                    best_points.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 300, 'adaptive': True, 'xatol': 1e-12, 'fatol': 1e-14}
                )
                cand = res.x.reshape(-1, 2)
                cs = score(cand)
                if cs > best_score:
                    best_points = cand
                    best_score = cs
            except Exception:
                pass

    if best_points is None:
        best_points = _regular_polygon(13, radius=1.0)

    return best_points

# EVOLVE-BLOCK-END
