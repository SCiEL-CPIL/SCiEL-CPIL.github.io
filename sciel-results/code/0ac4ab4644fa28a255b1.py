import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    time_limit = 140.0  # Leave 10s margin for safety

    # Precompute triple index arrays for vectorized area computation
    _tri_i = []
    _tri_j = []
    _tri_k = []
    for i in range(N):
        for j in range(i + 1, N):
            for k in range(j + 1, N):
                _tri_i.append(i)
                _tri_j.append(j)
                _tri_k.append(k)
    tri_i = np.array(_tri_i, dtype=np.int32)
    tri_j = np.array(_tri_j, dtype=np.int32)
    tri_k = np.array(_tri_k, dtype=np.int32)
    n_triples = len(tri_i)

    def compute_score(points):
        """Compute min triangle area / convex hull area for given points."""
        try:
            hull = ConvexHull(points)
            area_hull = hull.volume
        except Exception:
            return 0.0, 0.0

        if area_hull < 1e-9:
            return 0.0, 0.0

        x = points[:, 0]
        y = points[:, 1]
        # Vectorized cross products for all 286 triples
        v1x = x[tri_j] - x[tri_i]
        v1y = y[tri_j] - y[tri_i]
        v2x = x[tri_k] - x[tri_i]
        v2y = y[tri_k] - y[tri_i]
        areas = 0.5 * np.abs(v1x * v2y - v1y * v2x)
        min_area = float(np.min(areas))

        score = min_area / area_hull
        return score, area_hull

    def _soft_min_score(points, beta=50.0):
        """
        Smooth surrogate: use log-sum-exp to approximate the minimum triangle area.
        Returns negative soft-min score for minimization.
        """
        try:
            hull = ConvexHull(points)
            area_hull = hull.volume
        except Exception:
            return 1e6

        if area_hull < 1e-9:
            return 1e6

        x = points[:, 0]
        y = points[:, 1]

        # Vectorized triangle areas
        v1x = x[tri_j] - x[tri_i]
        v1y = y[tri_j] - y[tri_i]
        v2x = x[tri_k] - x[tri_i]
        v2y = y[tri_k] - y[tri_i]
        areas = 0.5 * np.abs(v1x * v2y - v1y * v2x)

        # Soft-min via log-sum-exp (stable)
        neg_beta_areas = -beta * areas
        max_val = np.max(neg_beta_areas)
        if max_val < -1e6:
            return 1e6
        log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_areas - max_val)))
        soft_min_area = -log_sum_exp / beta

        score = soft_min_area / area_hull
        return -score

    def _hard_score_neg(points_flat):
        """Negative hard score for final evaluation."""
        points = points_flat.reshape(-1, 2)
        score, _ = compute_score(points)
        return -score

    def _soft_objective(points_flat, beta=50.0):
        """Soft-min objective for gradient-based optimization."""
        points = points_flat.reshape(-1, 2)
        return _soft_min_score(points, beta)

    def _regular_polygon(n, radius=1.0, center=(0.0, 0.0)):
        """Generate points of a regular n-gon."""
        if n == 1:
            return np.array([center])
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        xs = center[0] + radius * np.cos(angles)
        ys = center[1] + radius * np.sin(angles)
        return np.column_stack([xs, ys])

    def _ring_config(r1, r2, phase1=0.0, phase2=0.0, n1=0, n2=0, n3=0):
        """
        Symmetric ring configuration:
        - n1 points on outer ring of radius r1
        - n2 points on middle ring of radius r2
        - n3 points on inner ring (or center if n3==1)
        Total must equal N.
        """
        pts = []
        if n1 > 0:
            a = np.linspace(phase1, phase1 + 2 * np.pi, n1, endpoint=False)
            pts.append(np.column_stack([r1 * np.cos(a), r1 * np.sin(a)]))
        if n2 > 0:
            a = np.linspace(phase2, phase2 + 2 * np.pi, n2, endpoint=False)
            pts.append(np.column_stack([r2 * np.cos(a), r2 * np.sin(a)]))
        if n3 == 1:
            pts.append(np.array([[0.0, 0.0]]))
        elif n3 > 1:
            a = np.linspace(0, 2 * np.pi, n3, endpoint=False)
            # Inner ring: use a fraction of r2
            r3 = 0.25 * r2 if r2 > 0 else 0.25
            pts.append(np.column_stack([r3 * np.cos(a), r3 * np.sin(a)]))
        if not pts:
            return _regular_polygon(N, radius=1.0)
        return np.vstack(pts)

    def _optimize_from(start_pts, time_budget):
        """
        Run multi-stage optimization from a starting configuration.
        Uses L-BFGS-B with annealed beta, then Powell for robust local search.
        """
        best = start_pts.copy()
        best_s, _ = compute_score(best)
        current = start_pts.copy()

        # Stage 1: L-BFGS-B with annealed soft-min beta
        # More stages for smoother convergence (inspired by observation p206)
        for beta in [10.0, 25.0, 60.0, 120.0, 250.0]:
            if time.time() - start_time > time_budget:
                break
            try:
                res = opt.minimize(
                    lambda x: _soft_min_score(x.reshape(-1, 2), beta),
                    current.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 50, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                cand = res.x.reshape(-1, 2)
                s, _ = compute_score(cand)
                if s > best_s:
                    best_s = s
                    best = cand.copy()
                current = cand
            except Exception:
                pass

        # Stage 2: Powell with hard objective (robust local search, no gradients needed)
        if time.time() - start_time < time_budget - 2:
            try:
                res = opt.minimize(
                    _hard_score_neg,
                    best.flatten(),
                    method='Powell',
                    options={'maxiter': 80, 'xtol': 1e-12, 'ftol': 1e-12}
                )
                cand = res.x.reshape(-1, 2)
                s, _ = compute_score(cand)
                if s > best_s:
                    best_s = s
                    best = cand.copy()
            except Exception:
                pass

        # Stage 3: Nelder-Mead final polish
        if time.time() - start_time < time_budget:
            try:
                res = opt.minimize(
                    lambda x: -compute_score(x.reshape(-1, 2))[0],
                    best.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 150, 'adaptive': True, 'xatol': 1e-10, 'fatol': 1e-12}
                )
                cand = res.x.reshape(-1, 2)
                s, _ = compute_score(cand)
                if s > best_s:
                    best_s = s
                    best = cand.copy()
            except Exception:
                pass

        return best, best_s

    best_points = None
    best_score = -float('inf')

    # --- Phase 1: Diverse symmetric initializations ---
    ring_starts = [
        # 1 center + 12 outer
        _ring_config(r1=1.0, r2=0.0, n1=12, n2=0, n3=1),
        # 1 center + 6 mid + 6 outer
        _ring_config(r1=1.0, r2=0.5, n1=6, n2=6, n3=1),
        # 1 center + 5 mid + 7 outer
        _ring_config(r1=1.0, r2=0.45, n1=7, n2=5, n3=1),
        # 1 center + 4 mid + 8 outer
        _ring_config(r1=1.0, r2=0.4, n1=8, n2=4, n3=1),
        # Regular 13-gon
        _regular_polygon(13, radius=1.0),
        # 11 outer + 2 inner (dipole)
        np.vstack([_regular_polygon(11, 1.0), np.array([[-0.4, 0.0], [0.4, 0.0]])]),
        # 10 outer + 3 inner (triangle)
        np.vstack([_regular_polygon(10, 1.0), _regular_polygon(3, 0.5)]),
        # 9 outer + 4 inner (square)
        np.vstack([_regular_polygon(9, 1.0), _regular_polygon(4, 0.5)]),
        # 8 outer + 5 inner (pentagon)
        np.vstack([_regular_polygon(8, 1.0), _regular_polygon(5, 0.5)]),
        # 7 outer + 6 inner (hexagon)
        np.vstack([_regular_polygon(7, 1.0), _regular_polygon(6, 0.5)]),
        # 6 outer + 1 center + 6 inner
        np.vstack([_regular_polygon(6, 1.0), np.array([[0.0, 0.0]]), _regular_polygon(6, 0.5)]),
        # 5 outer + 8 inner
        np.vstack([_regular_polygon(5, 1.0), _regular_polygon(8, 0.5)]),
    ]

    # Add perturbed versions to break symmetry
    rng = np.random.RandomState(42)
    for i in range(0, len(ring_starts), 2):
        perturbed = ring_starts[i].copy()
        perturbed += rng.randn(*perturbed.shape) * 0.05
        ring_starts.append(perturbed)

    n_starts = len(ring_starts)
    # Reserve ~15s for final polish
    search_budget = time_limit - 15.0
    per_start = search_budget / n_starts

    for idx, start_pts in enumerate(ring_starts):
        if time.time() - start_time > time_limit - 15:
            break
        # Give symmetric starts slightly more budget
        alloc = per_start * (1.2 if idx < 12 else 1.0)
        s, _ = compute_score(start_pts)
        if s > best_score:
            best_score = s
            best_points = start_pts.copy()

        cand, cand_s = _optimize_from(start_pts, start_time + alloc)
        if cand_s > best_score:
            best_score = cand_s
            best_points = cand.copy()

    # --- Phase 2: Final high-precision polish ---
    if best_points is not None:
        remaining = time_limit - 5.0 - (time.time() - start_time)
        if remaining > 8:
            # Soft-min L-BFGS-B with high beta
            try:
                res = opt.minimize(
                    lambda x: _soft_min_score(x.reshape(-1, 2), 300.0),
                    best_points.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-15, 'gtol': 1e-13}
                )
                cand = res.x.reshape(-1, 2)
                s, _ = compute_score(cand)
                if s > best_score:
                    best_score = s
                    best_points = cand.copy()
            except Exception:
                pass

        remaining = time_limit - 3.0 - (time.time() - start_time)
        if remaining > 5:
            # Powell final polish
            try:
                res = opt.minimize(
                    _hard_score_neg,
                    best_points.flatten(),
                    method='Powell',
                    options={'maxiter': 100, 'xtol': 1e-13, 'ftol': 1e-13}
                )
                cand = res.x.reshape(-1, 2)
                s, _ = compute_score(cand)
                if s > best_score:
                    best_score = s
                    best_points = cand.copy()
            except Exception:
                pass

        remaining = time_limit - 2.0 - (time.time() - start_time)
        if remaining > 3:
            # Nelder-Mead final polish
            try:
                res = opt.minimize(
                    lambda x: -compute_score(x.reshape(-1, 2))[0],
                    best_points.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'adaptive': True, 'xatol': 1e-12, 'fatol': 1e-14}
                )
                cand = res.x.reshape(-1, 2)
                s, _ = compute_score(cand)
                if s > best_score:
                    best_score = s
                    best_points = cand.copy()
            except Exception:
                pass

    if best_points is None:
        best_points = _regular_polygon(13, radius=1.0)

    return best_points

# EVOLVE-BLOCK-END
