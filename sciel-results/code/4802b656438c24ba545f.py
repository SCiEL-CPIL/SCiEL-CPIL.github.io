import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import ConvexHull

N = 13

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    time_limit = 145.0  # Leave 5s margin for safety

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

    def _normalize_points(points):
        """Center points and scale to unit convex hull area."""
        pts = points - points.mean(axis=0)
        try:
            hull = ConvexHull(pts)
            area = hull.volume
            if area < 1e-9:
                return pts
            pts = pts * np.sqrt(1.0 / area)
        except Exception:
            pass
        return pts

    def compute_score(points):
        """Compute min triangle area / convex hull area for given points."""
        try:
            pts = points - points.mean(axis=0)
            hull = ConvexHull(pts)
            area_hull = hull.volume
        except Exception:
            return 0.0, 0.0

        if area_hull < 1e-9:
            return 0.0, 0.0

        x = pts[:, 0]
        y = pts[:, 1]
        # Vectorized cross products for all 286 triples
        v1x = x[tri_j] - x[tri_i]
        v1y = y[tri_j] - y[tri_i]
        v2x = x[tri_k] - x[tri_i]
        v2y = y[tri_k] - y[tri_i]
        areas = 0.5 * np.abs(v1x * v2y - v1y * v2x)
        min_area = float(np.min(areas))

        score = min_area / area_hull
        return score, area_hull

    def _soft_objective(points_flat, beta=50.0):
        """
        Soft-min objective for gradient-based optimization.
        Normalizes points to unit hull area internally, so the objective
        is just -soft_min_area (since hull area = 1).
        """
        pts = points_flat.reshape(-1, 2)
        pts = _normalize_points(pts)

        x = pts[:, 0]
        y = pts[:, 1]

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

        return -soft_min_area

    def _hard_objective(points_flat):
        """Negative hard score for final evaluation."""
        pts = points_flat.reshape(-1, 2)
        pts = _normalize_points(pts)
        score, _ = compute_score(pts)
        if score < 1e-12:
            return 1e6
        return -score

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
            r3 = 0.25 * r2 if r2 > 0 else 0.25
            pts.append(np.column_stack([r3 * np.cos(a), r3 * np.sin(a)]))
        if not pts:
            return _regular_polygon(N, radius=1.0)
        return np.vstack(pts)

    def _optimize_from(start_pts, time_budget):
        """Run multi-stage optimization from a starting configuration."""
        best = start_pts.copy()
        best_s, _ = compute_score(best)
        current = start_pts.copy()

        # Stage 1: L-BFGS-B with annealed soft-min beta
        for beta in [10.0, 20.0, 40.0, 80.0, 120.0, 200.0]:
            if time.time() - start_time > time_budget:
                break
            try:
                res = opt.minimize(
                    lambda x, b=beta: _soft_objective(x, b),
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

        # Stage 2: Powell refinement on hard objective
        if time.time() - start_time < time_budget - 4:
            try:
                res = opt.minimize(
                    _hard_objective,
                    best.flatten(),
                    method='Powell',
                    options={'maxiter': 200, 'xtol': 1e-10, 'ftol': 1e-12}
                )
                cand = res.x.reshape(-1, 2)
                s, _ = compute_score(cand)
                if s > best_s:
                    best_s = s
                    best = cand.copy()
            except Exception:
                pass

        # Stage 3: Nelder-Mead polish on hard objective
        if time.time() - start_time < time_budget - 2:
            try:
                res = opt.minimize(
                    _hard_objective,
                    best.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 400, 'adaptive': True, 'xatol': 1e-11, 'fatol': 1e-13}
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

    # --- Phase 1: Symmetric ring initializations ---
    ring_starts = [
        # 1 center + 12 outer
        _ring_config(r1=1.0, r2=0.0, n1=12, n2=0, n3=1),
        # 1 center + 6 mid + 6 outer
        _ring_config(r1=1.0, r2=0.5, n1=6, n2=6, n3=1),
        # 1 center + 5 mid + 7 outer
        _ring_config(r1=1.0, r2=0.45, n1=7, n2=5, n3=1),
        # 1 center + 4 mid + 8 outer
        _ring_config(r1=1.0, r2=0.4, n1=8, n2=4, n3=1),
        # 1 center + 12 outer with phase offset
        _ring_config(r1=1.0, r2=0.0, phase1=np.pi/12, n1=12, n2=0, n3=1),
        # 1 center + 3 mid + 9 outer
        _ring_config(r1=1.0, r2=0.55, n1=9, n2=3, n3=1),
        # 1 center + 7 mid + 5 outer
        _ring_config(r1=1.0, r2=0.35, n1=5, n2=7, n3=1),
        # 1 center + 6 mid + 6 outer with phase offset
        _ring_config(r1=1.0, r2=0.5, phase1=0.0, phase2=np.pi/6, n1=6, n2=6, n3=1),
        # 1 center + 11 outer + 1 extra inner
        _ring_config(r1=1.0, r2=0.2, n1=11, n2=1, n3=1),
        # Regular 13-gon
        _regular_polygon(13, 1.0),
    ]

    # Add diverse configurations from inspiration: varying inner ring sizes
    for r_inner in [0.3, 0.5, 0.7]:
        # 10 outer + 3 inner
        ring_starts.append(_ring_config(r1=1.0, r2=r_inner, n1=10, n2=3, n3=0))
        # 8 outer + 5 inner
        ring_starts.append(_ring_config(r1=1.0, r2=r_inner, n1=8, n2=5, n3=0))
        # 7 outer + 6 inner
        ring_starts.append(_ring_config(r1=1.0, r2=r_inner, n1=7, n2=6, n3=0))

    # Add some perturbed versions to break symmetry
    rng = np.random.RandomState(42)
    for idx in [0, 1, 4, 9]:
        perturbed = ring_starts[idx].copy()
        perturbed += rng.randn(*perturbed.shape) * 0.1
        ring_starts.append(perturbed)

    n_starts = len(ring_starts)
    search_budget = time_limit - 15.0
    per_start = search_budget / n_starts

    for idx, start_pts in enumerate(ring_starts):
        if time.time() - start_time > time_limit - 15:
            break
        alloc = per_start * (1.2 if idx < 10 else 0.8)
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
        if remaining > 10:
            # Soft-min L-BFGS-B with very high beta
            try:
                res = opt.minimize(
                    lambda x: _soft_objective(x, 300.0),
                    best_points.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 150, 'ftol': 1e-15, 'gtol': 1e-14}
                )
                cand = res.x.reshape(-1, 2)
                s, _ = compute_score(cand)
                if s > best_score:
                    best_score = s
                    best_points = cand.copy()
            except Exception:
                pass

        remaining = time_limit - 4.0 - (time.time() - start_time)
        if remaining > 7:
            # Powell final polish
            try:
                res = opt.minimize(
                    _hard_objective,
                    best_points.flatten(),
                    method='Powell',
                    options={'maxiter': 300, 'xtol': 1e-12, 'ftol': 1e-14}
                )
                cand = res.x.reshape(-1, 2)
                s, _ = compute_score(cand)
                if s > best_score:
                    best_score = s
                    best_points = cand.copy()
            except Exception:
                pass

        remaining = time_limit - 2.0 - (time.time() - start_time)
        if remaining > 4:
            # Nelder-Mead final polish
            try:
                res = opt.minimize(
                    _hard_objective,
                    best_points.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'adaptive': True, 'xatol': 1e-13, 'fatol': 1e-15}
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
