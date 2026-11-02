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

    def _regular_polygon(n, radius=1.0, center=(0.0, 0.0), phase=0.0):
        """Generate points of a regular n-gon."""
        if n == 1:
            return np.array([center])
        angles = phase + np.linspace(0, 2 * np.pi, n, endpoint=False)
        xs = center[0] + radius * np.cos(angles)
        ys = center[1] + radius * np.sin(angles)
        return np.column_stack([xs, ys])

    def _make_symmetric_config(n_outer, n_mid, n_inner, r_outer=1.0, r_mid=0.5, 
                                phase_outer=0.0, phase_mid=0.0, phase_inner=0.0):
        """
        Create a symmetric configuration with points on concentric rings.
        n_outer + n_mid + n_inner = 13
        """
        pts = []
        if n_outer > 0:
            pts.append(_regular_polygon(n_outer, radius=r_outer, center=(0.0, 0.0), phase=phase_outer))
        if n_mid > 0:
            pts.append(_regular_polygon(n_mid, radius=r_mid, center=(0.0, 0.0), phase=phase_mid))
        if n_inner == 1:
            pts.append(np.array([[0.0, 0.0]]))
        elif n_inner > 1:
            r_inner = 0.2 * r_mid if r_mid > 0 else 0.2
            pts.append(_regular_polygon(n_inner, radius=r_inner, center=(0.0, 0.0), phase=phase_inner))
        if not pts:
            return _regular_polygon(N, radius=1.0)
        return np.vstack(pts)

    def _optimize_from(start_pts, time_budget):
        """Run multi-stage optimization from a starting configuration."""
        best = start_pts.copy()
        best_s, _ = compute_score(best)
        current = start_pts.copy()

        # Stage 1: L-BFGS-B with annealed soft-min beta
        for beta in [20.0, 50.0, 100.0, 200.0]:
            if time.time() - start_time > time_budget:
                break
            try:
                res = opt.minimize(
                    lambda x: _soft_min_score(x.reshape(-1, 2), beta),
                    current.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 80, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                cand = res.x.reshape(-1, 2)
                s, _ = compute_score(cand)
                if s > best_s:
                    best_s = s
                    best = cand.copy()
                current = cand
            except Exception:
                pass

        # Stage 2: Nelder-Mead polish on hard objective
        if time.time() - start_time < time_budget - 2:
            try:
                res = opt.minimize(
                    lambda x: -compute_score(x.reshape(-1, 2))[0],
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

        # Stage 3: Local perturbation search
        if time.time() - start_time < time_budget - 1:
            for _ in range(5):
                if time.time() - start_time > time_budget - 1:
                    break
                perturbed = best + np.random.normal(0, 0.01, size=best.shape)
                s, _ = compute_score(perturbed)
                if s > best_s:
                    best_s = s
                    best = perturbed.copy()
                    # Further optimize from this new point
                    try:
                        res = opt.minimize(
                            lambda x: _soft_min_score(x.reshape(-1, 2), 100.0),
                            best.flatten(),
                            method='L-BFGS-B',
                            options={'maxiter': 50, 'ftol': 1e-14, 'gtol': 1e-12}
                        )
                        cand = res.x.reshape(-1, 2)
                        s2, _ = compute_score(cand)
                        if s2 > best_s:
                            best_s = s2
                            best = cand.copy()
                    except Exception:
                        pass

        return best, best_s

    best_points = None
    best_score = -float('inf')

    # --- Phase 1: Symmetric configurations as starting points ---
    # These enforce high symmetry and avoid local minima traps
    symmetric_starts = [
        # 1 center + 12 outer (regular 12-gon)
        _make_symmetric_config(12, 0, 1, r_outer=1.0, r_mid=0.0,
                               phase_outer=0.0, phase_mid=0.0, phase_inner=0.0),
        # 6 outer + 6 mid + 1 center
        _make_symmetric_config(6, 6, 1, r_outer=1.0, r_mid=0.55,
                               phase_outer=0.0, phase_mid=np.pi/6, phase_inner=0.0),
        # 6 outer + 5 mid + 2 inner
        _make_symmetric_config(6, 5, 2, r_outer=1.0, r_mid=0.5,
                               phase_outer=0.0, phase_mid=np.pi/5, phase_inner=0.0),
        # 7 outer + 5 mid + 1 center
        _make_symmetric_config(7, 5, 1, r_outer=1.0, r_mid=0.48,
                               phase_outer=0.0, phase_mid=np.pi/7, phase_inner=0.0),
        # 5 outer + 7 mid + 1 center
        _make_symmetric_config(5, 7, 1, r_outer=1.0, r_mid=0.6,
                               phase_outer=0.0, phase_mid=np.pi/7, phase_inner=0.0),
        # 4 outer + 8 mid + 1 center
        _make_symmetric_config(4, 8, 1, r_outer=1.0, r_mid=0.55,
                               phase_outer=0.0, phase_mid=np.pi/8, phase_inner=0.0),
        # 13 on a single ring (regular 13-gon)
        _make_symmetric_config(13, 0, 0, r_outer=1.0, r_mid=0.0,
                               phase_outer=0.0, phase_mid=0.0, phase_inner=0.0),
    ]

    # Also add some asymmetric variations
    for phase_offset in [np.pi/12, np.pi/10, np.pi/8]:
        symmetric_starts.append(
            _make_symmetric_config(6, 6, 1, r_outer=1.0, r_mid=0.55,
                                   phase_outer=0.0, phase_mid=phase_offset, phase_inner=0.0)
        )

    # Allocate time per start
    n_starts = len(symmetric_starts)
    # Reserve ~15s for final polish
    search_budget = time_limit - 15.0
    per_start = search_budget / n_starts

    for idx, start_pts in enumerate(symmetric_starts):
        if time.time() - start_time > time_limit - 15:
            break
        # Give first few starts slightly more budget
        alloc = per_start * (1.2 if idx < 4 else 1.0)
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
                    options={'maxiter': 120, 'ftol': 1e-15, 'gtol': 1e-14}
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
            # Nelder-Mead final polish
            try:
                res = opt.minimize(
                    lambda x: -compute_score(x.reshape(-1, 2))[0],
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
