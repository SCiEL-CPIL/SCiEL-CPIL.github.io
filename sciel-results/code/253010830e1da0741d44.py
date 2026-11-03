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
            pts = points - points.mean(axis=0)
            hull = ConvexHull(pts)
            area_hull = hull.volume
        except Exception:
            return 0.0
        if area_hull < 1e-9:
            return 0.0
        x = pts[:, 0]
        y = pts[:, 1]
        v1x = x[tri_j] - x[tri_i]
        v1y = y[tri_j] - y[tri_i]
        v2x = x[tri_k] - x[tri_i]
        v2y = y[tri_k] - y[tri_i]
        areas = 0.5 * np.abs(v1x * v2y - v1y * v2x)
        min_area = float(np.min(areas))
        return min_area / area_hull

    def _symmetric_config(r_outer, r_inner, phase_offset):
        """
        Generate 13 points as two nested regular polygons:
        - 12 points on outer ring of radius r_outer
        - 1 point at center
        With relative phase offset between rings.
        Actually, let's use: 12 outer + 1 center, but parameterize more flexibly.
        
        Better: Use 11 outer + 2 inner (one at center, one offset)
        Or: 12 outer ring + 1 center, with the 12 having a phase offset
        """
        # 12 points on outer ring
        angles_outer = np.linspace(0, 2 * np.pi, 12, endpoint=False) + phase_offset
        outer_pts = np.column_stack([r_outer * np.cos(angles_outer), r_outer * np.sin(angles_outer)])
        
        # 1 point at center
        center_pt = np.array([[0.0, 0.0]])
        
        pts = np.vstack([outer_pts, center_pt])
        return pts

    def _symmetric_score(params):
        """
        Score for symmetric configuration.
        params = [r_outer, r_inner_ratio, phase_offset]
        We use 12 outer + 1 center, but also allow an inner ring.
        
        Actually, let's use a more general symmetric form:
        - 12 points on outer ring of radius r_outer
        - 1 point at center
        But this is only 3 parameters. Let's add more flexibility.
        
        Alternative: 11 outer + 2 inner
        Or: 12 outer with slight radial perturbation
        """
        r_outer, r_inner_ratio, phase_offset = params
        if r_outer < 0.1 or r_inner_ratio < 0 or r_inner_ratio > 1:
            return -1e6
        
        # 12 points on outer ring
        angles_outer = np.linspace(0, 2 * np.pi, 12, endpoint=False) + phase_offset
        outer_pts = np.column_stack([r_outer * np.cos(angles_outer), r_outer * np.sin(angles_outer)])
        
        # 1 point at center (or slightly offset for more flexibility)
        center_pt = np.array([[0.0, 0.0]])
        
        pts = np.vstack([outer_pts, center_pt])
        
        # Normalize to unit area
        pts_centered = pts - pts.mean(axis=0)
        try:
            hull = ConvexHull(pts_centered)
            area = hull.volume
            if area < 1e-9:
                return -1e6
            pts_norm = pts_centered * np.sqrt(1.0 / area)
        except Exception:
            return -1e6
        
        return compute_score(pts_norm)

    def _neg_symmetric_score(params):
        """Negative symmetric score for minimization."""
        s = _symmetric_score(params)
        if s <= 0:
            return 1e6
        return -s

    def _normalize_points(points):
        """Center and scale points so convex hull has unit area."""
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

    def _optimize_symmetric_space(time_budget):
        """
        Optimize in the reduced symmetric parameter space first.
        Returns best 13-point configuration found in symmetric space.
        """
        # Grid search over symmetric parameters
        best_sym_pts = None
        best_sym_score = -float('inf')
        
        # Coarse grid
        r_outer_vals = np.linspace(0.8, 1.5, 10)
        phase_vals = np.linspace(0, np.pi, 8)
        
        for r_o in r_outer_vals:
            for phase in phase_vals:
                params = [r_o, 0.5, phase]  # r_inner_ratio not used in simple version
                s = _symmetric_score(params)
                if s > best_sym_score:
                    best_sym_score = s
                    best_sym_pts = _symmetric_config(r_o, 0.5, phase)
        
        # Refine with Nelder-Mead in symmetric space
        if best_sym_pts is not None and time.time() < time_budget:
            start_params = [1.0, 0.5, 0.0]
            try:
                res = opt.minimize(
                    _neg_symmetric_score,
                    start_params,
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
                s = _symmetric_score(res.x)
                if s > best_sym_score:
                    best_sym_score = s
                    best_sym_pts = _symmetric_config(res.x[0], res.x[1], res.x[2])
            except Exception:
                pass
        
        # Try multiple starting points in symmetric space
        rng_sym = np.random.RandomState(123)
        for _ in range(5):
            if time.time() > time_budget:
                break
            init = rng_sym.uniform([0.7, 0.3, 0], [1.3, 0.8, np.pi])
            try:
                res = opt.minimize(
                    _neg_symmetric_score,
                    init,
                    method='Nelder-Mead',
                    options={'maxiter': 100, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
                s = _symmetric_score(res.x)
                if s > best_sym_score:
                    best_sym_score = s
                    best_sym_pts = _symmetric_config(res.x[0], res.x[1], res.x[2])
            except Exception:
                pass
        
        return best_sym_pts, best_sym_score

    def _soft_objective(points_flat, beta=50.0):
        """
        Normalized soft-min objective. Points are centered and scaled to unit hull area.
        Returns negative soft-min triangle area (since hull area = 1, score = min_area).
        """
        pts = points_flat.reshape(-1, 2)
        pts = _normalize_points(pts)
        x = pts[:, 0]
        y = pts[:, 1]
        v1x = x[tri_j] - x[tri_i]
        v1y = y[tri_j] - y[tri_i]
        v2x = x[tri_k] - x[tri_i]
        v2y = y[tri_k] - y[tri_i]
        areas = 0.5 * np.abs(v1x * v2y - v1y * v2x)
        neg_beta_areas = -beta * areas
        max_val = np.max(neg_beta_areas)
        if max_val < -1e6:
            return 1e6
        log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_areas - max_val)))
        soft_min_area = -log_sum_exp / beta
        return -soft_min_area

    def _refine_from_symmetric(sym_pts, time_budget):
        """
        Take a symmetric configuration and refine it in full 26D space.
        This allows breaking symmetry to find better solutions.
        """
        if sym_pts is None:
            return None, -float('inf')
        
        best = sym_pts.copy()
        best_s = compute_score(best)
        current = sym_pts.copy()
        
        # Stage 1: L-BFGS-B with annealed soft-min
        for beta in [10.0, 25.0, 60.0, 120.0, 250.0]:
            if time.time() > time_budget:
                break
            try:
                res = opt.minimize(
                    lambda x: _soft_objective(x, beta),
                    current.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 50, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                cand = res.x.reshape(-1, 2)
                s = compute_score(cand)
                if s > best_s:
                    best_s = s
                    best = cand.copy()
                current = cand
            except Exception:
                pass
        
        # Stage 2: Powell with hard objective
        if time.time() < time_budget - 2:
            try:
                res = opt.minimize(
                    _hard_objective,
                    best.flatten(),
                    method='Powell',
                    options={'maxiter': 80, 'xtol': 1e-12, 'ftol': 1e-12}
                )
                cand = res.x.reshape(-1, 2)
                s = compute_score(cand)
                if s > best_s:
                    best_s = s
                    best = cand.copy()
            except Exception:
                pass
        
        # Stage 3: Coordinate descent
        if time.time() < time_budget - 1:
            try:
                cd_best, cd_s = _coordinate_descent(best, n_rounds=2, time_budget=time_budget - 1)
                if cd_s > best_s:
                    best_s = cd_s
                    best = cd_best.copy()
            except Exception:
                pass
        
        # Stage 4: Nelder-Mead final polish
        if time.time() < time_budget:
            try:
                res = opt.minimize(
                    _hard_objective,
                    best.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 150, 'adaptive': True, 'xatol': 1e-10, 'fatol': 1e-12}
                )
                cand = res.x.reshape(-1, 2)
                s = compute_score(cand)
                if s > best_s:
                    best_s = s
                    best = cand.copy()
            except Exception:
                pass
        
        return best, best_s

    def _hard_objective(points_flat):
        """Negative hard score on normalized points."""
        pts = points_flat.reshape(-1, 2)
        pts = _normalize_points(pts)
        score = compute_score(pts)
        if score <= 0:
            return 1e6
        return -score

    def _coordinate_descent(pts, n_rounds=3, time_budget=None):
        """
        Iteratively optimize each point (2D) while fixing others, on normalized space.
        """
        best = pts.copy()
        best_s = compute_score(best)
        for rnd in range(n_rounds):
            if time_budget is not None and time.time() > time_budget:
                break
            improved = False
            for i in range(N):
                if time_budget is not None and time.time() > time_budget:
                    break
                fixed = best.copy()
                def obj_i(x2):
                    cand = fixed.copy()
                    cand[i] = x2
                    return _hard_objective(cand.flatten())
                try:
                    res = opt.minimize(
                        obj_i, best[i].copy(), method='Nelder-Mead',
                        options={'maxiter': 40, 'xatol': 1e-9, 'fatol': 1e-11}
                    )
                    cand = best.copy()
                    cand[i] = res.x
                    s = compute_score(cand)
                    if s > best_s + 1e-12:
                        best_s = s
                        best = cand
                        improved = True
                except Exception:
                    pass
            if not improved:
                break
        return best, best_s

    def _regular_polygon(n, radius=1.0, center=(0.0, 0.0)):
        if n == 1:
            return np.array([center])
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        xs = center[0] + radius * np.cos(angles)
        ys = center[1] + radius * np.sin(angles)
        return np.column_stack([xs, ys])

    best_points = None
    best_score = -float('inf')

    # --- Phase 1: Optimize in symmetric parameter space ---
    sym_budget = time_limit - 30.0
    sym_pts, sym_score = _optimize_symmetric_space(sym_budget)
    
    if sym_pts is not None:
        best_points = sym_pts.copy()
        best_score = sym_score

    # --- Phase 2: Refine symmetric solution in full 26D space ---
    if best_points is not None:
        refine_budget = time_limit - 15.0
        cand, cand_s = _refine_from_symmetric(best_points, refine_budget)
        if cand is not None and cand_s > best_score:
            best_score = cand_s
            best_points = cand.copy()

    # --- Phase 3: Try diverse asymmetric initializations ---
    rng = np.random.RandomState(42)
    diverse_starts = [
        _regular_polygon(13, radius=1.0),
        np.vstack([_regular_polygon(12, 1.0), np.array([[0.0, 0.0]])]),
        np.vstack([_regular_polygon(11, 1.0), np.array([[-0.3, 0.2], [0.3, -0.2]])]),
        np.vstack([_regular_polygon(10, 1.0), _regular_polygon(3, 0.4)]),
        np.vstack([_regular_polygon(9, 1.0), _regular_polygon(4, 0.5)]),
        np.vstack([_regular_polygon(8, 1.0), _regular_polygon(5, 0.5)]),
    ]
    
    # Add perturbed versions
    for start in diverse_starts[:4]:
        perturbed = start.copy()
        perturbed += rng.randn(*perturbed.shape) * 0.03
        diverse_starts.append(perturbed)
    
    remaining_budget = time_limit - 10.0
    per_diverse = remaining_budget / len(diverse_starts)
    
    for idx, start_pts in enumerate(diverse_starts):
        if time.time() - start_time > time_limit - 10:
            break
        s = compute_score(start_pts)
        if s > best_score:
            best_score = s
            best_points = start_pts.copy()
        
        # Quick refinement for diverse starts
        alloc = min(per_diverse, time_limit - 10 - (time.time() - start_time))
        if alloc > 2:
            try:
                # Quick L-BFGS-B
                res = opt.minimize(
                    lambda x: _soft_objective(x, 50.0),
                    start_pts.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 30, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                cand = res.x.reshape(-1, 2)
                s2 = compute_score(cand)
                if s2 > best_score:
                    best_score = s2
                    best_points = cand.copy()
            except Exception:
                pass

    # --- Phase 4: Final high-precision polish ---
    if best_points is not None:
        remaining = time_limit - 5.0 - (time.time() - start_time)
        if remaining > 8:
            try:
                res = opt.minimize(
                    lambda x: _soft_objective(x, 300.0),
                    best_points.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-15, 'gtol': 1e-13}
                )
                cand = res.x.reshape(-1, 2)
                s = compute_score(cand)
                if s > best_score:
                    best_score = s
                    best_points = cand.copy()
            except Exception:
                pass

        remaining = time_limit - 3.0 - (time.time() - start_time)
        if remaining > 5:
            try:
                res = opt.minimize(
                    _hard_objective,
                    best_points.flatten(),
                    method='Powell',
                    options={'maxiter': 100, 'xtol': 1e-13, 'ftol': 1e-13}
                )
                cand = res.x.reshape(-1, 2)
                s = compute_score(cand)
                if s > best_score:
                    best_score = s
                    best_points = cand.copy()
            except Exception:
                pass

        remaining = time_limit - 2.0 - (time.time() - start_time)
        if remaining > 3:
            try:
                cd_best, cd_s = _coordinate_descent(best_points, n_rounds=3, time_budget=time_limit - 2.0)
                if cd_s > best_score:
                    best_score = cd_s
                    best_points = cd_best.copy()
            except Exception:
                pass

        remaining = time_limit - 1.0 - (time.time() - start_time)
        if remaining > 2:
            try:
                res = opt.minimize(
                    _hard_objective,
                    best_points.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'adaptive': True, 'xatol': 1e-12, 'fatol': 1e-14}
                )
                cand = res.x.reshape(-1, 2)
                s = compute_score(cand)
                if s > best_score:
                    best_score = s
                    best_points = cand.copy()
            except Exception:
                pass

    if best_points is None:
        best_points = _regular_polygon(13, radius=1.0)

    return best_points

# EVOLVE-BLOCK-END
