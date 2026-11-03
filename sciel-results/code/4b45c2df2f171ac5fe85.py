import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution

N = 13

def construct():
    start_time = time.time()
    time_limit = 145.0  # Leave 5s margin for safety

    # Precompute indices for all triples as numpy arrays for vectorized computation
    tri_i = []
    tri_j = []
    tri_k = []
    for i in range(N):
        for j in range(i+1, N):
            for k in range(j+1, N):
                tri_i.append(i)
                tri_j.append(j)
                tri_k.append(k)
    tri_i = np.array(tri_i, dtype=np.int32)
    tri_j = np.array(tri_j, dtype=np.int32)
    tri_k = np.array(tri_k, dtype=np.int32)
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
        # Vectorized triangle area computation
        v1x = x[tri_j] - x[tri_i]
        v1y = y[tri_j] - y[tri_i]
        v2x = x[tri_k] - x[tri_i]
        v2y = y[tri_k] - y[tri_i]
        cross = np.abs(v1x * v2y - v1y * v2x)
        areas = 0.5 * cross
        min_area = np.min(areas)

        score = min_area / area_hull
        return score, area_hull

    def _regular_polygon(n, radius=1.0, center=(0.0, 0.0)):
        """Generate points of a regular n-gon."""
        if n == 1:
            return np.array([center])
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        xs = center[0] + radius * np.cos(angles)
        ys = center[1] + radius * np.sin(angles)
        return np.column_stack([xs, ys])

    # Strategy 1: 1+12 ring symmetry - central point + 12 boundary points
    # Parameterize 12 boundary points with optimized radii (angles fixed at regular spacing)
    # This reduces the search space significantly while maintaining convexity

    def _make_ring_config(radii, center_offset=(0.0, 0.0)):
        """
        Create a configuration with 1 central point and 12 boundary points.
        radii: array of 12 radii for boundary points
        The 12 boundary points are at angles 2*pi*k/12 with radius radii[k]
        """
        angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        xs = center_offset[0] + radii * np.cos(angles)
        ys = center_offset[1] + radii * np.sin(angles)
        boundary_pts = np.column_stack([xs, ys])
        center_pt = np.array([[center_offset[0], center_offset[1]]])
        return np.vstack([center_pt, boundary_pts])

    def _ring_objective(params):
        """
        Objective for the ring configuration.
        params: 14 values = 12 radii + 2 center offsets
        Returns negative score (for minimization).
        """
        if time.time() > start_time + time_limit - 5:
            return 1e6

        radii = params[:12]
        cx, cy = params[12], params[13]

        # Ensure radii are positive and within bounds
        radii = np.clip(radii, 0.1, 2.0)

        points = _make_ring_config(radii, (cx, cy))

        score, _ = compute_score(points)
        return -score

    # Strategy 2: General 13-gon with optimized positions
    def _general_objective(params):
        """
        Objective for general 13-point configuration.
        params: 26 values = 13 x (x, y)
        Returns negative score (for minimization).
        """
        if time.time() > start_time + time_limit - 5:
            return 1e6

        points = params.reshape(13, 2)
        score, _ = compute_score(points)
        return -score

    best_points = None
    best_score = -float('inf')

    # Try multiple initial configurations and optimize each

    # Initial config 1: Regular 13-gon
    config1 = _regular_polygon(13, radius=1.0)
    s1, _ = compute_score(config1)
    if s1 > best_score:
        best_score = s1
        best_points = config1.copy()

    # Initial config 2: 1+12 ring with uniform radii
    radii_uniform = np.ones(12) * 1.0
    config2 = _make_ring_config(radii_uniform, (0.0, 0.0))
    s2, _ = compute_score(config2)
    if s2 > best_score:
        best_score = s2
        best_points = config2.copy()

    # Initial config 3: 1+12 ring with slightly perturbed radii
    rng = np.random.RandomState(42)
    radii_perturbed = np.ones(12) * 1.0 + rng.randn(12) * 0.05
    config3 = _make_ring_config(radii_perturbed, (0.0, 0.0))
    s3, _ = compute_score(config3)
    if s3 > best_score:
        best_score = s3
        best_points = config3.copy()

    # Optimize using differential evolution on the ring parameter space
    # This is a 14-dimensional problem (12 radii + 2 center offsets)
    if time.time() < start_time + time_limit - 30:
        try:
            bounds_ring = [(0.1, 2.0)] * 12 + [(-0.5, 0.5), (-0.5, 0.5)]
            de_result = differential_evolution(
                _ring_objective,
                bounds_ring,
                maxiter=50,
                popsize=15,
                tol=1e-8,
                seed=42,
                polish=True,
                init='sobol'
            )
            if de_result.success or de_result.fun < 0:
                opt_params = de_result.x
                radii_opt = np.clip(opt_params[:12], 0.1, 2.0)
                cx_opt, cy_opt = opt_params[12], opt_params[13]
                opt_points = _make_ring_config(radii_opt, (cx_opt, cy_opt))
                opt_score, _ = compute_score(opt_points)
                if opt_score > best_score:
                    best_score = opt_score
                    best_points = opt_points.copy()
        except Exception:
            pass

    # Optimize using L-BFGS-B on the general 13-point space from best config
    if best_points is not None and time.time() < start_time + time_limit - 20:
        try:
            result = opt.minimize(
                _general_objective,
                best_points.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-13, 'maxls': 100}
            )
            opt_points = result.x.reshape(13, 2)
            opt_score, _ = compute_score(opt_points)
            if opt_score > best_score:
                best_score = opt_score
                best_points = opt_points.copy()
        except Exception:
            pass

    # Try a few more random restarts with differential evolution on ring space
    if time.time() < start_time + time_limit - 25:
        for seed in [1, 2, 3]:
            if time.time() > start_time + time_limit - 15:
                break
            try:
                bounds_ring = [(0.1, 2.0)] * 12 + [(-0.5, 0.5), (-0.5, 0.5)]
                de_result = differential_evolution(
                    _ring_objective,
                    bounds_ring,
                    maxiter=30,
                    popsize=12,
                    tol=1e-7,
                    seed=seed + 100,
                    polish=True
                )
                if de_result.fun < 0:
                    opt_params = de_result.x
                    radii_opt = np.clip(opt_params[:12], 0.1, 2.0)
                    cx_opt, cy_opt = opt_params[12], opt_params[13]
                    opt_points = _make_ring_config(radii_opt, (cx_opt, cy_opt))
                    opt_score, _ = compute_score(opt_points)
                    if opt_score > best_score:
                        best_score = opt_score
                        best_points = opt_points.copy()
            except Exception:
                pass

    # Final Nelder-Mead polish
    if best_points is not None and time.time() < start_time + time_limit - 10:
        try:
            res = opt.minimize(
                _general_objective,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 300, 'adaptive': True, 'xatol': 1e-12, 'fatol': 1e-14}
            )
            nm_points = res.x.reshape(13, 2)
            nm_score, _ = compute_score(nm_points)
            if nm_score > best_score:
                best_score = nm_score
                best_points = nm_points.copy()
        except Exception:
            pass

    # Ensure we have a valid result
    if best_points is None:
        best_points = _regular_polygon(13, radius=1.0)

    return best_points
