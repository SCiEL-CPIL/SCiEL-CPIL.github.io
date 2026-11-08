import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13

def construct():
    start_time = time.time()
    time_limit = 110.0  # Safety margin

    # Precompute triangle indices for vectorization
    tri_indices = list(combinations(range(N), 3))
    tri_i = np.array([t[0] for t in tri_indices], dtype=np.int32)
    tri_j = np.array([t[1] for t in tri_indices], dtype=np.int32)
    tri_k = np.array([t[2] for t in tri_indices], dtype=np.int32)

    def compute_score(points):
        """
        Compute min triangle area / convex hull area.
        Returns 0.0 if invalid.
        """
        try:
            # Center the points to avoid numerical issues with large coordinates
            pts = points - points.mean(axis=0)
            hull = ConvexHull(pts)
            area_hull = hull.volume
            if area_hull < 1e-9:
                return 0.0
            
            x = pts[:, 0]
            y = pts[:, 1]
            v1x = x[tri_j] - x[tri_i]
            v1y = y[tri_j] - y[tri_i]
            v2x = x[tri_k] - x[tri_i]
            v2y = y[tri_k] - y[tri_i]
            cross = np.abs(v1x * v2y - v1y * v2x)
            areas = 0.5 * cross
            min_area = np.min(areas)
            
            if min_area < 0:
                return 0.0
            return min_area / area_hull
        except Exception:
            return 0.0

    def _regular_polygon(n, radius=1.0):
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        xs = radius * np.cos(angles)
        ys = radius * np.sin(angles)
        return np.column_stack([xs, ys])

    def points_from_fourier(coeffs, max_k):
        """
        Generate points from Fourier coefficients.
        r(theta) = 1 + sum_{k=1}^{max_k} (a_k cos(k theta) + b_k sin(k theta))
        """
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        radii = np.ones(N)
        for k in range(1, max_k + 1):
            a_k = coeffs[2 * (k - 1)]
            b_k = coeffs[2 * (k - 1) + 1]
            radii += a_k * np.cos(k * angles) + b_k * np.sin(k * angles)
        
        # Ensure points are well-defined (radii > 0)
        # If radii <= 0, the point is invalid or on the boundary.
        # For optimization, we can just let it be, but ConvexHull might fail.
        # We will clamp radii to a small positive value for safety in the objective.
        radii = np.maximum(radii, 0.1)
        
        xs = radii * np.cos(angles)
        ys = radii * np.sin(angles)
        return np.column_stack([xs, ys])

    def objective_fourier(coeffs, max_k):
        pts = points_from_fourier(coeffs, max_k)
        score = compute_score(pts)
        return -score

    def objective_full(points_flat):
        pts = points_flat.reshape(-1, 2)
        score = compute_score(pts)
        return -score

    best_score = -float('inf')
    best_points = None

    def try_update(pts):
        nonlocal best_score, best_points
        score = compute_score(pts)
        if score > best_score:
            best_score = score
            best_points = pts.copy()

    # Initialize with regular polygon
    init_pts = _regular_polygon(N, 1.0)
    try_update(init_pts)

    # Phase 1: Optimize shape using Fourier coefficients
    # Try different levels of detail (max_k)
    # Small max_k captures global shape, larger captures local wiggles
    for max_k in [2, 3, 4, 5]:
        if time.time() > start_time + time_limit - 30:
            break
        
        dim = 2 * max_k
        initial_coeffs = np.zeros(dim)
        
        # Add some random perturbations to escape local minima
        rng = np.random.RandomState(max_k)
        initial_coeffs = rng.normal(0, 0.05, dim)
        
        try:
            res = opt.minimize(
                objective_fourier,
                initial_coeffs,
                args=(max_k,),
                method='Powell',
                options={'maxiter': 200, 'xtol': 1e-6, 'ftol': 1e-8}
            )
            candidate_pts = points_from_fourier(res.x, max_k)
            try_update(candidate_pts)
        except Exception:
            continue

    # Phase 2: Polish in full Cartesian space
    # This allows breaking the radial symmetry if it improves the score
    if best_points is not None:
        if time.time() > start_time + time_limit - 20:
            return best_points
        
        try:
            res = opt.minimize(
                objective_full,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-7, 'fatol': 1e-9}
            )
            candidate_pts = res.x.reshape(-1, 2)
            try_update(candidate_pts)
        except Exception:
            pass

    # Phase 3: Random restarts in Cartesian space with local optimization
    # This helps if the global optimum is not close to the radial symmetric family
    if time.time() < start_time + time_limit - 10:
        rng = np.random.RandomState(123)
        for _ in range(5):
            if time.time() > start_time + time_limit - 5:
                break
            
            # Generate random points in a convex position
            # Sort random angles and radii
            theta = rng.uniform(0, 2 * np.pi, N)
            theta.sort()
            r = rng.uniform(0.8, 1.2, N)
            pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
            
            try:
                res = opt.minimize(
                    objective_full,
                    pts.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-8}
                )
                candidate_pts = res.x.reshape(-1, 2)
                try_update(candidate_pts)
            except Exception:
                continue

    if best_points is None:
        best_points = init_pts

    return best_points
