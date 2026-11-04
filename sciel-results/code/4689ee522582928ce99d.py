"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from itertools import combinations


def construct():
    """
    Place 13 points in a planar convex region normalized to unit area to maximize
    the minimum area over all triples.
    
    Approach:
    1. Parameterize points by sorted angles and radii to enforce convex position.
    2. Use a smooth min-triangle-area surrogate (softmin) for differentiability.
    3. Optimize the ratio min_triangle_area / hull_area using L-BFGS-B.
    4. Normalize output so convex hull area = 1.
    """
    
    # Precompute all triple indices
    triplets = list(combinations(range(N), 3))
    n_triplets = len(triplets)
    
    def compute_hull_area(pts):
        """Compute convex hull area using shoelace formula. Points assumed in angular order."""
        n = len(pts)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += pts[i, 0] * pts[j, 1]
            area -= pts[j, 0] * pts[i, 1]
        return 0.5 * abs(area)
    
    def min_triangle_area_soft(pts, temperature=0.01):
        """
        Compute a smooth approximation of the minimum triangle area.
        Uses log-sum-exp trick on squared areas for numerical stability.
        Returns the softmin of triangle areas.
        """
        n = len(pts)
        areas = np.empty(n_triplets)
        for idx, (i, j, k) in enumerate(triplets):
            a = pts[i]
            b = pts[j]
            c = pts[k]
            cross = (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])
            areas[idx] = 0.5 * abs(cross)
        
        # Softmin: -log(sum(exp(-a/T)))/T approximates min(a)
        # For better conditioning, work with squared areas
        sq_areas = areas ** 2
        max_sq = np.max(sq_areas)
        # Use log-sum-exp for stability
        shifted = -sq_areas / temperature
        max_shifted = np.max(shifted)
        log_sum = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
        softmin_sq = -log_sum * temperature
        softmin_area = np.sqrt(np.maximum(softmin_sq, 0.0))
        return softmin_area
    
    def true_min_triangle_area(pts):
        """Compute exact minimum triangle area."""
        n = len(pts)
        min_area = float('inf')
        for i, j, k in triplets:
            a = pts[i]
            b = pts[j]
            c = pts[k]
            cross = (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])
            area = 0.5 * abs(cross)
            if area < min_area:
                min_area = area
        return min_area
    
    def is_convex_position(pts):
        """Check if points are in convex position using cross products (assumed in angular order)."""
        n = len(pts)
        for i in range(n):
            p1 = pts[i]
            p2 = pts[(i + 1) % n]
            p3 = pts[(i + 2) % n]
            cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1])
            if cross <= 1e-12:
                return False
        return True
    
    def points_from_params(params):
        """
        Convert parameters to points.
        params: [angle_0, ..., angle_{N-1}, r_0, ..., r_{N-1}]
        Angles are sorted, radii are positive.
        Points are in angular order around origin.
        """
        angles = params[:N]
        radii = params[N:]
        # Ensure angles are in [0, 2*pi) and sorted
        angles = np.sort(angles)
        # Wrap to [0, 2*pi)
        angles = angles % (2 * np.pi)
        # Ensure sorted after wrap (handle wrap-around)
        if angles[-1] < angles[0]:
            angles = np.sort(angles)
        # Ensure minimum angular separation
        for i in range(N):
            radii[i] = max(radii[i], 0.01)
        
        x = radii * np.cos(angles)
        y = radii * np.sin(angles)
        return np.column_stack([x, y])
    
    def objective(params):
        pts = points_from_params(params)
        
        # Check convex position
        if not is_convex_position(pts):
            return 1e6
        
        hull_area = compute_hull_area(pts)
        if hull_area < 1e-10:
            return 1e6
        
        mt_area = min_triangle_area_soft(pts, temperature=0.01)
        
        # Maximize mt_area / hull_area
        return -(mt_area / hull_area)
    
    def objective_grad(params):
        """Numerical gradient for L-BFGS-B."""
        eps = 1e-7
        grad = np.zeros_like(params)
        f0 = objective(params)
        for i in range(len(params)):
            params_plus = params.copy()
            params_plus[i] += eps
            f_plus = objective(params_plus)
            grad[i] = (f_plus - f0) / eps
        return grad
    
    # Initial guess: regular 13-gon
    angles_init = np.linspace(0, 2 * np.pi, N, endpoint=False)
    radii_init = np.ones(N)
    params_init = np.concatenate([angles_init, radii_init])
    
    # Run optimization with L-BFGS-B
    result = minimize(
        objective,
        params_init,
        method='L-BFGS-B',
        options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
    )
    
    optimized_params = result.x
    optimized_points = points_from_params(optimized_params)
    
    # If optimization failed, try Nelder-Mead as fallback
    if result.fun > 1e5 or not is_convex_position(optimized_points):
        # Fall back to regular polygon
        points_init = np.column_stack([np.cos(angles_init), np.sin(angles_init)])
        
        def nm_objective(x):
            pts = x.reshape(N, 2)
            if not is_convex_position(pts):
                return 1e6
            hull_area = compute_hull_area(pts)
            if hull_area < 1e-10:
                return 1e6
            mt_area = min_triangle_area_soft(pts, temperature=0.01)
            return -(mt_area / hull_area)
        
        result2 = minimize(
            nm_objective,
            points_init.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
        )
        optimized_points = result2.x.reshape(N, 2)
        if result2.fun > 1e5:
            optimized_points = points_init
    
    # Final check: ensure convex position
    if not is_convex_position(optimized_points):
        optimized_points = np.column_stack([np.cos(angles_init), np.sin(angles_init)])
    
    # Normalize: center at origin and scale so convex hull area = 1
    hull_area = compute_hull_area(optimized_points)
    if hull_area > 1e-10:
        scale = 1.0 / np.sqrt(hull_area)
        centroid = optimized_points.mean(axis=0)
        optimized_points = (optimized_points - centroid) * scale
    
    return optimized_points
# EVOLVE-BLOCK-END
