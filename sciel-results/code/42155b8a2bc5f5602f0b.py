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
    1. Parameterize points by angles (sorted) and radii to enforce convex position.
    2. Use logarithmic barrier for convexity: enforce positive cross products.
    3. Optimize to maximize minimum triangle area / hull area.
    """
    
    # Generate regular 13-gon centered at origin as initial guess
    angles_init = np.linspace(0, 2 * np.pi, N, endpoint=False)
    points_init = np.column_stack([np.cos(angles_init), np.sin(angles_init)])
    
    def compute_hull_area(pts):
        """Compute convex hull area using shoelace formula on sorted points."""
        # Sort by angle around centroid
        centroid = pts.mean(axis=0)
        angles = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
        sorted_idx = np.argsort(angles)
        pts_sorted = pts[sorted_idx]
        
        # Shoelace formula
        n = len(pts_sorted)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += pts_sorted[i, 0] * pts_sorted[j, 1]
            area -= pts_sorted[j, 0] * pts_sorted[i, 1]
        return 0.5 * abs(area)
    
    def min_triangle_area(pts):
        """Compute minimum triangle area among all triples."""
        n = len(pts)
        min_area = float('inf')
        for i, j, k in combinations(range(n), 3):
            a = pts[i]
            b = pts[j]
            c = pts[k]
            area = 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]))
            if area < min_area:
                min_area = area
        return min_area
    
    def is_convex_position(pts):
        """Check if points are in convex position using cross products."""
        # Sort by angle around centroid
        centroid = pts.mean(axis=0)
        angles = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
        sorted_idx = np.argsort(angles)
        pts_sorted = pts[sorted_idx]
        
        n = len(pts_sorted)
        for i in range(n):
            p1 = pts_sorted[i]
            p2 = pts_sorted[(i + 1) % n]
            p3 = pts_sorted[(i + 2) % n]
            cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1])
            if cross <= 1e-10:
                return False
        return True
    
    # Objective function with barrier for convexity
    def objective(x):
        pts = x.reshape(N, 2)
        
        # Check convex position
        if not is_convex_position(pts):
            return 1e6
        
        hull_area = compute_hull_area(pts)
        if hull_area < 1e-10:
            return 1e6
        
        mt_area = min_triangle_area(pts)
        
        # Maximize mt_area / hull_area
        return -(mt_area / hull_area)
    
    # Start from regular polygon
    x0 = points_init.flatten()
    
    # Use Nelder-Mead optimization
    result = minimize(
        objective,
        x0,
        method='Nelder-Mead',
        options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
    )
    
    optimized_points = result.x.reshape(N, 2)
    
    # If optimization failed, fall back to regular polygon
    if result.fun > 1e5:
        optimized_points = points_init
    
    # Normalize: center at origin and scale so convex hull area = 1
    hull_area = compute_hull_area(optimized_points)
    if hull_area > 1e-10:
        scale = 1.0 / np.sqrt(hull_area)
        centroid = optimized_points.mean(axis=0)
        optimized_points = (optimized_points - centroid) * scale
    
    return optimized_points
# EVOLVE-BLOCK-END
