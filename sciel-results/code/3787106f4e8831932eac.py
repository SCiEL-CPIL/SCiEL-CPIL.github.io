"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from itertools import combinations
from scipy.spatial import ConvexHull


def construct():
    """
    Place 13 points in a planar convex region normalized to unit area to maximize
    the minimum area over all triples.
    
    Approach: Logarithmic barrier method for convex position constraint.
    1. Start with regular 13-gon.
    2. Optimize coordinates to maximize smooth min-triangle-area surrogate.
    3. Enforce convex position via log-barrier on cross products of angularly-ordered points.
    4. Normalize by convex hull area.
    """
    
    def compute_cross_products(pts):
        """Compute cross products for consecutive triples in angular order."""
        # Sort by angle from centroid
        centroid = pts.mean(axis=0)
        angles = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
        order = np.argsort(angles)
        pts_sorted = pts[order]
        
        n = len(pts_sorted)
        cross_prods = np.zeros(n)
        for i in range(n):
            p0 = pts_sorted[i]
            p1 = pts_sorted[(i + 1) % n]
            p2 = pts_sorted[(i + 2) % n]
            # Cross product (p1-p0) x (p2-p1)
            v1 = p1 - p0
            v2 = p2 - p1
            cross_prods[i] = v1[0] * v2[1] - v1[1] * v2[0]
        return cross_prods, order
    
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
    
    def smooth_min_triangle_area(pts, k=50.0):
        """Smooth approximation of minimum triangle area using log-sum-exp."""
        n = len(pts)
        areas = np.zeros(len(list(combinations(range(n), 3))))
        idx = 0
        for i, j, k in combinations(range(n), 3):
            a = pts[i]
            b = pts[j]
            c = pts[k]
            area = 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]))
            areas[idx] = area
            idx += 1
        
        # Smooth min: -log(sum(exp(-k*areas)))/k approximates min(areas) for large k
        # Use log-sum-exp for numerical stability
        max_neg = -k * np.min(areas)
        log_sum = np.log(np.sum(np.exp(-k * areas - max_neg))) + max_neg
        smooth_min = -log_sum / k
        return smooth_min
    
    def convex_hull_area(pts):
        try:
            hull = ConvexHull(pts)
            return hull.volume
        except:
            return 1e-10
    
    # Generate regular 13-gon centered at origin
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Objective function with logarithmic barrier
    def objective(x, mu):
        pts = x.reshape(N, 2)
        
        # Compute cross products for convexity check
        cross_prods, _ = compute_cross_products(pts)
        
        # If any cross product is non-positive, we're not in convex position
        min_cross = np.min(cross_prods)
        if min_cross <= 1e-10:
            # Heavy penalty for non-convex
            return 1e10 - mu * np.sum(np.log(np.maximum(cross_prods, 1e-15)))
        
        # Log barrier term: -mu * sum(log(cross_prods))
        barrier = -mu * np.sum(np.log(cross_prods))
        
        # Smooth min triangle area (we want to maximize this, so minimize negative)
        smooth_min = smooth_min_triangle_area(pts, k=50.0)
        
        # Normalize by hull area
        hull_area = convex_hull_area(pts)
        if hull_area < 1e-10:
            return 1e10
        
        ratio = smooth_min / hull_area
        return -ratio + barrier
    
    # Multi-stage optimization with decreasing mu
    best_points = points.copy()
    best_score = -1e10
    
    for mu in [1.0, 0.1, 0.01, 0.001, 0.0001]:
        x0 = best_points.flatten()
        
        result = minimize(
            lambda x: objective(x, mu),
            x0,
            method='Nelder-Mead',
            options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
        )
        
        opt_pts = result.x.reshape(N, 2)
        
        # Check if this is valid (convex position)
        cross_prods, _ = compute_cross_products(opt_pts)
        if np.min(cross_prods) > 1e-8:
            mt_area = min_triangle_area(opt_pts)
            hull_area = convex_hull_area(opt_pts)
            if hull_area > 1e-10:
                score = mt_area / hull_area
                if score > best_score:
                    best_score = score
                    best_points = opt_pts
    
    # Final: try a few more iterations with very small mu if we have a good solution
    if best_score > 0:
        x0 = best_points.flatten()
        result = minimize(
            lambda x: objective(x, 1e-6),
            x0,
            method='Nelder-Mead',
            options={'maxiter': 300, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
        )
        opt_pts = result.x.reshape(N, 2)
        cross_prods, _ = compute_cross_products(opt_pts)
        if np.min(cross_prods) > 1e-8:
            mt_area = min_triangle_area(opt_pts)
            hull_area = convex_hull_area(opt_pts)
            if hull_area > 1e-10:
                score = mt_area / hull_area
                if score > best_score:
                    best_points = opt_pts
    
    # Normalize: center at origin and scale so convex hull area = 1
    hull_area = convex_hull_area(best_points)
    if hull_area > 1e-10:
        scale = 1.0 / np.sqrt(hull_area)
        centroid = best_points.mean(axis=0)
        best_points = (best_points - centroid) * scale
    
    return best_points
# EVOLVE-BLOCK-END
