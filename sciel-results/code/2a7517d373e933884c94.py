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
    1. Use SLSQP optimization with smooth surrogate for min triangle area.
    2. Enforce convexity via constraints on cross products.
    3. Fix scale and center to remove invariances.
    """
    
    # Precompute indices for all triples
    triples = list(combinations(range(N), 3))
    
    def compute_hull_area(pts):
        """Compute convex hull area using shoelace formula on points assumed to be in convex position (sorted)."""
        # Points are assumed to be in CCW order
        n = len(pts)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += pts[i, 0] * pts[j, 1]
            area -= pts[j, 0] * pts[i, 1]
        return 0.5 * abs(area)
    
    def get_min_triangle_area(pts):
        """Compute minimum triangle area among all triples."""
        min_area = float('inf')
        for i, j, k in triples:
            a = pts[i]
            b = pts[j]
            c = pts[k]
            area = 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]))
            if area < min_area:
                min_area = area
        return min_area

    def soft_min_area(pts, lam=100.0):
        """
        Smooth approximation of the minimum triangle area.
        Uses log-sum-exp trick: min(x) approx -1/lam * log(sum(exp(-lam*x)))
        """
        areas = np.zeros(len(triples))
        for idx, (i, j, k) in enumerate(triples):
            a = pts[i]
            b = pts[j]
            c = pts[k]
            areas[idx] = 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]))
        
        # To avoid overflow, subtract max
        max_area = np.max(areas)
        if max_area == 0:
            return 0.0
        exp_terms = np.exp(-lam * (areas - max_area))
        log_sum = np.log(np.sum(exp_terms))
        soft_min = max_area - log_sum / lam
        return soft_min

    def objective(x):
        pts = x.reshape(N, 2)
        
        # Calculate hull area
        hull_area = compute_hull_area(pts)
        if hull_area < 1e-10:
            return 1e6
        
        # Calculate soft min triangle area
        min_area_approx = soft_min_area(pts)
        
        # We want to maximize min_area / hull_area
        # So minimize - (min_area / hull_area)
        return -(min_area_approx / hull_area)

    # Constraints for SLSQP
    def constraint_convexity(x):
        pts = x.reshape(N, 2)
        # Cross products of consecutive edges must be positive (CCW)
        crosses = np.zeros(N)
        for i in range(N):
            p1 = pts[i]
            p2 = pts[(i + 1) % N]
            p3 = pts[(i + 2) % N]
            crosses[i] = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1])
        # We want crosses > 0. Constraint: crosses - epsilon >= 0
        return crosses - 1e-6

    def constraint_centroid(x):
        pts = x.reshape(N, 2)
        return pts.sum(axis=0)

    def constraint_scale(x):
        pts = x.reshape(N, 2)
        # Fix sum of squared distances to 1 to remove scale invariance
        return np.sum(pts**2) - 1.0

    constraints = [
        {'type': 'ineq', 'fun': constraint_convexity},
        {'type': 'eq', 'fun': constraint_centroid},
        {'type': 'eq', 'fun': constraint_scale}
    ]
    
    # Initial guess: Regular polygon
    # Scale it to satisfy sum of squares = 1
    angles_init = np.linspace(0, 2 * np.pi, N, endpoint=False)
    points_init = np.column_stack([np.cos(angles_init), np.sin(angles_init)])
    norm = np.sqrt(np.sum(points_init**2))
    points_init = points_init / norm
    
    x0 = points_init.flatten()
    
    # Run optimization
    result = minimize(
        objective,
        x0,
        method='SLSQP',
        constraints=constraints,
        options={'maxiter': 500, 'ftol': 1e-12}
    )
    
    optimized_points = result.x.reshape(N, 2)
    
    # If optimization failed or converged to bad spot, try regular polygon
    # Check if result is valid (convex)
    def is_valid(pts):
        crosses = []
        for i in range(N):
            p1 = pts[i]
            p2 = pts[(i + 1) % N]
            p3 = pts[(i + 2) % N]
            cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1])
            if cross <= 1e-8:
                return False
        return True
    
    if not is_valid(optimized_points) or result.fun > 0.01: # Regular polygon score is ~0.017
        optimized_points = points_init
        
    # Final normalization: Scale so hull area is 1
    hull_area = compute_hull_area(optimized_points)
    if hull_area > 1e-10:
        scale = 1.0 / np.sqrt(hull_area)
        centroid = optimized_points.mean(axis=0)
        optimized_points = (optimized_points - centroid) * scale
        
    return optimized_points
# EVOLVE-BLOCK-END
