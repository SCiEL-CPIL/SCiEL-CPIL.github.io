"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _polygon_area_from_sorted_angles(angles):
    """
    Calculate the area of the convex hull (polygon) formed by points 
    on the unit circle with given sorted angles.
    """
    if len(angles) < 3:
        return 0.0
    # Sort angles to ensure correct polygon order
    sorted_angles = np.sort(angles)
    x = np.cos(sorted_angles)
    y = np.sin(sorted_angles)
    # Shoelace formula
    x_next = np.roll(x, -1)
    y_next = np.roll(y, -1)
    area = 0.5 * np.abs(np.sum(x * y_next - x_next * y))
    return area


def _min_triangle_area_normalized_fast(angles):
    """
    Compute the minimum area of any triangle formed by 3 points 
    on the unit circle with given angles, normalized by the convex hull area.
    
    angles: (N,) array of angles (radians)
    Returns: float, the normalized min triangle area (higher is better)
    """
    # Calculate convex hull area (polygon area for points on circle)
    hull_area = _polygon_area_from_sorted_angles(angles)
    if hull_area <= 1e-12:
        return 0.0
    
    # Convert angles to points
    x = np.cos(angles)
    y = np.sin(angles)
    points = np.column_stack([x, y])
    n = len(angles)
    
    # Find minimum triangle area
    min_area = float('inf')
    # Vectorized calculation of triangle areas is possible but N=14 is small,
    # so a loop is sufficient and easier to manage for clarity, 
    # though we can optimize the inner loop.
    for i in range(n):
        p1 = points[i]
        for j in range(i + 1, n):
            p2 = points[j]
            v1 = p2 - p1
            for k in range(j + 1, n):
                p3 = points[k]
                v2 = p3 - p1
                cross = v1[0] * v2[1] - v1[1] * v2[0]
                area = 0.5 * abs(cross)
                if area < min_area:
                    min_area = area
    
    return min_area / hull_area


def _objective(params):
    """
    Objective function for optimization: minimize negative of min normalized triangle area.
    
    params: (N,) array of angles (radians) on a circle
    Returns: float, negative of the normalized min triangle area
    """
    score = _min_triangle_area_normalized_fast(params)
    return -score


def construct():
    """
    Construct N points in a planar convex region to maximize the minimum
    triangle area normalized by convex hull area.
    
    Returns: (N, 2) array of points
    """
    best_score = -np.inf
    best_angles = None
    
    # Strategy: Multi-start Nelder-Mead
    # Start 1: Regular polygon
    initial_configs = []
    initial_configs.append(np.linspace(0, 2 * np.pi, N, endpoint=False))
    
    # Add several random perturbations of the regular polygon
    rng = np.random.default_rng(42)
    for _ in range(5):
        noise = rng.normal(0, 0.1, N)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False) + noise
        # Normalize angles to [0, 2pi)
        angles = angles % (2 * np.pi)
        initial_configs.append(angles)
        
    # Also try a few purely random sorted configurations
    for _ in range(3):
        random_angles = rng.uniform(0, 2 * np.pi, N)
        random_angles.sort()
        initial_configs.append(random_angles)
    
    for init_angles in initial_configs:
        # Ensure angles are sorted for consistency, though Nelder-Mead doesn't strictly need it
        # as the objective function handles sorting for area calc, but it helps convergence.
        # However, if we sort, we lose the "identity" of points for the optimizer?
        # No, the objective is symmetric with respect to permutation of points.
        # So sorting is safe and might help.
        # But Nelder-Mead operates on the vector. If we sort, the vector changes discontinuously?
        # No, we just pass the vector to the objective. The objective sorts it.
        # So the optimizer sees a continuous landscape? 
        # Actually, if the optimizer swaps two angles, the sorted order changes, 
        # but the set of points is the same, so the score is the same.
        # This creates a flat direction in the parameter space, which Nelder-Mead handles okay.
        
        try:
            result = minimize(
                _objective,
                init_angles,
                method='Nelder-Mead',
                options={
                    'maxiter': 1000,
                    'xatol': 1e-6,
                    'fatol': 1e-8,
                    'adaptive': True
                }
            )
            
            score = -result.fun
            if score > best_score:
                best_score = score
                best_angles = result.x
        except Exception:
            continue
            
    if best_angles is None:
        # Fallback to regular polygon
        best_angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        
    points = np.column_stack([np.cos(best_angles), np.sin(best_angles)])
    return points
# EVOLVE-BLOCK-END
