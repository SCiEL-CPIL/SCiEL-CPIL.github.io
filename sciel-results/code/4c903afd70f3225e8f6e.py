"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import itertools

def _min_triangle_area_over_hull(points):
    """
    Compute min triangle area / convex hull area for given points.
    Returns the ratio (higher is better).
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    n = points.shape[0]
    if n < 3:
        return 0.0
    
    # Compute convex hull area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume  # In 2D, volume is area
    except Exception:
        return 0.0
    
    if hull_area <= 1e-10:
        return 0.0
    
    # Vectorized triangle area calculation
    min_area = float('inf')
    # Precompute cross products for speed? For N=14, 364 triangles is small.
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                # Area = 0.5 * |cross(p2-p1, p3-p1)|
                v1 = p2 - p1
                v2 = p3 - p1
                area = 0.5 * abs(v1[0]*v2[1] - v1[1]*v2[0])
                if area < min_area:
                    min_area = area
                    if min_area < 1e-12:
                        return 0.0
    
    return min_area / hull_area

def _smoothed_min_triangle_area(points, beta=100.0):
    """
    Smoothed approximation of the negative minimum triangle area ratio.
    We want to maximize min_area/hull_area, so we minimize -min_area/hull_area.
    Using log-sum-exp to approximate the minimum: min(x) ~ -1/beta * log(sum(exp(-beta*x)))
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    n = points.shape[0]
    if n < 3:
        return 1e10 # Penalize invalid
    
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 1e10
        
    if hull_area <= 1e-10:
        return 1e10

    # Calculate all triangle areas
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                v1 = p2 - p1
                v2 = p3 - p1
                area = 0.5 * abs(v1[0]*v2[1] - v1[1]*v2[0])
                areas.append(area)
    
    areas = np.array(areas)
    
    # Smoothed min: -1/beta * log(sum(exp(-beta * areas)))
    # To avoid overflow/underflow, shift by max
    max_area = np.max(areas)
    if max_area == 0:
        return 1e10
        
    # exp(-beta * (areas - max_area)) is stable
    # sum(exp(-beta * (areas - max_area)))
    # log_sum = log(sum) + beta * max_area
    # smoothed_min = -1/beta * log_sum
    # We want to maximize smoothed_min/hull_area.
    # So minimize -smoothed_min/hull_area.
    
    # Actually, let's just minimize the negative of the smoothed min area ratio.
    # ratio = smoothed_min / hull_area
    # We want to maximize ratio.
    # Loss = -ratio
    
    # Compute log-sum-exp safely
    # y = -beta * (areas - max_area)
    y = -beta * (areas - max_area)
    log_sum_exp = np.log(np.sum(np.exp(y)))
    log_min_approx = -1.0 / beta * (log_sum_exp + beta * max_area)
    
    # log_min_approx is approx min_area
    ratio = log_min_approx / hull_area
    
    return -ratio

def construct():
    """
    Uses deterministic local optimization (L-BFGS-B) on a smoothed objective
    from multiple diverse starting points.
    """
    
    best_points = None
    best_score = -1.0
    
    # Define diverse starting configurations
    starts = []
    
    # 1. Regular Polygon
    angles = 2 * np.pi * np.arange(N) / N
    starts.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # 2. Circle with random jitter (fixed seed for reproducibility of start)
    rng = np.random.RandomState(123)
    angles = 2 * np.pi * np.arange(N) / N + rng.randn(N) * 0.1
    radii = 1.0 + rng.randn(N) * 0.1
    starts.append(np.column_stack([radii * np.cos(angles), radii * np.sin(angles)]))
    
    # 3. Grid-like (two concentric rings)
    # 7 points on outer ring, 6 on inner ring, 1 center? No, 14 points.
    # Let's try 1 center, 5 inner, 8 outer? Or 7/7.
    # 7 points on circle r=1, 7 points on circle r=0.5
    angles_outer = 2 * np.pi * np.arange(7) / 7
    angles_inner = 2 * np.pi * np.arange(7) / 7 + np.pi/7 # Offset
    p_outer = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
    p_inner = np.column_stack([0.5 * np.cos(angles_inner), 0.5 * np.sin(angles_inner)])
    starts.append(np.vstack([p_outer, p_inner]))
    
    # 4. Random points in unit disk
    angles = 2 * np.pi * rng.rand(N)
    radii = np.sqrt(rng.rand(N))
    starts.append(np.column_stack([radii * np.cos(angles), radii * np.sin(angles)]))

    for start in starts:
        # Flatten for optimizer
        x0 = start.flatten()
        
        # Objective function wrapper
        def objective(x):
            pts = x.reshape(N, 2)
            return _smoothed_min_triangle_area(pts, beta=200.0)
        
        try:
            # Use L-BFGS-B for speed, it's good for smooth-ish functions
            res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-12})
            
            # Evaluate the true (non-smoothed) score on the result
            res_points = res.x.reshape(N, 2)
            score = _min_triangle_area_over_hull(res_points)
            
            if score > best_score:
                best_score = score
                best_points = res_points.copy()
                
        except Exception:
            continue
            
    # Fallback if all optimizations failed or returned bad scores
    if best_points is None or best_score < 0.01:
        # Return regular polygon
        angles = 2 * np.pi * np.arange(N) / N
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        
    return best_points
# EVOLVE-BLOCK-END
