import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize_scalar

N = 14

def _compute_areas(points):
    """
    Compute areas of all triangles formed by triples of points.
    Returns a 1D array of length C(N, 3).
    """
    n = len(points)
    # Use broadcasting to compute all differences
    # points shape: (n, 2)
    # We want v1 = p[j] - p[i], v2 = p[k] - p[i]
    # It is more efficient to compute cross products directly.
    # Area(i,j,k) = 0.5 * | (x_j - x_i)*(y_k - y_i) - (y_j - y_i)*(x_k - x_i) |
    
    # Vectorized approach for C(14,3) = 364 triangles is fast enough,
    # but we can do it even faster with full broadcasting if memory allows,
    # or just simple loops since 364 is small. 
    # However, for coordinate descent, we call this many times.
    # Let's use a semi-vectorized approach.
    
    # Create indices
    i, j, k = np.indices((n, n, n))
    # Mask for i < j < k
    mask = (i < j) & (j < k)
    
    # Extract coordinates
    xi = points[i[mask], 0]
    yi = points[i[mask], 1]
    xj = points[j[mask], 0]
    yj = points[j[mask], 1]
    xk = points[k[mask], 0]
    yk = points[k[mask], 1]
    
    # Calculate cross products
    # cross = (xj-xi)*(yk-yi) - (yj-yi)*(xk-xi)
    cross = (xj - xi) * (yk - yi) - (yj - yi) * (xk - xi)
    
    areas = 0.5 * np.abs(cross)
    return areas

def _hull_area(points):
    """Compute area of convex hull."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """
    Score: Min triangle area / Convex Hull Area.
    """
    hull_area = _hull_area(points)
    if hull_area < 1e-6:
        return 0.0
    areas = _compute_areas(points)
    if len(areas) == 0:
        return 0.0
    min_area = np.min(areas)
    return min_area / hull_area

def _generate_two_ring(n_inner, n_outer, r_inner, r_outer, phase_inner=0, phase_outer=0):
    """
    Generate points on two concentric circles.
    """
    points = []
    # Inner ring
    angles_in = 2 * np.pi * np.arange(n_inner) / n_inner + phase_inner
    for a in angles_in:
        points.append([r_inner * np.cos(a), r_inner * np.sin(a)])
    
    # Outer ring
    angles_out = 2 * np.pi * np.arange(n_outer) / n_outer + phase_outer
    for a in angles_out:
        points.append([r_outer * np.cos(a), r_outer * np.sin(a)])
    
    return np.array(points)

def _polish_points(points, max_iter=100, time_limit=10.0):
    """
    Coordinate descent polish.
    Iterates through each point and optimizes its x and y coordinates
    to maximize the score.
    """
    start_time = time.time()
    current_score = _score(points)
    points = points.copy()
    
    for iter in range(max_iter):
        if time.time() - start_time > time_limit:
            break
            
        improved = False
        for i in range(N):
            # Save original point
            p_orig = points[i].copy()
            
            # Optimize x
            # Define function to maximize: score with x varied
            def obj_x(x):
                points[i, 0] = x
                return -_score(points) # Minimize negative score
            
            # Bounds for x: roughly within hull extent
            # Use a reasonable box
            x_min = np.min(points[:, 0]) - 1.0
            x_max = np.max(points[:, 0]) + 1.0
            
            try:
                res_x = minimize_scalar(obj_x, bounds=(x_min, x_max), method='bounded', options={'maxiter': 20})
                if res_x.success:
                    points[i, 0] = res_x.x
            except Exception:
                points[i, 0] = p_orig[0]
            
            # Optimize y
            def obj_y(y):
                points[i, 1] = y
                return -_score(points)
            
            y_min = np.min(points[:, 1]) - 1.0
            y_max = np.max(points[:, 1]) + 1.0
            
            try:
                res_y = minimize_scalar(obj_y, bounds=(y_min, y_max), method='bounded', options={'maxiter': 20})
                if res_y.success:
                    points[i, 1] = res_y.y
            except Exception:
                points[i, 1] = p_orig[1]
                
            # Check improvement
            new_score = _score(points)
            if new_score > current_score:
                current_score = new_score
                improved = True
            else:
                # Revert if no improvement (or if it got worse)
                points[i] = p_orig
                
        if not improved:
            break
            
    return points, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    best_score = 0.0
    best_points = None
    
    # Generate initial configurations
    # Strategy: Two rings.
    # Total N=14.
    # Possible splits: (2,12), (3,11), (4,10), (5,9), (6,8), (7,7)
    # We'll try a few promising splits and radii.
    
    configs = []
    
    # Common heuristic for packing: outer ring has more points.
    splits = [(2, 12), (3, 11), (4, 10), (5, 9), (6, 8)]
    
    for n_in, n_out in splits:
        # Try different radius ratios
        # Area of circle is pi*r^2.
        # We want to maximize min triangle area.
        # Regular polygon area scales with R^2.
        # Let's normalize the final points anyway, so relative radii matter.
        
        # Typical good ratio for inner radius vs outer radius?
        # If inner points are too close to outer, triangles become thin.
        # If inner points are too close to center, they are far from outer points but close to each other?
        
        for r_ratio in [0.5, 0.6, 0.7, 0.8]:
            r_out = 1.0
            r_in = r_ratio * r_out
            
            # Try different phase shifts to break symmetry
            for phase_in in [0, np.pi / n_in]:
                for phase_out in [0, np.pi / n_out]:
                    pts = _generate_two_ring(n_in, n_out, r_in, r_out, phase_in, phase_out)
                    configs.append(pts)
                    
    # Also add a single ring (regular 14-gon) as a baseline
    angles = 2 * np.pi * np.arange(N) / N
    R = 1.0 # Normalized later
    init_regular = np.column_stack([R * np.cos(angles), R * np.sin(angles)])
    configs.append(init_regular)
    
    # Shuffle configs to ensure we try diverse ones early?
    # Or just iterate.
    
    for pts in configs:
        if time.time() > deadline - 5.0: # Leave 5s for final check
            break
            
        # Normalize initial points to unit hull area to help optimizer?
        # The score is invariant to scaling, but the optimizer works in fixed coordinates.
        # It's better to normalize so the scale is consistent.
        hull_a = _hull_area(pts)
        if hull_a > 0:
            pts = pts / np.sqrt(hull_a)
            
        # Polish
        polished_pts, score = _polish_points(pts, max_iter=50, time_limit=10.0)
        
        if score > best_score:
            best_score = score
            best_points = polished_pts
            
        # If we have a very good score, we might want to stop early?
        # Theoretical max for 14 points?
        # For N points in unit area, max min triangle area is related to packing.
        # Let's just run through all configs if time permits.
        
    if best_points is None:
        # Fallback: Regular polygon
        best_points = init_regular
        hull_a = _hull_area(best_points)
        if hull_a > 0:
            best_points = best_points / np.sqrt(hull_a)
        else:
            best_points = np.eye(2, N) # Just some valid points
            # Need to return shape (N,2)
            best_points = np.array([[0,0],[1,0],[0,1]] + [[0.1, 0.1]]*(N-3))
            hull_a = _hull_area(best_points)
            if hull_a > 0:
                best_points = best_points / np.sqrt(hull_a)

    return best_points

# EVOLVE-BLOCK-END
