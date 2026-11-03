"""Heilbronn triangles: maximize min triangle area for 11 points in unit equilateral triangle."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
TRI_AREA = 0.5 * np.sqrt(3) / 2  # area of unit equilateral triangle

def _point_in_triangle(p, eps=1e-8):
    """Check if point p is inside or on the triangle V0,V1,V2."""
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    d1 = sign(p, V0, V1)
    d2 = sign(p, V1, V2)
    d3 = sign(p, V2, V0)
    has_neg = (d1 < -eps) or (d2 < -eps) or (d3 < -eps)
    has_pos = (d1 > eps) or (d2 > eps) or (d3 > eps)
    return not (has_neg and has_pos)

def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by 3 of the points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = 0.5 * abs((points[j][0] - points[i][0]) * (points[k][1] - points[i][1])
                             - (points[k][0] - points[i][0]) * (points[j][1] - points[i][1]))
                if area < min_area:
                    min_area = area
    return min_area

def _fps_init(rng, n_points):
    """Farthest Point Sampling to generate well-spread initial points."""
    # Start with a random point inside the triangle
    points = []
    while len(points) == 0:
        u, v = rng.random(), rng.random()
        if u + v < 1.0:
            p = V0 + u * (V1 - V0) + v * (V2 - V0)
            points.append(p)
    
    for _ in range(n_points - 1):
        # Find the point that maximizes the minimum distance to existing points
        best_p = None
        best_min_dist = -1
        for _ in range(200):  # Sample candidate points
            u, v = rng.random(), rng.random()
            if u + v < 1.0:
                p = V0 + u * (V1 - V0) + v * (V2 - V0)
                # Compute min distance to all existing points
                min_dist = min(np.linalg.norm(p - q) for q in points)
                if min_dist > best_min_dist:
                    best_min_dist = min_dist
                    best_p = p
        points.append(best_p)
    
    return np.array(points)

def _penalized_objective(flat):
    """Objective to minimize: negative min area + penalty for points outside triangle."""
    pts = flat.reshape(-1, 2)
    # Penalty for points outside triangle
    penalty = 0.0
    for p in pts:
        if not _point_in_triangle(p, eps=1e-4):
            # Compute distance to triangle boundary (approximate)
            # Use barycentric coordinates to find how far outside
            # Simple penalty: sum of squared violations
            def sign(p1, p2, p3):
                return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
            d1 = sign(p, V0, V1)
            d2 = sign(p, V1, V2)
            d3 = sign(p, V2, V0)
            # Area of triangle
            tri_area = 0.5 * abs((V1[0]-V0[0])*(V2[1]-V0[1]) - (V2[0]-V0[0])*(V1[1]-V0[1]))
            # Convert signed distances to actual distances (approx)
            # For equilateral triangle side 1, height = sqrt(3)/2
            # The signed distance d_i corresponds to 2*area of sub-triangle
            # So actual perpendicular distance = |d_i| / (side length) = |d_i| / 1
            # But we need to be careful: d_i = cross product, which is 2*area
            # Perpendicular distance from p to edge = |d_i| / |edge|
            # For unit edge, perpendicular distance = |d_i|
            # But the sign tells us which side
            violations = 0.0
            if d1 < 0: violations += (-d1) ** 2
            if d2 < 0: violations += (-d2) ** 2
            if d3 < 0: violations += (-d3) ** 2
            penalty += violations
    if penalty > 0:
        penalty *= 1000.0  # Large penalty
    
    min_area = _min_triangle_area(pts)
    return -min_area + penalty

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_points = None
    best_score = -float('inf')
    
    rng = np.random.RandomState(42)
    
    # Multi-start optimization with FPS initialization
    n_restarts = 50
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Generate initial points using FPS
        init_pts = _fps_init(rng, N)
        
        # Flatten for optimizer
        x0 = init_pts.flatten()
        
        # Run Nelder-Mead optimization
        try:
            result = minimize(
                _penalized_objective,
                x0,
                method='Nelder-Mead',
                options={
                    'maxiter': 500,
                    'xatol': 1e-6,
                    'fatol': 1e-8,
                    'adaptive': True
                }
            )
            
            if result.success or result.fun < best_score:
                opt_pts = result.x.reshape(-1, 2)
                # Check validity
                valid = all(_point_in_triangle(p, eps=1e-4) for p in opt_pts)
                if valid:
                    score = _min_triangle_area(opt_pts)
                    if score > best_score:
                        best_score = score
                        best_points = opt_pts.copy()
        except Exception:
            continue
    
    # If no valid result from optimization, fall back to best FPS init
    if best_points is None:
        # Try a few more FPS inits and pick the best
        for _ in range(20):
            if time.time() > deadline:
                break
            init_pts = _fps_init(rng, N)
            score = _min_triangle_area(init_pts)
            if score > best_score:
                best_score = score
                best_points = init_pts.copy()
    
    # Final check: if still no valid points, return a simple valid configuration
    if best_points is None or not all(_point_in_triangle(p, eps=1e-4) for p in best_points):
        # Return points at vertices and centroid
        best_points = np.array([
            V0, V1, V2,
            (V0 + V1 + V2) / 3,
            (V0 + V1) / 2, (V1 + V2) / 2, (V2 + V0) / 2,
            (V0 + V1 + V2) / 3 + np.array([0.1, 0.05]),
            (V0 + V1 + V2) / 3 + np.array([-0.1, 0.05]),
            (V0 + V1 + V2) / 3 + np.array([0.0, -0.1]),
            (V0 + V1 + V2) / 3 + np.array([0.05, 0.1])
        ])
        # Ensure all inside
        for i in range(len(best_points)):
            if not _point_in_triangle(best_points[i], eps=1e-3):
                best_points[i] = (V0 + V1 + V2) / 3
    
    return best_points
# EVOLVE-BLOCK-END
