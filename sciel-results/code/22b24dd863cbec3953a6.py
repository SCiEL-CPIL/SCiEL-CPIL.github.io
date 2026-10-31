"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def min_triangle_area(points):
    """Compute the minimum area of any triangle formed by 3 points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                area = 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))
                if area < min_area:
                    min_area = area
    return min_area

def objective_flat(x):
    """Negative of min triangle area with penalty for out-of-bounds points."""
    points = x.reshape(-1, 2)
    
    # Penalty for points outside the triangle
    penalty = 0.0
    for p in points:
        if not is_inside_triangle(p, eps=1e-6):
            # Compute distance to nearest edge
            penalty += 10.0 * _distance_outside_triangle(p)
    
    return -min_triangle_area(points) + penalty

def _distance_outside_triangle(p):
    """Approximate distance of point p outside the triangle."""
    A = np.array([0.0, 0.0])
    B = np.array([1.0, 0.0])
    C = np.array([0.5, np.sqrt(3)/2])
    
    min_dist = float('inf')
    for V1, V2 in [(A,B), (B,C), (C,A)]:
        # Distance from p to line segment V1-V2
        d = _point_to_segment_dist(p, V1, V2)
        if d < min_dist:
            min_dist = d
    
    # Check if inside
    if is_inside_triangle(p, eps=1e-9):
        return 0.0
    return min_dist

def _point_to_segment_dist(p, a, b):
    """Distance from point p to segment ab."""
    ab = b - a
    ap = p - a
    t = np.dot(ap, ab) / np.dot(ab, ab)
    t = max(0, min(1, t))
    closest = a + t * ab
    return np.linalg.norm(p - closest)

def is_inside_triangle(p, eps=1e-9):
    """Check if point p is inside or on the boundary of the reference triangle."""
    A = np.array([0.0, 0.0])
    B = np.array([1.0, 0.0])
    C = np.array([0.5, np.sqrt(3)/2])
    
    v0 = B - A
    v1 = C - A
    v2 = p - A
    
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d02 = np.dot(v0, v2)
    d11 = np.dot(v1, v1)
    d12 = np.dot(v1, v2)
    
    denom = d00 * d11 - d01 * d01
    if abs(denom) < eps:
        return False
    
    u = (d11 * d02 - d01 * d12) / denom
    v = (d00 * d12 - d01 * d02) / denom
    
    return u >= -eps and v >= -eps and (u + v) <= 1.0 + eps

def random_points_in_triangle(rng, n):
    """Generate n random points uniformly in the triangle."""
    points = []
    for _ in range(n):
        while True:
            u = rng.random()
            v = rng.random()
            if u + v <= 1.0:
                x = u + 0.5 * v
                y = (np.sqrt(3) / 2) * v
                points.append([x, y])
                break
    return np.array(points)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initial random points as fallback
    init_points = random_points_in_triangle(np.random.RandomState(7), N)
    best_points = init_points.copy()
    best_score = min_triangle_area(init_points)
    
    # Multi-start local optimization
    n_restarts = 30
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Generate random initial point
        x0 = random_points_in_triangle(np.random.RandomState(1000 + restart), N).flatten()
        
        # Try Nelder-Mead optimization
        try:
            result = minimize(
                objective_flat,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True}
            )
            
            candidate = result.x.reshape(-1, 2)
            # Verify all points are inside the triangle
            valid = all(is_inside_triangle(p, eps=1e-5) for p in candidate)
            if valid:
                score = min_triangle_area(candidate)
                if score > best_score:
                    best_score = score
                    best_points = candidate.copy()
        except Exception:
            continue
    
    return best_points
# EVOLVE-BLOCK-END
