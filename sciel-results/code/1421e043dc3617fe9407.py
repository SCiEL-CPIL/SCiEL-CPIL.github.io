import time
import numpy as np
from scipy.optimize import minimize
from itertools import combinations

N = 11
H = np.sqrt(3) / 2.0

# Precompute triangle indices for vectorized area calculation
_combos = list(combinations(range(N), 3))
_idx_i = np.array([c[0] for c in _combos])
_idx_j = np.array([c[1] for c in _combos])
_idx_k = np.array([c[2] for c in _combos])

def _min_triangle_area(points):
    """Compute minimum area of all C(11,3) triangles."""
    pi = points[_idx_i]
    pj = points[_idx_j]
    pk = points[_idx_k]
    v1 = pj - pi
    v2 = pk - pi
    areas = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    return np.min(areas)

def _is_inside_triangle(p):
    """Check if point p (x,y) is inside the equilateral triangle."""
    x, y = p
    if y < 0 or y > H:
        return False
    if x < 0 or x > 1:
        return False
    # Check against the slanted sides
    # Left side: y = sqrt(3) * x  => x = y / sqrt(3)
    # Right side: y = -sqrt(3) * (x - 1) => x = 1 - y / sqrt(3)
    if x < y / np.sqrt(3) - 1e-9:
        return False
    if x > 1 - y / np.sqrt(3) + 1e-9:
        return False
    return True

def _project_to_triangle(p):
    """Project point p to the closest point inside the triangle."""
    x, y = p
    # Simple projection: if outside, move towards centroid (0.5, H/3)
    # This is a heuristic, but sufficient for initialization or penalty relaxation
    # A better projection would be onto the edges.
    # For simplicity in the penalty function, we just use a soft penalty.
    # But for the final result, we must ensure validity.
    # Let's implement a proper projection onto the convex hull.
    # Vertices: A(0,0), B(1,0), C(0.5, H)
    A = np.array([0, 0])
    B = np.array([1, 0])
    C = np.array([0.5, H])
    
    # Check if inside
    if _is_inside_triangle(p):
        return p
    
    # Find closest point on edges AB, BC, CA
    def closest_point_on_segment(p, a, b):
        ab = b - a
        ap = p - a
        t = np.dot(ap, ab) / np.dot(ab, ab)
        t = np.clip(t, 0, 1)
        return a + t * ab

    p_ab = closest_point_on_segment(p, A, B)
    p_bc = closest_point_on_segment(p, B, C)
    p_ca = closest_point_on_segment(p, C, A)
    
    d_ab = np.linalg.norm(p - p_ab)
    d_bc = np.linalg.norm(p - p_bc)
    d_ca = np.linalg.norm(p - p_ca)
    
    if d_ab == min(d_ab, d_bc, d_ca):
        return p_ab
    elif d_bc == min(d_ab, d_bc, d_ca):
        return p_bc
    else:
        return p_ca

def _objective(x):
    """Objective function for minimization. We minimize negative min area."""
    points = x.reshape(N, 2)
    
    # Penalty for being outside the triangle
    penalty = 0.0
    for i in range(N):
        p = points[i]
        # Check bounds
        if not _is_inside_triangle(p):
            # Calculate distance to boundary for penalty
            # Using projection to find distance
            proj = _project_to_triangle(p)
            dist = np.linalg.norm(p - proj)
            penalty += dist * dist
    
    if penalty > 1e-3:
        # Return large value if significantly outside
        # The min area will be 0 or negative if points are weird, but we want to push them inside
        # Actually, if points are outside, the area calculation might be valid geometrically 
        # but the constraint is violated.
        # We return a large cost.
        return 1000.0 + penalty
    
    min_area = _min_triangle_area(points)
    return -min_area

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s buffer
    
    best_score = -np.inf
    best_points = None
    
    # Initial guess: Random points inside triangle
    # Use rejection sampling for initial points to ensure they are valid
    def random_point():
        while True:
            u, v = np.random.random(), np.random.random()
            if u + v < 1.0:
                x = u + 0.5 * v
                y = H * v
                return np.array([x, y])
    
    rng = np.random.default_rng(42)
    
    # Try multiple restarts
    num_restarts = 20
    
    for i in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Generate initial points
        init_points = np.array([random_point() for _ in range(N)])
        # Ensure they are distinct enough? Not strictly necessary, optimizer will move them.
        
        x0 = init_points.flatten()
        
        # Bounds for L-BFGS-B or Powell
        # x in [0, 1], y in [0, H]
        bounds = [(0, 1), (0, H)] * N
        
        try:
            # Use Powell method as it is derivative-free and robust to non-smoothness
            res = minimize(_objective, x0, method='Powell', bounds=bounds, 
                           options={'maxiter': 2000, 'maxfev': 2000})
            
            if res.success or res.fun < best_score:
                # Check validity of the final result
                final_points = res.x.reshape(N, 2)
                valid = True
                for p in final_points:
                    if not _is_inside_triangle(p):
                        valid = False
                        break
                
                if valid:
                    score = _min_triangle_area(final_points)
                    if score > best_score:
                        best_score = score
                        best_points = final_points
        except Exception:
            pass

    # If best_points is None, return a valid random configuration
    if best_points is None:
        best_points = np.array([random_point() for _ in range(N)])
        
    return best_points

# EVOLVE-BLOCK-END
