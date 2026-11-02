"""
Heilbronn triangle problem: maximize minimum triangle area for 11 points.
Strategy:
1. Use SLSQP with explicit inequality constraints on all triangle areas.
   This directly handles the non-smooth 'min' by enforcing A_i >= t for all i,
   maximizing t. This avoids the smoothing bias of log-sum-exp.
2. Multi-start with diverse initializations (Greedy, Structured, Random).
3. Time-budgeted iterations.
"""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 11
S3H = np.sqrt(3) / 2
DEADLINE = 115.0  # Safety margin before 150s limit

# Pre-compute triplet indices for vectorized area calculation
_triplets = []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            _triplets.append((i, j, k))
_triplets = np.array(_triplets)
_i_idx = _triplets[:, 0]
_j_idx = _triplets[:, 1]
_k_idx = _triplets[:, 2]

def _min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by three points."""
    p_i = pts[_i_idx]
    p_j = pts[_j_idx]
    p_k = pts[_k_idx]
    
    dx1 = p_j - p_i
    dx2 = p_k - p_i
    
    cross = dx1[:, 0] * dx2[:, 1] - dx1[:, 1] * dx2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def _check_inside(pts, eps=1e-9):
    """Check if all points are inside or on the boundary of the equilateral triangle."""
    for p in pts:
        x, y = p
        if y < -eps:
            return False
        if y > np.sqrt(3) * x + eps:
            return False
        if y > np.sqrt(3) * (1 - x) + eps:
            return False
        if x < -eps or x > 1 + eps:
            return False
    return True

def _project_to_triangle(pts):
    """Project points that are slightly outside the triangle back inside."""
    pts = np.array(pts, dtype=float)
    for i in range(len(pts)):
        x, y = pts[i]
        if y < 0:
            y = 0
        max_y = min(np.sqrt(3) * x, np.sqrt(3) * (1 - x))
        if y > max_y:
            y = max_y
        if x < 0:
            x = 0
        if x > 1:
            x = 1
        pts[i] = [x, y]
    return pts

def _barycentric_to_cartesian(bary):
    """Convert barycentric coordinates to Cartesian coordinates."""
    x = bary[:, 1] + 0.5 * bary[:, 2]
    y = S3H * bary[:, 2]
    return np.column_stack((x, y))

def _cartesian_to_bary(pts):
    """Convert Cartesian points to barycentric coordinates."""
    b3 = pts[:, 1] / S3H
    b2 = pts[:, 0] - 0.5 * b3
    b1 = 1.0 - b2 - b3
    return np.column_stack((b1, b2, b3))

def _random_point_in_triangle(rng):
    """Generate a random point inside the equilateral triangle."""
    u = rng.random()
    v = rng.random()
    while u + v > 1:
        u = rng.random()
        v = rng.random()
    x = u + 0.5 * v
    y = S3H * v
    return [x, y]

def _greedy_init(rng):
    """
    Greedy initialization: place points one by one to maximize the minimum distance to existing points.
    """
    pts = []
    p0 = _random_point_in_triangle(rng)
    pts.append(np.array(p0))
    
    for _ in range(N - 1):
        best_pos = None
        best_min_dist = -1
        for _ in range(500):
            cand = np.array(_random_point_in_triangle(rng))
            dists = np.linalg.norm(pts - cand, axis=1)
            min_d = np.min(dists)
            if min_d > best_min_dist:
                best_min_dist = min_d
                best_pos = cand
        pts.append(best_pos)
    
    return np.array(pts)

def _structured_init():
    """Create a structured initialization based on known Heilbronn-like configurations."""
    pts = [
        [0.0, 0.0], [1.0, 0.0], [0.5, S3H],
        [0.5, 0.0], [0.75, S3H/2], [0.25, S3H/2],
        [0.5, S3H/3],
        [0.25, 0.0], [0.75, 0.0],
        [0.125, S3H/4], [0.875, S3H/4]
    ]
    return np.array(pts)

def construct():
    """
    Construct 11 points in the unit equilateral triangle to maximize min triangle area.
    Uses SLSQP with area constraints to directly optimize the non-smooth objective.
    """
    start_time = time.time()
    rng = random.Random(42)
    
    best_pts = None
    best_score = -1.0
    
    # Multi-start optimization
    initializations = []
    
    # 1. Greedy
    try:
        p_greedy = _greedy_init(rng)
        initializations.append(p_greedy)
    except Exception:
        pass
    
    # 2. Random
    p_rand = np.array([_random_point_in_triangle(rng) for _ in range(N)])
    p_rand = _project_to_triangle(p_rand)
    initializations.append(p_rand)
    
    # 3. Structured
    p_struct = _structured_init()
    p_struct = _project_to_triangle(p_struct)
    initializations.append(p_struct)
    
    # 4. More random starts
    for _ in range(5):
        p_rand2 = np.array([_random_point_in_triangle(rng) for _ in range(N)])
        p_rand2 = _project_to_triangle(p_rand2)
        initializations.append(p_rand2)
    
    # SLSQP Optimization
    # We optimize in Cartesian coordinates (2N variables).
    # Objective: minimize -t (where t is the minimum area)
    # Constraints: Area(i,j,k) >= t for all triplets, and t <= some upper bound.
    # Actually, standard formulation:
    # Maximize t
    # Subject to: 0.5 * |cross(i,j,k)| >= t
    #             Points inside triangle
    
    # To make it work with SLSQP which minimizes:
    # Minimize -t
    # Subject to: t - 0.5 * |cross(i,j,k)| <= 0  (This is non-smooth due to abs)
    
    # Better: Use signed areas? No, area is always positive.
    # The non-smoothness of abs(cross) is a problem for SLSQP gradients.
    # However, we can use the fact that for a valid configuration, the sign of the cross product
    # is consistent for non-degenerate triangles. We can assume a fixed orientation or use 
    # a smooth approximation of abs(x) = sqrt(x^2 + eps).
    # Or, we can use the constraint: cross^2 >= (2*t)^2. This is smooth!
    # cross = (p_j - p_i) x (p_k - p_i)
    # cross^2 - 4*t^2 >= 0
    
    def objective(x):
        # x is 2N + 1 variables: 2N for points, 1 for t
        # Minimize -t
        return -x[-1]
    
    def constraints(x):
        pts = x[:2*N].reshape(N, 2)
        t = x[-1]
        
        p_i = pts[_i_idx]
        p_j = pts[_j_idx]
        p_k = pts[_k_idx]
        
        dx1 = p_j - p_i
        dx2 = p_k - p_i
        
        cross = dx1[:, 0] * dx2[:, 1] - dx1[:, 1] * dx2[:, 0]
        
        # Constraint: cross^2 - 4*t^2 >= 0
        # SLSQP expects g(x) >= 0 for inequality constraints
        return cross**2 - 4 * t**2
    
    # Bounds for points: 0 <= x <= 1, 0 <= y <= S3H
    # And triangle constraints: y <= sqrt(3)*x and y <= sqrt(3)*(1-x)
    # SLSQP doesn't handle complex geometric bounds well directly in bounds.
    # We must add them as constraints.
    
    def triangle_constraints(x):
        pts = x[:2*N].reshape(N, 2)
        c1 = pts[:, 1] # y >= 0
        c2 = np.sqrt(3) * pts[:, 0] - pts[:, 1] # y <= sqrt(3)*x
        c3 = np.sqrt(3) * (1 - pts[:, 0]) - pts[:, 1] # y <= sqrt(3)*(1-x)
        c4 = pts[:, 0] # x >= 0
        c5 = 1 - pts[:, 0] # x <= 1
        # Combine into one vector
        return np.concatenate([c1, c2, c3, c4, c5])
    
    bounds = [(0, 1)] * (2*N)
    # t can be between 0 and 0.5 (max area of triangle is 0.5*sqrt(3)/2 ~ 0.433, but let's be safe)
    bounds.append((0, 0.5))
    
    cons = [
        {'type': 'ineq', 'fun': constraints},
        {'type': 'ineq', 'fun': triangle_constraints}
    ]
    
    for s, init_pts in enumerate(initializations):
        if time.time() - start_time > DEADLINE - 10:
            break
            
        # Initialize t with current min area
        t0 = _min_triangle_area(init_pts)
        x0 = np.concatenate([init_pts.flatten(), [t0]])
        
        try:
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 100, 'ftol': 1e-10}
            )
            
            if result.success:
                pts_opt = result.x[:2*N].reshape(N, 2)
                pts_opt = _project_to_triangle(pts_opt)
                
                if _check_inside(pts_opt):
                    score = _min_triangle_area(pts_opt)
                    if score > best_score:
                        best_score = score
                        best_pts = pts_opt.copy()
                        
        except Exception as e:
            pass
            
    # If no optimization succeeded, fall back to the best initial guess
    if best_pts is None:
        for init_pts in initializations:
            score = _min_triangle_area(init_pts)
            if score > best_score:
                best_score = score
                best_pts = init_pts.copy()
                
    if best_pts is None:
        best_pts = _project_to_triangle(np.array([_random_point_in_triangle(rng) for _ in range(N)]))
        
    # Final check
    if not _check_inside(best_pts):
        best_pts = _project_to_triangle(best_pts)
        
    return best_pts

if __name__ == "__main__":
    pts = construct()
    print(f"Score: {_min_triangle_area(pts)}")
