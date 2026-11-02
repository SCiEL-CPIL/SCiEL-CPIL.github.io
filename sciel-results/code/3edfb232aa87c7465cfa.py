"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from itertools import combinations

# The equilateral triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2.0])
AREA_EQ = np.sqrt(3) / 4.0

# Precompute all triples of indices
TRIPLES = list(combinations(range(N), 3))
N_TRIPLES = len(TRIPLES)

def _area_vec(pts):
    """
    pts: shape (N, 2)
    returns: array of areas for all triples (N_TRIPLES,)
    Area = 0.5 * |cross(p_j - p_i, p_k - p_i)|
    """
    areas = np.empty(N_TRIPLES)
    for idx, (i, j, k) in enumerate(TRIPLES):
        v1 = pts[j] - pts[i]
        v2 = pts[k] - pts[i]
        areas[idx] = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
    return areas

def _min_area(pts):
    return np.min(_area_vec(pts))

def _inside_triangle(pts):
    """Check if all points are inside or on the boundary of the equilateral triangle."""
    # Barycentric coordinates relative to V0, V1, V2
    # For a point P, barycentric coords (a, b, c) satisfy P = a*V0 + b*V1 + c*V2, a+b+c=1, a,b,c >= 0
    # We can compute these via linear algebra.
    # Matrix M = [V1-V0, V2-V0] (2x2)
    # [b, c]^T = M^{-1} (P - V0)
    # a = 1 - b - c
    M = np.array([V1 - V0, V2 - V0]).T  # 2x2
    Minv = np.linalg.inv(M)
    for p in pts:
        diff = p - V0
        bc = Minv @ diff
        b, c = bc[0], bc[1]
        a = 1.0 - b - c
        if a < -1e-9 or b < -1e-9 or c < -1e-9:
            return False
    return True

def _objective(x):
    """
    x: flattened array of shape (2*N + 1,)
    First 2*N elements: coordinates of N points
    Last element: t (the minimum area we want to maximize)
    We want to minimize -t.
    """
    t = x[-1]
    return -t

def _constraints(x):
    """
    Constraints:
    1. All points must be inside the triangle.
       This is enforced by projecting or using penalty? 
       SLSQP handles inequality constraints g(x) >= 0.
       For each point, we need 3 barycentric coords >= 0.
       Total 3*N constraints.
    2. Each triangle area >= t.
       Total N_TRIPLES constraints.
    
    We will return a list of constraint dicts.
    """
    pts = x[:2*N].reshape(N, 2)
    t = x[-1]
    
    # Compute areas
    areas = _area_vec(pts)
    
    cons = []
    
    # Constraint: areas - t >= 0
    cons.append({'type': 'ineq', 'fun': lambda x, areas=areas, t=t: areas - t})
    
    # Constraints: barycentric coordinates >= 0
    M = np.array([V1 - V0, V2 - V0]).T
    Minv = np.linalg.inv(M)
    
    for i in range(N):
        p = pts[i]
        diff = p - V0
        bc = Minv @ diff
        b, c = bc[0], bc[1]
        a = 1.0 - b - c
        cons.append({'type': 'ineq', 'fun': lambda x, a=a: a})
        cons.append({'type': 'ineq', 'fun': lambda x, b=b: b})
        cons.append({'type': 'ineq', 'fun': lambda x, c=c: c})
        
    return cons

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_pts = None
    best_score = -1.0
    
    # Initial guess: random points inside triangle
    rng = np.random.RandomState(42)
    
    # We will run a few restarts with different initial guesses
    # and different t initializations
    
    # Initial t guess: small value
    t0 = 0.01
    
    # Initial points: random
    def random_points_in_triangle(rng):
        pts = np.zeros((N, 2))
        for i in range(N):
            u = rng.random()
            v = rng.random()
            while u + v > 1.0:
                u = rng.random()
                v = rng.random()
            # Barycentric to Cartesian
            p = u * V1 + v * V2  # Since V0 is origin, P = u*V1 + v*V2
            pts[i] = p
        return pts
    
    # Try a few restarts
    for restart in range(5):
        if time.time() > deadline:
            break
            
        pts0 = random_points_in_triangle(rng)
        
        # Initial x: points + t
        x0 = np.concatenate([pts0.flatten(), [t0]])
        
        # Bounds for x
        # Points must be within the bounding box of the triangle
        # x in [0, 1], y in [0, sqrt(3)/2]
        # t can be in [0, 1] (area is at most 1)
        bounds = []
        for i in range(N):
            bounds.append((0.0, 1.0))  # x_i
            bounds.append((0.0, np.sqrt(3)/2.0))  # y_i
        bounds.append((0.0, 1.0))  # t
        
        try:
            res = minimize(
                _objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=_constraints(x0),
                options={'maxiter': 500, 'ftol': 1e-8}
            )
        except Exception as e:
            continue
            
        if res.success or res.status == 0:
            x_opt = res.x
            pts_opt = x_opt[:2*N].reshape(N, 2)
            
            # Verify constraints
            if _inside_triangle(pts_opt):
                score = _min_area(pts_opt)
                if score > best_score:
                    best_score = score
                    best_pts = pts_opt
        else:
            # Even if not successful, check the result
            x_opt = res.x
            pts_opt = x_opt[:2*N].reshape(N, 2)
            if _inside_triangle(pts_opt):
                score = _min_area(pts_opt)
                if score > best_score:
                    best_score = score
                    best_pts = pts_opt
    
    # If no valid result from SLSQP, fallback to random
    if best_pts is None:
        # Use the seed program's logic as fallback
        import random
        rng_seed = random.Random(7)
        points = []
        while len(points) < N:
            u, v = rng_seed.random(), rng_seed.random()
            if u + v < 0.98 and u > 0.01 and v > 0.01:
                points.append([u + 0.5 * v, (3 ** 0.5) / 2 * v])
        best_pts = np.array(points)
        best_score = _min_area(best_pts)
    
    return best_pts
# EVOLVE-BLOCK-END
