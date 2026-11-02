"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from itertools import combinations

# Vertices of the equilateral triangle with area 1
# Side length s = sqrt(2/sqrt(3))
S = np.sqrt(2.0 / np.sqrt(3.0))
H = S * np.sqrt(3) / 2.0
V0 = np.array([0.0, 0.0])
V1 = np.array([S, 0.0])
V2 = np.array([S / 2.0, H])

# Precompute all triplets of indices 0..N-1
triplets = list(combinations(range(N), 3))
n_triplets = len(triplets)

def _get_points(params):
    """Reshape params (2N,) into (N, 2) points."""
    return params.reshape(N, 2)

def _project_to_triangle(points):
    """
    Project points onto the interior of the triangle defined by V0, V1, V2.
    Uses barycentric coordinates.
    """
    # Vectors
    v0 = V0
    v1 = V1
    v2 = V2
    
    # Precompute denominators
    # d = 2 * Area
    d = 2 * (0.5 * S * H)
    
    projected = np.zeros_like(points)
    
    for i in range(len(points)):
        p = points[i]
        
        # Barycentric coordinates
        # lambda0 = ((y1-y2)x + (x2-x1)y + x1y2 - x2y1) / d
        # lambda1 = ((y2-y0)x + (x0-x2)y + x2y0 - x0y2) / d
        # lambda2 = 1 - lambda0 - lambda1
        
        # Using standard formula for barycentric coords
        # Let A=V0, B=V1, C=V2
        # P = lambda0*A + lambda1*B + lambda2*C
        
        # Calculate using cross products or linear equations
        # We can solve for lambda1, lambda2 directly assuming lambda0 = 1 - lambda1 - lambda2
        # P = (1 - l1 - l2)V0 + l1 V1 + l2 V2
        # P - V0 = l1 (V1 - V0) + l2 (V2 - V0)
        
        v01 = V1 - V0
        v02 = V2 - V0
        pv = p - V0
        
        # Solve [v01, v02] [l1, l2]^T = pv
        # M = [[v01x, v02x], [v01y, v02y]]
        M = np.array([[v01[0], v02[0]],
                      [v01[1], v02[1]]])
        
        try:
            l1, l2 = np.linalg.solve(M, pv)
        except np.linalg.LinAlgError:
            # Fallback to centroid
            l1, l2 = 1/3, 1/3
            
        l0 = 1.0 - l1 - l2
        
        # Clamp to ensure inside (with small margin for numerical stability)
        eps = 1e-6
        l0 = max(l0, eps)
        l1 = max(l1, eps)
        l2 = max(l2, eps)
        
        # Normalize if they sum to > 1 due to clamping (though sum is usually 1)
        # If clamping made sum > 1, normalize
        s = l0 + l1 + l2
        if s > 1.0:
            l0 /= s
            l1 /= s
            l2 /= s
            
        projected[i] = l0 * V0 + l1 * V1 + l2 * V2
        
    return projected

def _min_triangle_area(points):
    """Calculate the minimum area of any triangle formed by 3 points."""
    min_area = np.inf
    for i, j, k in triplets:
        p1 = points[i]
        p2 = points[j]
        p3 = points[k]
        # Area = 0.5 * |cross(p2-p1, p3-p1)|
        area = 0.5 * abs(np.cross(p2 - p1, p3 - p1))
        if area < min_area:
            min_area = area
    return min_area

def _soft_min_area(points, temp=0.01):
    """
    Smooth approximation of the minimum area.
    Using log-sum-exp trick on -areas.
    """
    areas = np.zeros(n_triplets)
    idx = 0
    for i, j, k in triplets:
        p1 = points[i]
        p2 = points[j]
        p3 = points[k]
        areas[idx] = np.cross(p2 - p1, p3 - p1)
        idx += 1
        
    # We want to maximize the min area.
    # Soft min of areas is -soft max of -areas.
    # Let's use a simple sum of exponentials of negative areas.
    # Objective to minimize: -soft_min_area
    # soft_min(A) ≈ -1/temp * log(sum(exp(-A_i/temp)))
    
    # To avoid overflow, find max of -A (which is min of A)
    neg_areas = -areas
    max_neg = np.max(neg_areas)
    exp_sum = np.sum(np.exp(neg_areas - max_neg))
    soft_min = -max_neg - (1.0 / temp) * np.log(exp_sum)
    
    return soft_min

def _objective(params, temp):
    """
    Objective function to minimize.
    We want to maximize the min area.
    So we minimize -min_area.
    We add a penalty for points going out of bounds, but we project first.
    """
    points = _get_points(params)
    # Project to ensure validity
    points = _project_to_triangle(points)
    
    # Calculate soft min area
    min_area = _soft_min_area(points, temp)
    
    return -min_area

def _run_optimization(initial_points, time_limit):
    """Run L-BFGS-B optimization from a given initial state."""
    params = initial_points.flatten()
    
    # Annealing temperature
    temp_start = 0.1
    temp_end = 0.01
    steps = 50
    
    current_params = params
    best_params = params
    best_score = -_min_triangle_area(_project_to_triangle(_get_points(params)))
    
    for i in range(steps):
        # Linear annealing
        temp = temp_start + (temp_end - temp_start) * (i / steps)
        
        res = minimize(_objective, current_params, method='L-BFGS-B', 
                       args=(temp,), options={'maxiter': 20, 'ftol': 1e-8})
        
        current_params = res.x
        
        # Evaluate true min area (not soft) for tracking best
        points = _project_to_triangle(_get_points(current_params))
        score = _min_triangle_area(points)
        
        if score > best_score:
            best_score = score
            best_params = current_params.copy()
            
        if time.time() > time_limit:
            break
            
    return best_params, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Initial random points
    rng = np.random.default_rng(42)
    
    # Generate random points in triangle
    def random_point():
        u = rng.random()
        v = rng.random()
        if u + v > 1.0:
            u = 1.0 - u
            v = 1.0 - v
        # Barycentric to Cartesian
        return V0 + u * (V1 - V0) + v * (V2 - V0)
    
    best_params = None
    best_score = -1.0
    
    # Try several restarts
    n_restarts = 10
    
    for r in range(n_restarts):
        if time.time() > deadline:
            break
            
        # Initialize points
        # Mix of random and near-vertex points to encourage structure
        init_pts = np.array([random_point() for _ in range(N)])
        
        # Run optimization
        params, score = _run_optimization(init_pts, deadline)
        
        if score > best_score:
            best_score = score
            best_params = params
            # If we find a very good solution, we might want to polish it more
            # But for now, just track best.
            
    # Final polish on the best found
    if best_params is not None:
        # One more run with lower temperature for precision
        final_params, final_score = _run_optimization(best_params, deadline)
        if final_score > best_score:
            best_params = final_params
            best_score = final_score

    # Convert best_params to points and project
    final_points = _get_points(best_params)
    final_points = _project_to_triangle(final_points)
    
    return final_points
# EVOLVE-BLOCK-END
