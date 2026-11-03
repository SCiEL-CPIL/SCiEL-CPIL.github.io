"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize


def _min_triangle_area(points):
    """
    Calculate the minimum area of any triangle formed by the points.
    points: (N, 2) array of coordinates.
    Returns: min area (float).
    """
    n = len(points)
    min_area = float('inf')
    # Use vectorization for speed: compute all cross products
    # We need min over i<j<k of 0.5 * |cross(P_j-P_i, P_k-P_i)|
    
    # To save time, we can compute all pairwise differences and then cross products
    # But N=11 is small, so a loop over pairs (i,j) and vectorized k is fast.
    for i in range(n):
        pi = points[i]
        for j in range(i + 1, n):
            pj = points[j]
            d_ij = pj - pi
            # Cross product with all other points
            # cross = d_ij[0]*(y_k - y_i) - d_ij[1]*(x_k - x_i)
            diffs = points - pi
            crosses = d_ij[0] * diffs[:, 1] - d_ij[1] * diffs[:, 0]
            areas = 0.5 * np.abs(crosses)
            # Exclude i and j
            areas[i] = float('inf')
            areas[j] = float('inf')
            current_min = np.min(areas)
            if current_min < min_area:
                min_area = current_min
    return min_area


def _penalty(points):
    """
    Calculate penalty for points outside the triangle.
    Triangle vertices: (0,0), (1,0), (0.5, sqrt(3)/2)
    """
    # Check if point is inside
    # y >= 0
    # y <= sqrt(3)*x (left edge)
    # y <= sqrt(3)*(1-x) (right edge)
    
    x = points[:, 0]
    y = points[:, 1]
    
    # Penalty for y < 0
    pen_y_neg = np.sum(np.clip(-y, 0, None))
    
    # Penalty for y > sqrt(3)*x
    # If x < 0.5, max y is sqrt(3)*x
    # If x >= 0.5, max y is sqrt(3)*(1-x)
    # Actually, the constraint is y <= min(sqrt(3)*x, sqrt(3)*(1-x))
    max_y = np.minimum(np.sqrt(3)*x, np.sqrt(3)*(1-x))
    pen_y_high = np.sum(np.clip(y - max_y, 0, None))
    
    # Penalty for x < 0 or x > 1
    pen_x = np.sum(np.clip(-x, 0, None)) + np.sum(np.clip(x - 1, 0, None))
    
    return pen_y_neg + pen_y_high + pen_x


def _objective_flat(flat_points):
    """
    Objective function for differential evolution.
    Input: flat array of length 2*N.
    Returns: negative min area (to maximize) + penalty.
    """
    points = flat_points.reshape(N, 2)
    
    # If points are far outside, return a large penalty
    # This helps DE stay in the region
    penalty = _penalty(points)
    
    if penalty > 1e-6:
        return penalty * 1000.0
    
    min_area = _min_triangle_area(points)
    return -min_area


def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    # Define bounds for differential evolution
    # x in [0, 1], y in [0, sqrt(3)/2]
    bounds = [(0, 1), (0, np.sqrt(3)/2)] * N
    
    # Initial best guess
    best_points = None
    best_score = -float('inf')
    
    # Run Differential Evolution
    # maxiter and popsize are tuned for time limit
    # N=11, 22 variables.
    # DE is stochastic, so we might run it multiple times if time permits,
    # or just once with high maxiter.
    
    try:
        result = differential_evolution(
            _objective_flat,
            bounds,
            maxiter=1000,
            popsize=15,
            tol=1e-8,
            seed=42,
            polish=False,  # We will do our own polishing
            callback=lambda x, convergence: None  # No callback to save overhead
        )
        
        if result.success or result.fun < 0:
            best_points = result.x.reshape(N, 2)
            best_score = -result.fun
            
            # Check time
            if time.time() < deadline - 10:
                # Polish with Nelder-Mead
                # We need to be careful with constraints in Nelder-Mead
                # We can use the penalty method
                
                def _nm_objective(flat_points):
                    pts = flat_points.reshape(N, 2)
                    pen = _penalty(pts)
                    if pen > 1e-8:
                        return pen * 100.0
                    return -_min_triangle_area(pts)
                
                # Use multiple restarts for NM if time permits
                nm_result = minimize(
                    _nm_objective,
                    best_points.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10}
                )
                
                if nm_result.fun < best_score:
                    best_points = nm_result.x.reshape(N, 2)
                    best_score = -nm_result.fun
                    
    except Exception as e:
        pass # Fallback to initial if DE fails
    
    # If DE didn't run or failed, use a simple grid or random init
    if best_points is None:
        # Fallback: random points inside triangle
        rng = np.random.default_rng(42)
        # Barycentric random
        bary = rng.random((N, 3))
        bary /= bary.sum(axis=1, keepdims=True)
        # Convert to cartesian
        x = bary[:, 1] + 0.5 * bary[:, 2]
        y = np.sqrt(3)/2 * bary[:, 2]
        best_points = np.column_stack((x, y))
        
        # Try a few random restarts with NM if time permits
        for _ in range(10):
            if time.time() > deadline - 5:
                break
            bary = rng.random((N, 3))
            bary /= bary.sum(axis=1, keepdims=True)
            x = bary[:, 1] + 0.5 * bary[:, 2]
            y = np.sqrt(3)/2 * bary[:, 2]
            pts = np.column_stack((x, y))
            
            def _nm_objective(flat_points, pts_init=pts):
                pts = flat_points.reshape(N, 2)
                pen = _penalty(pts)
                if pen > 1e-8:
                    return pen * 100.0
                return -_min_triangle_area(pts)
                
            nm_res = minimize(_nm_objective, pts.flatten(), method='Nelder-Mead', options={'maxiter': 1000})
            if -nm_res.fun > best_score:
                best_score = -nm_res.fun
                best_points = nm_res.x.reshape(N, 2)
                
    # Final safety check: ensure points are inside
    # Project any points slightly outside back in
    for i in range(N):
        x, y = best_points[i]
        if y < 0: y = 0
        if y > np.sqrt(3)/2: y = np.sqrt(3)/2
        if x < 0: x = 0
        if x > 1: x = 1
        
        # Left edge constraint: y <= sqrt(3)*x
        if y > np.sqrt(3)*x:
            # Project to line y = sqrt(3)*x
            # Closest point on line to (x,y)
            # Line: sqrt(3)*x - y = 0
            # d = (sqrt(3)*x - y) / (3 + 1) = (sqrt(3)*x - y) / 4
            # x_new = x + sqrt(3)*d
            # y_new = y - d
            d = (np.sqrt(3)*x - y) / 4.0
            x = x + np.sqrt(3)*d
            y = y - d
            
        # Right edge constraint: y <= sqrt(3)*(1-x) => y <= sqrt(3) - sqrt(3)*x
        # Line: sqrt(3)*x + y - sqrt(3) = 0
        if y > np.sqrt(3)*(1-x):
            d = (np.sqrt(3)*x + y - np.sqrt(3)) / 4.0
            x = x + np.sqrt(3)*d
            y = y - d
            
        best_points[i] = [x, y]
        
    return best_points
# EVOLVE-BLOCK-END
