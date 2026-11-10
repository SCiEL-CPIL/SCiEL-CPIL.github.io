# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time


def _compute_score(points):
    """Compute R = (d_min/d_max)^2 for a set of points."""
    d = pdist(points)
    d_min = np.min(d)
    d_max = np.max(d)
    if d_max == 0:
        return 0.0
    return (d_min / d_max) ** 2


def _reconstruct_points(free_params):
    """
    Reconstruct 16 points from 28 free parameters.
    Points 0 and 1 are fixed at (0,0) and (1,0).
    """
    n = 16
    points = np.zeros((n, 2))
    points[0] = [0.0, 0.0]
    points[1] = [1.0, 0.0]
    points[2:] = free_params.reshape((n - 2, 2))
    return points


def _slsqp_objective_and_constraints(free_params, n=16):
    """
    Objective and constraints for SLSQP.
    We maximize t (min distance) subject to all distances <= 1 and >= t.
    Variables: free_params (28 values) + t (1 value) -> total 29 variables.
    """
    # Unpack t and free params
    t = free_params[0]
    x_params = free_params[1:]
    
    points = _reconstruct_points(x_params)
    
    # Objective: Minimize -t
    obj = -t
    
    # Constraints:
    # 1. dist(i,j) >= t  => dist(i,j) - t >= 0
    # 2. dist(i,j) <= 1  => 1 - dist(i,j) >= 0
    
    # Compute all pairwise distances
    d = pdist(points)
    
    constraints = []
    # Add min distance constraints
    # d - t >= 0
    # We can add these as a single vector constraint or individual.
    # SLSQP accepts a list of dicts.
    # For efficiency, we can compute the gradients if we do it manually, but SLSQP approximates.
    
    # Let's create a function that returns the constraint values.
    # We need to be careful with the number of constraints (120 * 2 = 240).
    # This is manageable for SLSQP.
    
    # Constraint 1: d - t >= 0
    c_min = d - t
    
    # Constraint 2: 1 - d >= 0
    c_max = 1.0 - d
    
    # Combine into a single array for 'ineq' constraint
    # SLSQP expects 'fun' to return an array of constraint values (all >= 0)
    all_constraints = np.concatenate([c_min, c_max])
    
    return obj, all_constraints


def _minimize_slsqp(x0, n=16):
    """
    Run SLSQP optimization.
    x0: initial guess for [t, x2, y2, ..., x15, y15]
    """
    # Initial t should be small to be feasible
    # We can estimate a safe t based on initial distances
    
    def fun(x):
        obj, _ = _slsqp_objective_and_constraints(x, n)
        return obj
    
    def con(x):
        _, cons = _slsqp_objective_and_constraints(x, n)
        return cons

    # Bounds for t: [0, 1] (since dist(0,1)=1, t cannot exceed 1)
    # Bounds for points: Let's use a generous box to allow exploration
    # If points go too far, dist > 1 constraint will push them back.
    # But unbounded variables can cause numerical issues.
    # Let's bound them to [-2, 3] or similar.
    
    bounds = []
    bounds.append((0.0, 1.0)) # t
    for i in range(14):
        bounds.append((-2.0, 3.0)) # x_i
        bounds.append((-2.0, 3.0)) # y_i
        
    res = minimize(
        fun, 
        x0, 
        method='SLSQP', 
        bounds=bounds, 
        constraints=[{'type': 'ineq', 'fun': con}],
        options={'maxiter': 1000, 'ftol': 1e-8}
    )
    
    return res


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    n = 16
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 630s limit

    best_points = None
    best_score = 0.0

    def update_best(points, score):
        nonlocal best_points, best_score
        if score > best_score:
            best_score = score
            best_points = points.copy()

    # Generate initial guesses
    # We need to provide [t, x2, y2, ..., x15, y15]
    # t should be small, e.g., 0.1
    
    initial_guesses = []
    
    # 1. Random points in a small region
    rng = np.random.default_rng(42)
    for i in range(10):
        # Generate 14 random points
        pts = rng.uniform(0.5, 1.5, size=(14, 2))
        # Calculate min distance to set a safe initial t
        full_pts = np.vstack([np.array([[0,0], [1,0]]), pts])
        d = pdist(full_pts)
        t_init = 0.5 * np.min(d)
        if t_init < 0.01:
            t_init = 0.01
        if t_init > 0.9:
            t_init = 0.9
        x0 = np.concatenate([[t_init], pts.flatten()])
        initial_guesses.append(x0)
        
    # 2. Grid-like start
    # 4x4 grid
    grid_pts = np.array([[i, j] for i in range(4) for j in range(4)], dtype=float)
    # Scale to fit in roughly unit box? 
    # We need to map grid[0] to (0,0) and grid[1] to (1,0)?
    # Grid[0]=(0,0), Grid[1]=(1,0) is already correct if we use standard grid.
    # But other points are at (2,0), (3,0) etc.
    # We will let SLSQP fix the scale.
    # Initial t: min dist in grid is 1. But max dist is sqrt(18) ~ 4.2.
    # Constraint dist <= 1 will be violated.
    # We need to scale the grid down so max dist <= 1.
    # Max dist in 4x4 grid is between (0,0) and (3,3) -> sqrt(18).
    # Scale factor = 1 / sqrt(18).
    grid_pts_scaled = grid_pts / np.sqrt(18)
    # Now grid[0] is (0,0). grid[1] is (1/sqrt(18), 0).
    # We need grid[1] to be (1,0).
    # So we need to scale x differently? Or just use a different starting config.
    
    # Let's just use a circle of points
    angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
    circle_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    # Shift so min is at 0?
    # We need P0=(0,0), P1=(1,0).
    # This is hard to map directly.
    
    # Let's stick to random starts, they are robust.
    # Add a few more random starts with different seeds
    for seed in [1, 2, 3, 4, 5]:
        rng = np.random.default_rng(seed)
        pts = rng.uniform(-0.5, 1.5, size=(14, 2))
        full_pts = np.vstack([np.array([[0,0], [1,0]]), pts])
        d = pdist(full_pts)
        t_init = 0.5 * np.min(d)
        if t_init < 0.01: t_init = 0.01
        if t_init > 0.9: t_init = 0.9
        x0 = np.concatenate([[t_init], pts.flatten()])
        initial_guesses.append(x0)

    # Run SLSQP for each initial guess
    for i, x0 in enumerate(initial_guesses):
        if time.time() > deadline - 20:
            break
            
        try:
            res = _minimize_slsqp(x0)
            if res.success:
                t_opt = res.x[0]
                pts_opt = _reconstruct_points(res.x[1:])
                score = _compute_score(pts_opt)
                update_best(pts_opt, score)
        except Exception:
            continue

    # Fallback
    if best_points is None:
        # Return a simple grid scaled to unit max distance
        grid = np.array([[i, j] for i in range(4) for j in range(4)], dtype=float)
        d = pdist(grid)
        max_d = np.max(d)
        grid = grid / max_d
        # Force P0=(0,0), P1=(1,0) is not guaranteed by scaling.
        # Just return the scaled grid. The evaluator will calculate score.
        # But we need shape (16,2).
        best_points = grid

    return best_points


# EVOLVE-BLOCK-END
