"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _radius(pts):
    """Compute the largest common radius for given centers."""
    n = len(pts)
    if n == 0:
        return 0.0
    # Distance to walls
    wall = 0.5 - np.abs(pts).max(axis=0)
    r_wall = wall.min()
    # Pairwise distances
    # Use a vectorized approach but chunked to avoid memory issues for large N
    # For N=50, a full NxN matrix is small (50x50 = 2500 entries)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=2))
    # Mask diagonal
    np.fill_diagonal(dist, np.inf)
    r_pair = dist.min() / 2.0
    return min(r_wall, r_pair)


def _objective_neg_r(pts):
    """Objective function: negative radius to maximize."""
    r = _radius(pts)
    return -r


def _constraint_pts(pts):
    """Ensure points are within the square [-0.5, 0.5]^2.
    Returns an array of constraint values (>= 0 for feasibility).
    We use 4 constraints: x_min, x_max, y_min, y_max for all points.
    Actually, scipy SLSQP prefers a single function returning an array or scalar.
    Let's use a single function returning the minimum margin to the boundary.
    margin = min(0.5 - |x|, 0.5 - |y|) for all points.
    If margin > 0, all points are strictly inside.
    """
    abs_pts = np.abs(pts)
    margins = 0.5 - abs_pts
    return margins.min()


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # Initial guess: Square grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    pts_init = grid[:N].copy()
    
    best_pts = pts_init.copy()
    best_r = _radius(best_pts)
    
    # Run SLSQP optimization
    # We flatten the points into a 1D array of length 2*N
    x0 = best_pts.flatten()
    
    # Constraints: points must be inside the square
    # g(x) >= 0
    # We define g(x) as the minimum distance to the boundary.
    # If min_dist > 0, feasible.
    # Actually, SLSQP handles 'ineq' constraints where func(x) >= 0.
    
    def constraint_func(x):
        pts = x.reshape(-1, 2)
        abs_pts = np.abs(pts)
        margins = 0.5 - abs_pts
        return margins.min()
    
    cons = ({'type': 'ineq', 'fun': constraint_func})
    
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    try:
        res = minimize(
            _objective_neg_r,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 500, 'ftol': 1e-12, 'disp': False}
        )
        
        if time.time() > deadline:
            return best_pts
        
        if res.success:
            res_pts = res.x.reshape(-1, 2)
            # Clip to be safe
            res_pts = np.clip(res_pts, -0.5, 0.5)
            res_r = _radius(res_pts)
            
            if res_r > best_r:
                best_pts = res_pts
                best_r = res_r
                
    except Exception as e:
        pass
    
    # If we have time left, try a few random restarts or local refinements
    # But SLSQP from grid is usually quite good for this problem.
    # Let's try a second pass with a tighter tolerance or different start if time permits.
    
    if time.time() < deadline - 10:
        # Try starting from a slightly perturbed grid
        np.random.seed(42)
        for i in range(5):
            if time.time() > deadline - 5:
                break
            
            noise = np.random.normal(0, 0.01, (N, 2))
            x0_pert = (best_pts + noise).flatten()
            x0_pert = np.clip(x0_pert, -0.49, 0.49).flatten()
            
            try:
                res2 = minimize(
                    _objective_neg_r,
                    x0_pert,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=cons,
                    options={'maxiter': 200, 'ftol': 1e-12, 'disp': False}
                )
                
                if res2.success:
                    res2_pts = res2.x.reshape(-1, 2)
                    res2_pts = np.clip(res2_pts, -0.5, 0.5)
                    res2_r = _radius(res2_pts)
                    
                    if res2_r > best_r:
                        best_pts = res2_pts
                        best_r = res2_r
                        
            except Exception:
                pass
    
    return best_pts
# EVOLVE-BLOCK-END
