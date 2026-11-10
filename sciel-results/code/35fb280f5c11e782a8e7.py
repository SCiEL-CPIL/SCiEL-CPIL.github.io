# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
import time
from scipy.optimize import minimize
from scipy.optimize import differential_evolution
import itertools

def _compute_score(points):
    """Compute R = (d_min/d_max)^2 for a set of points."""
    if points is None or points.shape != (16, 2):
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    d = pdist(points)
    if len(d) == 0:
        return 0.0
    d_min = np.min(d)
    d_max = np.max(d)
    if d_max == 0:
        return 0.0
    if d_min == 0:
        return 0.0
    return (d_min / d_max) ** 2

def _sqp_maximize_min_dist(start_points, max_iter=500, tol=1e-8):
    """
    Use SLSQP to maximize the minimum pairwise distance,
    subject to the constraint that the maximum pairwise distance is 1.
    This effectively searches for the best packing ratio.
    
    Returns:
        points: optimized points
        min_dist: the achieved minimum distance
        max_dist: the achieved maximum distance (should be ~1)
        success: boolean
    """
    n = 16
    x0 = start_points.ravel()
    
    # Constraints:
    # 1. max_dist <= 1 (inequality)
    # 2. To keep the problem bounded and prevent points from flying apart 
    #    while maintaining max_dist=1, we don't strictly need bounds if max_dist is fixed,
    #    but SLSQP handles bounded problems better.
    #    However, since we scale to max_dist=1, the points are effectively confined.
    
    def objective(x):
        pts = x.reshape(n, 2)
        d = pdist(pts)
        if len(d) == 0:
            return 0.0
        # We want to maximize min_dist. 
        # SLSQP minimizes. So we return -min_dist.
        # To make it smooth, we could use a soft-min, but min is often fine for SLSQP 
        # if we provide gradients or if it's robust enough. 
        # Actually, min is not differentiable. 
        # Better objective: Minimize the sum of squared "violations" of a lower bound?
        # No, let's stick to maximizing min_dist. 
        # A common trick: minimize -log(sum(exp(-d_i / T)))? No.
        # Let's just use -min(d). SLSQP can handle non-smooth functions to some extent,
        # but it's risky. 
        # Alternative: Minimize (max_d - 1)^2 + lambda * ( - min_d )? 
        # Let's try a different formulation:
        # Maximize r such that all d_ij >= r and all d_ij <= 1.
        # This is a feasibility problem. 
        # We can use a penalty method or just minimize -min_dist with constraints max_dist <= 1.
        
        return -np.min(d)
    
    def constraint_max_dist(x):
        pts = x.reshape(n, 2)
        d = pdist(pts)
        return 1.0 - np.max(d)
        
    # We also need to prevent points from collapsing? 
    # If min_dist becomes 0, score is 0. 
    # The objective pushes min_dist up.
    
    constraints = [
        {'type': 'ineq', 'fun': constraint_max_dist}
    ]
    
    # Bounds: Since max_dist <= 1, all points must lie within a circle of diameter 1? 
    # No, the diameter of the set is 1. So all points are within distance 1 of each other.
    # This implies they are all within a bounded region. 
    # We can set loose bounds, e.g., [-1, 1] for each coordinate, 
    # but the constraint max_dist <= 1 is tighter.
    # Let's provide bounds to help SLSQP.
    bounds = [(-1.0, 1.0)] * (2 * n)
    
    try:
        res = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                       constraints=constraints, options={'maxiter': max_iter, 'ftol': tol})
        pts = res.x.reshape(n, 2)
        d = pdist(pts)
        min_d = np.min(d)
        max_d = np.max(d)
        return pts, min_d, max_d, res.success
    except Exception:
        return start_points, 0.0, 1.0, False

def _bisection_solve(start_points, time_budget):
    """
    Perform a bisection search on the minimum distance ratio r = min_dist / max_dist.
    Since we can scale the points, we fix max_dist = 1 and maximize min_dist.
    The optimal value is r*. We search for the maximum feasible min_dist.
    
    Strategy:
    1. Start with a good initial guess (e.g., from a grid or random).
    2. Use SQP to maximize min_dist subject to max_dist <= 1.
    3. If SQP fails to improve or gets stuck, try perturbing the start.
    4. Bisection isn't strictly necessary if SQP is good, but we can use a 
       "feasibility check" approach: 
       - Can we achieve min_dist >= r? 
       - If yes, increase r. If no, decrease r.
    
    However, SQP directly maximizing min_dist is more direct.
    Let's combine: Use SQP to find a local optimum for min_dist.
    Then, to escape local optima, we can use a "simulated annealing" like 
    perturbation or just multiple restarts.
    
    For this implementation, we will:
    1. Generate several diverse initial points.
    2. Run SQP from each.
    3. Keep the best result.
    4. If time permits, refine the best result further.
    """
    start_time = time.time()
    best_points = None
    best_min_dist = 0.0
    best_max_dist = 1.0
    best_score = 0.0
    
    def _update(pts, min_d, max_d):
        nonlocal best_points, best_min_dist, best_max_dist, best_score
        if max_d == 0:
            return
        # Scale points so max_dist = 1 for consistent comparison?
        # Actually, the ratio is scale invariant. 
        # If we maximize min_dist subject to max_dist <= 1, the optimal min_dist is r*.
        # The score is (min_dist / max_dist)^2. 
        # If we enforce max_dist = 1, score = min_dist^2.
        # So maximizing min_dist is equivalent to maximizing score.
        
        ratio = min_d / max_d
        score = ratio ** 2
        if score > best_score:
            best_score = score
            best_min_dist = min_d
            best_max_dist = max_d
            best_points = pts.copy()
    
    # Generate initial points
    # 1. Square grid
    x = np.linspace(-1, 1, 4)
    y = np.linspace(-1, 1, 4)
    xx, yy = np.meshgrid(x, y)
    grid_points = np.column_stack([xx.ravel(), yy.ravel()])
    
    # 2. Hexagonal-like grid
    hex_points = []
    for i in range(4):
        for j in range(4):
            px = i * 1.0 + (0.5 if j % 2 else 0)
            py = j * 0.866
            hex_points.append([px, py])
    hex_points = np.array(hex_points)
    
    # 3. Random points
    rng = np.random.default_rng(42)
    rand_points = rng.uniform(-1, 1, size=(16, 2))
    
    # 4. Perturbed grid
    pert_grid = grid_points + rng.normal(0, 0.1, size=(16, 2))
    
    initial_candidates = [grid_points, hex_points, rand_points, pert_grid]
    
    for init_pts in initial_candidates:
        if time.time() - start_time > time_budget:
            break
        
        # Normalize initial points to have max_dist = 1 to help SQP
        d_init = pdist(init_pts)
        max_d_init = np.max(d_init)
        if max_d_init > 0:
            init_pts_scaled = init_pts / max_d_init
        else:
            init_pts_scaled = init_pts
            
        pts, min_d, max_d, success = _sqp_maximize_min_dist(init_pts_scaled, max_iter=200, tol=1e-6)
        _update(pts, min_d, max_d)
        
    # If we have time, refine the best point found
    if best_points is not None and time.time() - start_time < time_budget * 0.5:
        # Run SQP again with tighter tolerance
        pts, min_d, max_d, success = _sqp_maximize_min_dist(best_points, max_iter=500, tol=1e-8)
        _update(pts, min_d, max_d)
        
    # Final refinement with Nelder-Mead on the score directly?
    # SLSQP might get stuck. NM is robust.
    if best_points is not None and time.time() - start_time < time_budget * 0.8:
        def _neg_score(flat):
            pts = flat.reshape(16, 2)
            d = pdist(pts)
            if len(d) == 0:
                return 0.0
            d_min = np.min(d)
            d_max = np.max(d)
            if d_max == 0:
                return 0.0
            return -(d_min / d_max) ** 2
        
        try:
            res = minimize(_neg_score, best_points.ravel(), method='Nelder-Mead',
                           options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12})
            pts = res.x.reshape(16, 2)
            d = pdist(pts)
            min_d = np.min(d)
            max_d = np.max(d)
            _update(pts, min_d, max_d)
        except Exception:
            pass
            
    if best_points is None:
        return grid_points, _compute_score(grid_points)
        
    return best_points, best_score

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    start_time = time.time()
    deadline = start_time + 600  # safety margin before 630s kill
    
    # Phase 1: Feasibility-driven search
    points, score = _bisection_solve(None, time_budget=400)
    
    # Phase 2: If time remains, do a few random restarts with local refinement
    if time.time() < deadline - 100:
        rng = np.random.default_rng(123)
        best_score = score
        best_points = points
        
        for _ in range(5): # Max 5 restarts
            if time.time() > deadline - 100:
                break
            
            # Random start
            rand_pts = rng.uniform(-1, 1, size=(16, 2))
            
            # Quick SQP refinement
            pts, min_d, max_d, success = _sqp_maximize_min_dist(rand_pts, max_iter=100, tol=1e-5)
            d = pdist(pts)
            if len(d) > 0:
                ratio = np.min(d) / np.max(d)
                sc = ratio ** 2
                if sc > best_score:
                    best_score = sc
                    best_points = pts
    
    # Ensure valid output
    if best_points is None or best_points.shape != (16, 2) or not np.all(np.isfinite(best_points)):
        # Fallback to grid
        x = np.linspace(-1.5, 1.5, 4)
        y = np.linspace(-1.5, 1.5, 4)
        xx, yy = np.meshgrid(x, y)
        best_points = np.column_stack([xx.ravel(), yy.ravel()])
        
    return best_points

# EVOLVE-BLOCK-END
