"""
Packing 100 equal spheres in a unit cube using an Active-Set SOCP Strategy.

Strategy:
1. Initialize with a structured grid.
2. Iteratively identify the "active set" of constraints:
   - The K nearest-neighbor pairs (distance constraints).
   - The spheres closest to the cube walls (boundary constraints).
3. Solve a convex optimization problem (approximating an SOCP) using SLSQP
   to maximize the radius r subject to the active constraints.
   - Variables: positions (300) + radius (1) = 301 variables.
   - Objective: -r (maximize r).
   - Constraints:
     - ||p_i - p_j|| >= r for active pairs (i, j).
     - 0.5 - |p_ik| >= r for active boundary constraints.
     - Box constraints on positions.
4. Repeat with updated active sets and increasing K to refine the packing.
5. Final polish with a tight tolerance.

This approach is different from the soft-min gradient descent because it
enforces the geometry of the entire local neighborhood simultaneously,
allowing the solver to navigate the non-convex landscape by resolving
structural conflicts (e.g., "this sphere is too close to A, but if I move it,
it gets closer to B, but B is also constrained by C...").
"""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 100

def _score(points):
    """Return the maximal radius for the given set of centres."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    # Inside the cube [-0.5, 0.5]^3
    margin = 0.5 - np.abs(pts).max(axis=1)
    r_margin = margin.min()
    if r_margin <= 0:
        return 0.0
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = dists.min()
    r_pair = min_dist / 2.0
    return min(r_pair, r_margin)

def _grid_init(nx=5, ny=5, nz=4):
    """Generate a grid-based initialisation."""
    xs = np.linspace(-0.4, 0.4, nx)
    ys = np.linspace(-0.4, 0.4, ny)
    zs = np.linspace(-0.375, 0.375, nz)
    points = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    return points

def _solve_active_set_slsqp(points, K_pairs=10, K_walls=10, maxiter=1000, tol=1e-8):
    """
    Solve the packing problem using SLSQP with an active set of constraints.
    
    Parameters:
    -----------
    points : np.ndarray
        Current positions (N, 3).
    K_pairs : int
        Number of nearest-neighbor pairs to enforce.
    K_walls : int
        Number of wall constraints to enforce.
    maxiter : int
        Max iterations for SLSQP.
    tol : float
        Tolerance for SLSQP.
    
    Returns:
    --------
    new_points : np.ndarray
        Optimized positions.
    new_r : float
        Optimized radius.
    """
    n_vars = N * 3 + 1  # 300 positions + 1 radius
    
    # Initial guess
    x0 = np.zeros(n_vars)
    x0[:N*3] = points.flatten()
    # Initial radius estimate
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=2))
    np.fill_diagonal(dists, np.inf)
    r_pair = dists.min() / 2.0
    r_margin = (0.5 - np.abs(points).max(axis=1)).min()
    x0[-1] = min(r_pair, r_margin) * 0.99
    
    # Identify active sets
    # 1. Nearest neighbor pairs
    # Flatten the distance matrix to find the K smallest distances (excluding diagonal)
    dists_flat = dists.flatten()
    # Set diagonal to inf to ignore
    idx = np.arange(N)
    dists_flat[idx * N + idx] = np.inf
    
    # Get indices of K smallest distances
    # We need to be careful with duplicates (i,j) and (j,i)
    # Sort and pick unique pairs
    sorted_indices = np.argsort(dists_flat)
    active_pairs = []
    seen = set()
    for idx_flat in sorted_indices:
        i, j = np.unravel_index(idx_flat, (N, N))
        if i == j:
            continue
        pair = (min(i, j), max(i, j))
        if pair not in seen:
            active_pairs.append(pair)
            seen.add(pair)
            if len(active_pairs) >= K_pairs:
                break
    
    # 2. Wall constraints
    # Calculate margin for each sphere
    margins = 0.5 - np.abs(points).max(axis=1)
    # Sort by margin (smallest first)
    wall_order = np.argsort(margins)
    active_walls = wall_order[:K_walls]
    
    # Objective: minimize -r
    def objective(x):
        return -x[-1]
    
    # Gradient of objective
    def objective_grad(x):
        grad = np.zeros(n_vars)
        grad[-1] = -1.0
        return grad
    
    # Constraints: g(x) >= 0
    constraints = []
    
    # Pair constraints: ||p_i - p_j|| - r >= 0
    for (i, j) in active_pairs:
        def pair_constraint(x, i=i, j=j):
            p_i = x[i*3:(i+1)*3]
            p_j = x[j*3:(j+1)*3]
            r = x[-1]
            return np.linalg.norm(p_i - p_j) - r
        
        def pair_grad(x, i=i, j=j):
            grad = np.zeros(n_vars)
            p_i = x[i*3:(i+1)*3]
            p_j = x[j*3:(j+1)*3]
            diff = p_i - p_j
            dist = np.linalg.norm(diff)
            if dist > 1e-12:
                grad[i*3:(i+1)*3] = diff / dist
                grad[j*3:(j+1)*3] = -diff / dist
            grad[-1] = -1.0
            return grad
        
        constraints.append({'type': 'ineq', 'fun': pair_constraint, 'jac': pair_grad})
        
    # Wall constraints: 0.5 - |p_ik| - r >= 0
    # We need to handle the absolute value carefully.
    # For a specific sphere i, the constraint is min_k (0.5 - |p_ik|) - r >= 0
    # This is equivalent to 6 constraints per sphere (if we consider all faces), but 
    # for efficiency, we can just use the specific face that is currently closest.
    # However, for robustness in SLSQP, it's better to enforce all 6 faces for the active walls?
    # No, that's 6 * K_walls constraints. Let's stick to the specific active face per sphere.
    # Actually, the "active" wall constraint is the one that is tightest.
    # Let's enforce the constraint for the specific face that is closest for the active spheres.
    
    for i in active_walls:
        p_i = points[i]
        # Find the closest face for this sphere in the current configuration
        abs_p = np.abs(p_i)
        # The margin is 0.5 - max(abs_p)
        # The active coordinate is the one with max(abs_p)
        active_coord = np.argmax(abs_p)
        
        def wall_constraint(x, i=i, active_coord=active_coord):
            p_i = x[i*3:(i+1)*3]
            r = x[-1]
            # Constraint: 0.5 - |p_i[active_coord]| - r >= 0
            return 0.5 - np.abs(p_i[active_coord]) - r
            
        def wall_grad(x, i=i, active_coord=active_coord):
            grad = np.zeros(n_vars)
            p_i = x[i*3:(i+1)*3]
            r = x[-1]
            val = p_i[active_coord]
            sign = 1.0 if val >= 0 else -1.0
            # d/dx (0.5 - sign * p_i[active_coord] - r)
            grad[i*3 + active_coord] = -sign
            grad[-1] = -1.0
            return grad
            
        constraints.append({'type': 'ineq', 'fun': wall_constraint, 'jac': wall_grad})
        
    # Bounds
    bounds = [(-0.5, 0.5)] * (N * 3)
    bounds.append((0.0, 0.5))  # Radius bound
    
    try:
        result = minimize(objective, x0, jac=objective_grad, method='SLSQP', bounds=bounds,
                          constraints=constraints, options={'maxiter': maxiter, 'ftol': tol})
        
        if result.success or (result.fun < objective(x0)):
            new_points = result.x[:N*3].reshape(N, 3)
            new_r = result.x[-1]
            return new_points, new_r
        else:
            return points, x0[-1]
    except Exception:
        return points, x0[-1]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Initial solution from grid
    points = _grid_init()
    best_points = points.copy()
    best_score = _score(best_points)
    
    # Phase 1: Active Set Optimization with increasing K
    # Start with small K to find a rough structure, then increase K to refine
    K_values = [5, 10, 20, 50, 100, 200]
    
    for K in K_values:
        if time.time() > deadline - 20:
            break
            
        # Update active sets based on current best points
        # We call _solve_active_set_slsqp which internally identifies the active sets
        # based on the input points.
        
        # To make it more robust, we can run multiple iterations with the same K
        # to ensure convergence to the active set optimum.
        for _ in range(2):
            if time.time() > deadline - 10:
                break
                
            opt_points, opt_r = _solve_active_set_slsqp(best_points, K_pairs=K, K_walls=10, maxiter=500, tol=1e-8)
            
            # Calculate the actual score of the optimized points
            opt_score = _score(opt_points)
            
            if opt_score > best_score:
                best_points = opt_points
                best_score = opt_score
    
    # Phase 2: Multi-start with random perturbations
    random.seed(42)
    np.random.seed(42)
    
    while time.time() < deadline - 15:
        # Perturb the current best
        if random.random() < 0.5:
            new_start = best_points + np.random.normal(0, 0.02, (N, 3))
        else:
            # Try a different grid offset
            new_start = _grid_init() + np.random.normal(0, 0.01, (N, 3))
        
        new_start = np.clip(new_start, -0.5, 0.5)
        
        # Run a quick active set optimization
        opt_points, opt_r = _solve_active_set_slsqp(new_start, K_pairs=20, K_walls=10, maxiter=200, tol=1e-7)
        opt_score = _score(opt_points)
        
        if opt_score > best_score:
            best_points = opt_points
            best_score = opt_score
            
    # Phase 3: Final Polish
    # Use a very large K to enforce almost all constraints
    if time.time() < deadline - 5:
        # We can't enforce all 4950 pairs, but we can enforce a large number
        # The true active set in a tight packing is usually O(N)
        # Let's try K=500
        opt_points, opt_r = _solve_active_set_slsqp(best_points, K_pairs=500, K_walls=50, maxiter=1000, tol=1e-10)
        opt_score = _score(opt_points)
        if opt_score > best_score:
            best_points = opt_points
            best_score = opt_score

    return best_points

# EVOLVE-BLOCK-END
