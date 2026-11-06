"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, LinearConstraint

def _initial_grid(n_points):
    """Create an initial grid-like placement with slight perturbation."""
    m = int(np.ceil(np.sqrt(n_points)))
    # We want to fit n_points into an m x m grid, but we only use n_points of them
    # Actually, let's just make a square grid that covers the area.
    # To be more efficient, we can use a hexagonal packing initially.
    
    # Hexagonal packing initialization
    # Row height = sqrt(3)/2 * d
    # Let's try to fit as many as possible.
    # Approximate d: area per circle ~ 1/sqrt(3/2) * r^2? No, area fraction ~ pi/(2*sqrt(3)) ~ 0.9069
    # r ~ 0.0366, d ~ 0.0732
    # Let's just use a grid for simplicity and robustness, then optimize.
    
    # A simple square grid is a safe start.
    cols = int(np.ceil(np.sqrt(n_points)))
    rows = int(np.ceil(n_points / cols))
    
    # Scale to fit in [-0.5, 0.5]
    # If we use a grid of size cols x rows, the spacing is 1/cols and 1/rows?
    # No, we want to maximize the minimum distance.
    # Let's place them in a grid that fills the square.
    
    # Let's try a hexagonal lattice for better initial spacing.
    # This is more complex. Let's stick to a perturbed square grid for reliability.
    
    xs = np.linspace(-0.5 + 1/(2*cols), 0.5 - 1/(2*cols), cols)
    ys = np.linspace(-0.5 + 1/(2*rows), 0.5 - 1/(2*rows), rows)
    
    grid = []
    for y in ys:
        for x in xs:
            grid.append([x, y])
            if len(grid) == n_points:
                break
        if len(grid) == n_points:
            break
            
    grid = np.array(grid)
    
    # Add small random perturbation to break symmetry
    np.random.seed(42)
    grid += np.random.uniform(-0.001, 0.001, grid.shape)
    
    # Clip to square
    grid[:, 0] = np.clip(grid[:, 0], -0.49, 0.49)
    grid[:, 1] = np.clip(grid[:, 1], -0.49, 0.49)
    
    return grid

def _objective(x, n_points):
    """
    Objective function for optimization.
    We want to maximize the minimum distance.
    Equivalent to minimizing -min_distance.
    But min is non-smooth.
    We can use a soft-min or just minimize the sum of squared violations of a target radius?
    
    Better approach: Minimize the negative of the minimum distance.
    Or, even better, use a formulation that is differentiable.
    
    Let's try minimizing: sum_{i<j} max(0, d_target - dist(i,j))^2 + sum_i max(0, 0.5 - |x_i| - r_target)^2 ...
    This is getting complicated.
    
    Standard approach for circle packing:
    Maximize r such that dist(i,j) >= 2r and |x_i| <= 0.5 - r.
    
    We can use a penalty method or SLSQP with constraints.
    Let's use SLSQP.
    Variables: x_0, y_0, ..., x_{N-1}, y_{N-1}, r
    Objective: minimize -r
    Constraints:
    1. dist(i,j) - 2r >= 0 for all i < j
    2. 0.5 - |x_i| - r >= 0
    3. 0.5 - |y_i| - r >= 0
    
    Number of variables: 2N + 1 = 401
    Number of constraints: N(N-1)/2 + 2N = 19900 + 400 = 20300
    This is a large non-linear programming problem.
    SLSQP might be slow or get stuck.
    
    Alternative: Gradient descent with a soft objective.
    f(x, r) = -r + lambda * sum_{i<j} max(0, 2r - dist(i,j))^2 + mu * sum_i (max(0, |x_i| - (0.5 - r))^2 + max(0, |y_i| - (0.5 - r))^2)
    
    This is differentiable almost everywhere.
    We can use scipy.optimize.minimize with L-BFGS-B or similar.
    
    Let's try a simpler approach first: Local search / Gradient descent on positions only, keeping r as the min distance.
    But we want to maximize r.
    
    Let's use the penalty method with scipy.optimize.minimize.
    """
    # Unpack variables
    r = x[-1]
    pos = x[:-1].reshape(-1, 2)
    
    # Objective: -r
    obj = -r
    
    # Penalty terms
    # Pairwise distances
    # dist(i,j) = sqrt((x_i-x_j)^2 + (y_i-y_j)^2)
    # We want dist(i,j) >= 2r
    # Penalty: sum max(0, 2r - dist(i,j))^2
    
    # Vectorized distance calculation
    # pos is (N, 2)
    # diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]  # (N, N, 2)
    # dists = np.linalg.norm(diff, axis=2) # (N, N)
    
    # This is N^2 = 40000 operations. Acceptable.
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2) + 1e-12) # +1e-12 to avoid division by zero in gradients if we computed them manually, but we use finite diff or analytical?
    
    # For scipy.optimize.minimize, we can provide the gradient.
    # But let's start without gradient and see if it's fast enough.
    # 401 variables, 150s limit.
    
    # Boundary penalties
    # We want |x_i| <= 0.5 - r  =>  x_i^2 + y_i^2? No, square.
    # |x_i| <= 0.5 - r  =>  x_i <= 0.5 - r and -x_i <= 0.5 - r
    # Penalty: sum (max(0, x_i - (0.5 - r))^2 + max(0, -x_i - (0.5 - r))^2 + max(0, y_i - (0.5 - r))^2 + max(0, -y_i - (0.5 - r))^2)
    
    # Let's compute penalties
    # Pairwise
    # dists is (N, N). We only need upper triangle.
    # mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    # violations = np.where(dists[mask] < 2*r, (2*r - dists[mask])**2, 0)
    # pair_penalty = np.sum(violations)
    
    # Boundary
    # limit = 0.5 - r
    # x_pen = np.sum(np.maximum(0, pos[:, 0] - limit)**2) + np.sum(np.maximum(0, -pos[:, 0] - limit)**2)
    # y_pen = np.sum(np.maximum(0, pos[:, 1] - limit)**2) + np.sum(np.maximum(0, -pos[:, 1] - limit)**2)
    # bound_penalty = x_pen + y_pen
    
    # Total objective
    # lambda = 1000
    # mu = 1000
    # return obj + lambda * pair_penalty + mu * bound_penalty
    
    # This function is called many times. Need to be fast.
    # Let's optimize the calculation.
    
    return None # Placeholder, will implement in wrapper

def construct():
    start_time = time.time()
    deadline = 140.0 # Leave 10s margin
    
    # Initial guess
    x0 = _initial_grid(N)
    # Initial radius guess: slightly smaller than optimal to be safe
    # Optimal is ~0.0366
    r0 = 0.036
    x0_full = np.concatenate([x0.flatten(), [r0]])
    
    best_x = x0_full.copy()
    best_score = -1.0
    
    def _calc_score(pos, r):
        """Calculate the actual score (min radius) for a given configuration."""
        # Check boundaries
        min_boundary = 0.5 - np.max(np.abs(pos))
        # Check pairwise distances
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2) + 1e-12)
        # Mask diagonal
        mask = ~np.eye(N, dtype=bool)
        min_dist = np.min(dists[mask])
        min_pair = min_dist / 2.0
        return min(min_boundary, min_pair)
    
    current_score = _calc_score(x0, r0)
    best_score = current_score
    
    # Optimization loop
    # We will use scipy.optimize.minimize with a penalty objective.
    
    def _penalty_obj(x_full):
        r = x_full[-1]
        pos = x_full[:-1].reshape(-1, 2)
        
        # Boundaries
        limit = 0.5 - r
        # If r is too large, limit becomes negative, which is bad.
        # We should constrain r to be positive and small enough.
        
        # Pairwise
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2) + 1e-12)
        mask = ~np.eye(N, dtype=bool)
        # Only consider pairs where dist < 2r
        # violations = np.where(dists[mask] < 2*r, (2*r - dists[mask])**2, 0)
        # pair_pen = np.sum(violations)
        
        # More efficient:
        # dists_flat = dists[mask]
        # violations = np.maximum(0, 2*r - dists_flat)**2
        # pair_pen = np.sum(violations)
        
        # Boundary
        # x_pen = np.sum(np.maximum(0, pos[:, 0] - limit)**2) + np.sum(np.maximum(0, -pos[:, 0] - limit)**2)
        # y_pen = np.sum(np.maximum(0, pos[:, 1] - limit)**2) + np.sum(np.maximum(0, -pos[:, 1] - limit)**2)
        # bound_pen = x_pen + y_pen
        
        # Let's combine
        # To save time, we can approximate or simplify.
        
        # Actually, let's just use the constraints directly with SLSQP?
        # SLSQP is slower per iteration but handles constraints explicitly.
        # Given the large number of constraints, it might be too slow.
        
        # Let's stick to penalty method with L-BFGS-B.
        
        # Calculate penalties
        dists_flat = dists[mask]
        pair_viol = np.maximum(0, 2*r - dists_flat)
        pair_pen = np.sum(pair_viol**2)
        
        # Boundary
        # pos is (N, 2)
        # limit is scalar
        # x_coords = pos[:, 0]
        # y_coords = pos[:, 1]
        # x_pen = np.sum(np.maximum(0, x_coords - limit)**2) + np.sum(np.maximum(0, -x_coords - limit)**2)
        # y_pen = np.sum(np.maximum(0, y_coords - limit)**2) + np.sum(np.maximum(0, -y_coords - limit)**2)
        # bound_pen = x_pen + y_pen
        
        # Combine
        lam = 1000.0
        mu = 1000.0
        return -r + lam * pair_pen + mu * bound_pen
        
    # Wait, I didn't define bound_pen in _penalty_obj above. Let's fix.
    
    def _penalty_obj_final(x_full):
        r = x_full[-1]
        pos = x_full[:-1].reshape(-1, 2)
        
        # Pairwise
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2) + 1e-12)
        mask = ~np.eye(N, dtype=bool)
        dists_flat = dists[mask]
        pair_viol = np.maximum(0, 2*r - dists_flat)
        pair_pen = np.sum(pair_viol**2)
        
        # Boundary
        limit = 0.5 - r
        x_coords = pos[:, 0]
        y_coords = pos[:, 1]
        x_pen = np.sum(np.maximum(0, x_coords - limit)**2) + np.sum(np.maximum(0, -x_coords - limit)**2)
        y_pen = np.sum(np.maximum(0, y_coords - limit)**2) + np.sum(np.maximum(0, -y_coords - limit)**2)
        bound_pen = x_pen + y_pen
        
        lam = 1000.0
        mu = 1000.0
        return -r + lam * pair_pen + mu * bound_pen
        
    # Run optimization
    try:
        # Use L-BFGS-B
        # Bounds for r: [0, 0.25]
        # Bounds for positions: [-0.5, 0.5]
        bounds = [(-0.5, 0.5)] * (2 * N) + [(0.001, 0.25)]
        
        res = minimize(_penalty_obj_final, x0_full, method='L-BFGS-B', bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
        
        if res.success or res.fun < _penalty_obj_final(x0_full):
            best_x = res.x
            best_pos = best_x[:-1].reshape(-1, 2)
            best_r = best_x[-1]
            new_score = _calc_score(best_pos, best_r)
            if new_score > best_score:
                best_score = new_score
                
    except Exception as e:
        print(f"Optimization failed: {e}")
        
    # Check time
    elapsed = time.time() - start_time
    if elapsed > deadline:
        # Return best found
        pos = best_x[:-1].reshape(-1, 2)
        # Ensure validity
        pos = np.clip(pos, -0.49, 0.49)
        return pos
        
    # Local refinement: Simple gradient ascent on min distance
    # Or just return the result from scipy
    
    pos = best_x[:-1].reshape(-1, 2)
    # Final clipping to ensure validity
    pos = np.clip(pos, -0.49, 0.49)
    
    return pos

# EVOLVE-BLOCK-END
