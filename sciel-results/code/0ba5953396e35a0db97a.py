"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _solve_radii_fast(x, y, max_iter=1000, tol=1e-10):
    """
    Given fixed positions x, y, find the maximum radii r_i such that
    circles are disjoint and inside [0,1]^2.
    Uses iterative constraint propagation: r_i = min(wall_i, min_j(d_ij - r_j)).
    This is fast and robust for fixed positions.
    """
    n = len(x)
    # Precompute distances
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    np.fill_diagonal(dist, np.inf)
    
    # Wall distances
    wall = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    
    # Initialize radii
    r = np.maximum(wall * 0.5, 1e-9)
    
    for _ in range(max_iter):
        new_r = r.copy()
        for i in range(n):
            # Wall constraint
            cand = wall[i]
            # Neighbor constraints
            # r_i <= d_ij - r_j  =>  r_i + r_j <= d_ij
            # We want max r_i, so r_i = min(wall_i, min_j(d_ij - r_j))
            # Note: This assumes we update r_j from previous iteration (Jacobi) or current (Gauss-Seidel).
            # Gauss-Seidel is usually faster.
            for j in range(n):
                if i == j:
                    continue
                if dist[i, j] < 1e-9:
                    continue
                neighbor_limit = dist[i, j] - r[j]
                if neighbor_limit < cand:
                    cand = neighbor_limit
            new_r[i] = max(0.0, cand)
        
        delta = np.max(np.abs(new_r - r))
        r = new_r
        if delta < tol:
            break
            
    return r

def _evaluate_position(x, y):
    """
    Evaluate the sum of radii for a given set of positions.
    Returns sum of radii and the radii array.
    """
    r = _solve_radii_fast(x, y)
    # Check validity strictly
    # If any radius is 0 or very small, or if constraints are violated due to floating point,
    # the solver above should handle it, but let's be safe.
    # The solver guarantees r_i <= wall_i and r_i + r_j <= d_ij (approximately).
    return np.sum(r), r

def _objective_positions(params, n):
    """
    Objective for optimizing positions only.
    params: shape (2n,) -> [x1, y1, x2, y2, ...]
    Returns negative sum of radii (to be minimized).
    """
    x = params[0::2]
    y = params[1::2]
    
    # Ensure positions are within [0,1]
    # The radius solver handles the boundary, but if a center is outside, wall distance is negative.
    # We should penalize centers outside [0,1] or clip them.
    # Clipping is better for optimization continuity.
    x_clip = np.clip(x, 0.0, 1.0)
    y_clip = np.clip(y, 0.0, 1.0)
    
    score, _ = _evaluate_position(x_clip, y_clip)
    
    # Penalty if any circle has radius 0 (likely due to bad positioning)
    # This encourages spreading out
    if score < 1e-6:
        return 100.0 - score
    
    return -score

def _make_initial_grid(n):
    """Create a simple grid-based initial configuration."""
    side = int(np.ceil(np.sqrt(n)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) >= n:
                break
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            cells.append([x, y])
    return np.array(cells[:n])

def construct():
    """
    Pack 26 circles with disjoint interiors inside [0,1]^2 and maximize the sum of their radii.
    Returns array of shape (n, 3): rows (x, y, radius) of n disjoint circles inside the unit square.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_circles = None
    best_score = 0.0
    
    # Initial grid
    init_pos = _make_initial_grid(N)
    init_x = init_pos[:, 0]
    init_y = init_pos[:, 1]
    
    # Evaluate initial grid
    init_score, init_r = _evaluate_position(init_x, init_y)
    if init_score > best_score:
        best_score = init_score
        best_circles = np.column_stack((init_x, init_y, init_r))
    
    rng = np.random.RandomState(42)
    
    # Multi-start optimization of positions
    n_restarts = 20  # We have time for more restarts as radius solving is fast
    
    for restart in range(n_restarts):
        if time.time() > deadline - 5.0:
            break
        
        # Generate initial position parameters
        if restart == 0:
            params = init_pos.flatten()
        elif restart < 5:
            # Small perturbation of best
            if best_circles is not None:
                params = best_circles[:, :2].flatten().copy()
            else:
                params = init_pos.flatten().copy()
            noise = rng.randn(2 * N) * 0.05
            params += noise
        else:
            # Larger perturbation or random
            if best_circles is not None:
                params = best_circles[:, :2].flatten().copy()
            else:
                params = init_pos.flatten().copy()
            noise = rng.randn(2 * N) * 0.2
            params += noise
            
        # Clip to [0,1] to keep optimizer in bounds
        params = np.clip(params, 0.0, 1.0)
        
        try:
            result = minimize(
                _objective_positions,
                params,
                args=(N,),
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6, 'maxfun': 500}
            )
        except Exception:
            continue
        
        new_params = result.x
        new_x = new_params[0::2]
        new_y = new_params[1::2]
        
        # Solve for radii with the new positions
        new_score, new_r = _evaluate_position(new_x, new_y)
        
        if new_score > best_score:
            best_score = new_score
            best_circles = np.column_stack((new_x, new_y, new_r))
            
    # If no valid circles found (should not happen), fallback
    if best_circles is None:
        best_circles = np.column_stack((init_x, init_y, init_r))
        
    # Final check and return
    # Ensure radii are positive
    best_circles[:, 2] = np.maximum(best_circles[:, 2], 1e-9)
    
    return best_circles

# EVOLVE-BLOCK-END
