"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution


def _objective(params):
    """
    params: flattened array of length 3*N
    Returns: negative sum of radii (to minimize)
    """
    params = np.reshape(params, (N, 3))
    xs = params[:, 0]
    ys = params[:, 1]
    rs = params[:, 2]
    
    # Bounds: 0 <= x <= W, 0 <= y <= H, 0 <= r <= 1
    # We need to ensure circles are inside the rectangle
    # Rectangle has perimeter 4, so 2*(W+H) = 4 => W+H = 2
    # We'll fix W and H for now, or optimize them too? 
    # Let's fix W and H to a reasonable aspect ratio first, e.g., square 1x1 or 0.5x1.5
    # Actually, the problem says "variable rectangle of perimeter 4".
    # So we can choose W and H such that 2*(W+H)=4.
    # Let's optimize W and H as well? That adds 2 variables.
    # Or fix aspect ratio? 
    # Let's try optimizing W and H too.
    # Total vars: 21*3 + 2 = 65.
    # But the interface requires returning n circles, and the evaluator checks the bounding rectangle.
    # The bounding rectangle is determined by the circles' extents? 
    # "bounding rectangle has perimeter at most 4"
    # This implies the rectangle is the minimal bounding box of the circles?
    # Or is the rectangle fixed and circles must fit inside?
    # "Pack 21 circles ... in a variable rectangle of perimeter 4"
    # Usually this means we choose the rectangle dimensions W, H with 2(W+H)=4, and pack circles inside.
    # The evaluator likely checks:
    # 1. All circles are inside [0, W] x [0, H] (or some origin)
    # 2. No overlaps
    # 3. 2(W+H) <= 4
    # But the output is just (n, 3) array of (x, y, r).
    # How does the evaluator know W and H?
    # It must infer W and H from the circles, or assume a fixed rectangle?
    # "construct() -> array of shape (n, 3): n disjoint circles whose bounding rectangle has perimeter at most 4"
    # This suggests the bounding rectangle is computed from the circles.
    # So, W = max(x+r) - min(x-r), H = max(y+r) - min(y-r).
    # And we need 2*(W+H) <= 4.
    # Objective: maximize sum(r_i)
    
    # Let's assume the evaluator computes the bounding box from the returned circles.
    # So we need to ensure the bounding box perimeter is <= 4.
    # And circles must be disjoint.
    
    # For the optimizer, we can add penalties for constraint violations.
    # Or use a method that handles constraints.
    # differential_evolution doesn't support constraints directly in scipy (as of recent versions? 
    # Actually, scipy.optimize.differential_evolution does NOT support constraints.
    # We can use penalties.
    
    # Let's define the bounding box:
    min_x = np.min(xs - rs)
    max_x = np.max(xs + rs)
    min_y = np.min(ys - rs)
    max_y = np.max(ys + rs)
    
    W = max_x - min_x
    H = max_y - min_y
    perimeter = 2 * (W + H)
    
    # Penalty for perimeter > 4
    pen_perim = max(0, perimeter - 4) * 1000
    
    # Penalty for overlaps
    pen_overlap = 0
    for i in range(N):
        for j in range(i+1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx*dx + dy*dy)
            min_dist = rs[i] + rs[j]
            if dist < min_dist:
                pen_overlap += (min_dist - dist) * 1000
    
    # Also, radii should be positive
    pen_radius = np.sum(np.maximum(0, -rs)) * 1000
    
    return -np.sum(rs) + pen_perim + pen_overlap + pen_radius


def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s margin
    
    # Initial guess: uniform grid
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    initial = np.array(cells[:N])
    
    # Flatten
    x0 = initial.flatten()
    
    # Bounds for each parameter
    # x, y: [0, 2] (since W+H=2, max dimension is 2)
    # r: [0, 1]
    bounds = [(0, 2)] * (2 * N) + [(0, 1)] * N
    
    # Run differential evolution
    best_result = initial
    best_val = _objective(x0)
    
    try:
        # Use a reasonable number of iterations
        # tol=1e-8, maxiter=100, popsize=15
        result = differential_evolution(
            _objective, 
            bounds, 
            maxiter=50, 
            popsize=10, 
            tol=1e-8, 
            seed=42,
            polish=True
        )
        
        if result.fun < best_val:
            best_params = np.reshape(result.x, (N, 3))
            best_val = result.fun
            best_result = best_params
    except Exception:
        pass
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > 140:
        # Return best found so far
        pass
    
    # Ensure the result is valid (perimeter <= 4, no overlaps)
    # If not, fall back to initial
    xs = best_result[:, 0]
    ys = best_result[:, 1]
    rs = best_result[:, 2]
    
    min_x = np.min(xs - rs)
    max_x = np.max(xs + rs)
    min_y = np.min(ys - rs)
    max_y = np.max(ys + rs)
    
    W = max_x - min_x
    H = max_y - min_y
    perimeter = 2 * (W + H)
    
    # Check overlaps
    valid = True
    if perimeter > 4:
        valid = False
    for i in range(N):
        for j in range(i+1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx*dx + dy*dy)
            min_dist = rs[i] + rs[j]
            if dist < min_dist - 1e-9:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        return initial
    
    return best_result
# EVOLVE-BLOCK-END
