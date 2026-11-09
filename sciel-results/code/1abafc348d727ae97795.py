"""Circle packing in unit square: 26 circles, maximize sum of radii."""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 26
TIME_LIMIT = 120.0
SAFETY_MARGIN = 5.0  # seconds before hard limit


def _compute_score(circles):
    """Compute sum of radii for a valid packing, or -inf if invalid."""
    n = len(circles)
    if n == 0:
        return 0.0

    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]

    # Check all radii are positive
    if np.any(r <= 0):
        return -np.inf

    # Check all circles are inside the unit square
    if np.any(x - r < -1e-9) or np.any(x + r > 1 + 1e-9):
        return -np.inf
    if np.any(y - r < -1e-9) or np.any(y + r > 1 + 1e-9):
        return -np.inf

    # Check no overlapping circles
    for i in range(n):
        for j in range(i + 1, n):
            dist = math.hypot(x[i] - x[j], y[i] - y[j])
            if dist < r[i] + r[j] - 1e-9:
                return -np.inf

    return np.sum(r)


def _make_initial_grid(n):
    """Create initial grid layout with strict non-overlap."""
    side = int(np.ceil(np.sqrt(n)))
    xs = []
    ys = []
    rs = []
    count = 0
    for i in range(side):
        for j in range(side):
            if count >= n:
                break
            cx = (i + 0.5) / side
            cy = (j + 0.5) / side
            # Shrink radius slightly to ensure strict disjoint interiors
            r = 0.45 / side
            xs.append(cx)
            ys.append(cy)
            rs.append(r)
            count += 1
    return np.array(xs), np.array(ys), np.array(rs)


def _make_initial_random(n, seed=42):
    """Create random initial layout with small radii."""
    rng = np.random.RandomState(seed)
    x = rng.uniform(0.1, 0.9, n)
    y = rng.uniform(0.1, 0.9, n)
    r = np.full(n, 0.02)
    return x, y, r


def _objective(params, n):
    """Objective function for optimization: minimize negative sum of radii."""
    x = params[0:n]
    y = params[n:2*n]
    r = params[2*n:3*n]

    # Penalty for being outside the unit square
    penalty = 0.0
    for i in range(n):
        if x[i] - r[i] < 0:
            penalty += (x[i] - r[i])**2
        if x[i] + r[i] > 1:
            penalty += (x[i] + r[i] - 1)**2
        if y[i] - r[i] < 0:
            penalty += (y[i] - r[i])**2
        if y[i] + r[i] > 1:
            penalty += (y[i] + r[i] - 1)**2

    # Penalty for overlapping circles
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
            min_dist = r[i] + r[j]
            if dist < min_dist:
                penalty += (min_dist - dist)**2

    # Objective: maximize sum of radii, so minimize negative sum + penalties
    return -np.sum(r) + 100.0 * penalty


def _optimize_from_initial(initial, n, time_budget):
    """Run optimization from a given initial configuration using L-BFGS-B."""
    start_time = time.time()
    params0 = initial.flatten()

    # Define bounds: x,y in [0,1], r in [0, 0.5]
    bounds = [(0, 1)] * (2 * n) + [(0, 0.5)] * n

    # Use L-BFGS-B which is faster and handles bounds
    try:
        result = minimize(
            _objective,
            params0,
            args=(n,),
            method='L-BFGS-B',
            bounds=bounds,
            options={
                'maxiter': 1000,
                'ftol': 1e-8,
                'gtol': 1e-5
            }
        )

        elapsed = time.time() - start_time
        if elapsed > time_budget:
            return None

        # Convert back to circle format
        x = result.x[0:n]
        y = result.x[n:2*n]
        r = result.x[2*n:3*n]

        # Clip radii to be positive and valid
        r = np.clip(r, 1e-6, 0.5)

        # Re-clip positions to be consistent with radii
        for i in range(n):
            if x[i] - r[i] < 0:
                x[i] = r[i]
            if x[i] + r[i] > 1:
                x[i] = 1 - r[i]
            if y[i] - r[i] < 0:
                y[i] = r[i]
            if y[i] + r[i] > 1:
                y[i] = 1 - r[i]

        circles = np.column_stack([x, y, r])

        # Validate and compute score
        score = _compute_score(circles)
        if np.isinf(score):
            return None

        return circles, score

    except Exception:
        return None


def _max_radius(x, y, xs, ys, rs, i):
    """
    Calculate the maximum radius for circle i at position (x, y)
    given the positions and radii of all other circles.
    """
    # Max radius limited by walls
    r = min(x, 1.0 - x, y, 1.0 - y)
    
    if r <= 1e-9:
        return 1e-9

    # Max radius limited by other circles
    # Vectorized distance calculation
    dx = xs - x
    dy = ys - y
    dists = np.sqrt(dx*dx + dy*dy)
    
    # Create a mask to exclude i
    mask = np.ones(N, dtype=bool)
    mask[i] = False
    
    # Calculate constraint radii
    constrained_r = dists[mask] - rs[mask]
    
    if np.any(constrained_r):
        r = min(r, np.min(constrained_r))
        
    return max(r, 1e-9)


def _local_search(xs, ys, rs, deadline):
    """
    Local search: iterate through circles, try to move each one to a position
    that allows for a larger radius.
    """
    best_score = np.sum(rs)
    n = N
    rng = np.random.RandomState(42)
    
    # We will perform multiple passes
    # In each pass, we try to improve each circle
    # If no improvement in a full pass, we do some random jumps
    
    while time.time() < deadline:
        improved = False
        
        for i in range(n):
            if time.time() >= deadline:
                return xs, ys, rs, best_score
            
            old_r = rs[i]
            old_x, old_y = xs[i], ys[i]
            
            # Try small perturbations around current position
            # and some random jumps
            best_r = old_r
            best_x, best_y = old_x, old_y
            
            # 1. Small local search (grid-like or random steps)
            # We try a few random steps in a small neighborhood
            for _ in range(30):
                # Random step size, decreasing? No, fixed small step for local search
                step = 0.02
                dx = rng.uniform(-step, step)
                dy = rng.uniform(-step, step)
                
                nx = old_x + dx
                ny = old_y + dy
                
                # Clamp to valid range for the *current* radius to ensure feasibility
                # Actually, we want to find a new radius. 
                # If we move to (nx, ny), the max radius is _max_radius(nx, ny, ...)
                # But we must ensure the circle doesn't overlap *during* the move? 
                # No, the move is instantaneous in the model. 
                # However, if we pick a position that is currently overlapping, 
                # the _max_radius function will return a small radius (limited by overlap).
                # We want to find a position where the *potential* radius is high.
                
                # Clamp position to be inside the square (center can be anywhere, 
                # radius will be limited by walls)
                if nx < 0 or nx > 1 or ny < 0 or ny > 1:
                    continue
                    
                new_r = _max_radius(nx, ny, xs, ys, rs, i)
                
                # We only accept if new_r is significantly larger than old_r
                # Note: moving to a position with a smaller radius is bad for sum, 
                # but might allow others to grow? 
                # This simple local search maximizes *individual* radius.
                # This is a greedy approach. 
                if new_r > best_r + 1e-8:
                    best_r = new_r
                    best_x, best_y = nx, ny
            
            # 2. Random jump to escape local minima
            # Occasionally try a larger jump
            if rng.rand() < 0.1:
                for _ in range(10):
                    nx = rng.uniform(0.05, 0.95)
                    ny = rng.uniform(0.05, 0.95)
                    new_r = _max_radius(nx, ny, xs, ys, rs, i)
                    if new_r > best_r + 1e-8:
                        best_r = new_r
                        best_x, best_y = nx, ny
            
            if best_r > old_r + 1e-8:
                xs[i], ys[i], rs[i] = best_x, best_y, best_r
                improved = True
                current_score = np.sum(rs)
                if current_score > best_score:
                    best_score = current_score
        
        # If no improvement in a full pass, we might be stuck.
        # The random jumps inside the loop help, but if we stagnate completely, 
        # we can break or just continue until time runs out.
        # For now, we just continue until deadline.
        
    return xs, ys, rs, best_score


def construct():
    """Construct 26 circles in unit square maximizing sum of radii."""
    start_time = time.time()
    deadline = start_time + TIME_LIMIT - SAFETY_MARGIN

    best_circles = None
    best_score = -np.inf

    # Try multiple initial configurations
    initial_configs = []

    # Grid initial
    grid_xs, grid_ys, grid_rs = _make_initial_grid(N)
    grid_init = np.column_stack([grid_xs, grid_ys, grid_rs])
    initial_configs.append(grid_init)

    # Random initials with different seeds
    for seed in range(10):
        rx, ry, rr = _make_initial_random(N, seed=seed + 100)
        rand_init = np.column_stack([rx, ry, rr])
        initial_configs.append(rand_init)

    # Run optimization from each initial
    for i, init in enumerate(initial_configs):
        elapsed = time.time() - start_time
        remaining = deadline - time.time()

        if remaining < 5.0:  # Need at least 5 seconds for one optimization
            break

        # Allocate time budget based on remaining time
        time_budget = min(remaining * 0.8, 20.0)

        result = _optimize_from_initial(init, N, time_budget)

        if result is not None:
            circles, score = result
            if score > best_score:
                best_score = score
                best_circles = circles.copy()

        # Check if we should stop
        if time.time() > deadline:
            break

    # If no valid result from optimization, fall back to grid
    if best_circles is None:
        best_circles = grid_init.copy()
        best_score = _compute_score(best_circles)

    # Now, apply local search to the best configuration found
    if best_circles is not None and time.time() < deadline:
        xs = best_circles[:, 0].copy()
        ys = best_circles[:, 1].copy()
        rs = best_circles[:, 2].copy()
        
        # Run local search with remaining time
        local_deadline = deadline
        xs_opt, ys_opt, rs_opt, score_opt = _local_search(xs, ys, rs, local_deadline)
        
        circles_opt = np.column_stack([xs_opt, ys_opt, rs_opt])
        final_score = _compute_score(circles_opt)
        
        if not np.isinf(final_score) and final_score > best_score:
            best_score = final_score
            best_circles = circles_opt

    # Final validation
    if best_circles is None:
        best_circles = grid_init

    final_score = _compute_score(best_circles)
    if np.isinf(final_score):
        # As a last resort, return the grid
        best_circles = grid_init

    return best_circles
