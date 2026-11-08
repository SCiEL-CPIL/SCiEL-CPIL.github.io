"""Circle packing in perimeter-4 rectangle using Voronoi inradii + SA + L-BFGS-B."""
import numpy as np
import time
import math
from scipy.optimize import minimize
from scipy.spatial import Voronoi

N = 21
PERIMETER = 4.0
TIME_LIMIT = 145.0  # Safety margin before 150s limit


def _inradii(positions, W, H):
    """Compute inradius of each circle's Voronoi cell.
    
    positions: (n,2) array of circle centers
    W, H: bounding box dimensions
    Returns: (n,) array of inradii (max radius before collision)
    """
    n = len(positions)
    inradii = np.zeros(n)
    
    # Add boundary points far away to create Voronoi cells touching the boundary
    # We'll use a large bounding box approach
    # Actually, we need to constrain circles to be inside [0,W] x [0,H]
    # The inradius is min(distance to other circles, distance to boundary)
    
    # Distance to boundaries
    dist_boundary = np.minimum(
        np.minimum(positions[:, 0], W - positions[:, 0]),
        np.minimum(positions[:, 1], H - positions[:, 1])
    )
    
    # Distance to other circles
    for i in range(n):
        min_dist = dist_boundary[i]
        for j in range(n):
            if i == j:
                continue
            d = math.hypot(positions[i, 0] - positions[j, 0], 
                          positions[i, 1] - positions[j, 1])
            if d < min_dist:
                min_dist = d
        inradii[i] = 0.5 * min_dist
    
    return inradii


def _inradii_fast(positions, W, H):
    """Vectorized inradius computation."""
    n = len(positions)
    # Distance to boundaries
    dist_boundary = np.minimum(
        np.minimum(positions[:, 0], W - positions[:, 0]),
        np.minimum(positions[:, 1], H - positions[:, 1])
    )
    
    # Distance to other circles - vectorized
    # Compute pairwise distances
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist_other = np.min(dists, axis=1)
    
    inradii = 0.5 * np.minimum(dist_boundary, min_dist_other)
    return inradii


def _objective(params, W, H, penalty_weight=1000.0):
    """Objective to minimize: -sum(inradii) + penalty for perimeter violation.
    
    params: (n, 2) positions
    Returns: scalar objective
    """
    positions = params.reshape(-1, 2)
    
    # Clamp to box
    positions = np.clip(positions, 0, W)
    positions = np.clip(positions[:, 1:2], 0, H) if False else positions
    positions[:, 0] = np.clip(positions[:, 0], 0, W)
    positions[:, 1] = np.clip(positions[:, 1], 0, H)
    
    inradii = _inradii_fast(positions, W, H)
    sum_r = np.sum(inradii)
    
    # Check perimeter: the bounding box of the circles
    x_min = np.min(positions[:, 0] - inradii)
    x_max = np.max(positions[:, 0] + inradii)
    y_min = np.min(positions[:, 1] - inradii)
    y_max = np.max(positions[:, 1] + inradii)
    
    cw = x_max - x_min
    ch = y_max - y_min
    perimeter = 2 * (cw + ch)
    
    penalty = 0.0
    if perimeter > PERIMETER:
        penalty = penalty_weight * (perimeter - PERIMETER) ** 2
    
    return -sum_r + penalty


def _seed_hex():
    """Create a hexagonal packing seed."""
    # Try different hexagonal configurations
    best_val = -1
    best_positions = None
    
    for cols in range(2, 10):
        rows = int(np.ceil(N / cols))
        if rows * cols < N:
            continue
        
        # Hexagonal offset
        s = 1.0  # base spacing, will scale
        positions = []
        for j in range(rows):
            for i in range(cols):
                if len(positions) >= N:
                    break
                x = (i + 0.5) * s
                y = (j + 0.5) * s
                if j % 2 == 1:
                    x += 0.5 * s
                positions.append([x, y])
        
        positions = np.array(positions[:N])
        
        # Compute bounding box
        x_min = np.min(positions[:, 0])
        x_max = np.max(positions[:, 0])
        y_min = np.min(positions[:, 1])
        y_max = np.max(positions[:, 1])
        
        W = x_max - x_min
        H = y_max - y_min
        perimeter = 2 * (W + H)
        
        if perimeter > PERIMETER:
            # Scale down
            scale = PERIMETER / perimeter
            W *= scale
            H *= scale
            positions = (positions - positions.min(axis=0)) * scale
            # Center
            positions += (np.array([W, H]) - positions.max(axis=0)) / 2
        
        inradii = _inradii_fast(positions, W, H)
        val = np.sum(inradii)
        
        if val > best_val:
            best_val = val
            best_positions = positions.copy()
            best_W = W
            best_H = H
    
    return best_positions, best_W, best_H


def _seed_grid():
    """Create a grid packing seed."""
    best_val = -1
    best_positions = None
    best_W = None
    best_H = None
    
    for cols in range(1, N + 1):
        rows = (N + cols - 1) // cols
        if rows * cols < N:
            continue
        
        s = 1.0
        positions = []
        for j in range(rows):
            for i in range(cols):
                if len(positions) >= N:
                    break
                x = (i + 0.5) * s
                y = (j + 0.5) * s
                positions.append([x, y])
        
        positions = np.array(positions[:N])
        
        x_min = np.min(positions[:, 0])
        x_max = np.max(positions[:, 0])
        y_min = np.min(positions[:, 1])
        y_max = np.max(positions[:, 1])
        
        W = x_max - x_min
        H = y_max - y_min
        perimeter = 2 * (W + H)
        
        if perimeter > PERIMETER:
            scale = PERIMETER / perimeter
            W *= scale
            H *= scale
            positions = (positions - positions.min(axis=0)) * scale
            positions += (np.array([W, H]) - positions.max(axis=0)) / 2
        
        inradii = _inradii_fast(positions, W, H)
        val = np.sum(inradii)
        
        if val > best_val:
            best_val = val
            best_positions = positions.copy()
            best_W = W
            best_H = H
    
    return best_positions, best_W, best_H


def _validate_and_compute(circles, tol=1e-9):
    """Check validity and compute sum of radii."""
    if len(circles) == 0:
        return False, 0.0
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    
    if np.any(r <= 0):
        return False, 0.0
    
    x_min = np.min(x - r)
    x_max = np.max(x + r)
    y_min = np.min(y - r)
    y_max = np.max(y + r)
    
    W = x_max - x_min
    H = y_max - y_min
    
    if 2 * (W + H) > PERIMETER + tol:
        return False, 0.0
    
    n = len(circles)
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist_sq = dx * dx + dy * dy
            sum_r = r[i] + r[j]
            if dist_sq < (sum_r - tol) ** 2:
                return False, 0.0
    return True, np.sum(r)


def _make_circles(positions, W, H):
    """Create circles array from positions and box dimensions."""
    inradii = _inradii_fast(positions, W, H)
    circles = np.column_stack([positions, inradii])
    return circles


def _refine_lbfgsb(positions, W, H, time_budget):
    """Refine using L-BFGS-B."""
    t_start = time.time()
    
    # Objective function for L-BFGS-B
    def obj(params):
        pos = params.reshape(-1, 2)
        pos[:, 0] = np.clip(pos[:, 0], 0, W)
        pos[:, 1] = np.clip(pos[:, 1], 0, H)
        inradii = _inradii_fast(pos, W, H)
        return -np.sum(inradii)
    
    x0 = positions.flatten()
    
    # Bounds
    bounds = []
    for i in range(N):
        bounds.append((0, W))
        bounds.append((0, H))
    
    try:
        result = minimize(obj, x0, method='L-BFGS-B', bounds=bounds,
                         options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
        if result.success and time.time() - t_start < time_budget:
            new_positions = result.x.reshape(-1, 2)
            new_positions[:, 0] = np.clip(new_positions[:, 0], 0, W)
            new_positions[:, 1] = np.clip(new_positions[:, 1], 0, H)
            return new_positions
    except:
        pass
    
    return positions


def _simulated_annealing(positions, W, H, time_budget):
    """Simulated annealing for global exploration."""
    t_start = time.time()
    current = positions.copy()
    best = positions.copy()
    
    inradii_best = _inradii_fast(best, W, H)
    best_val = np.sum(inradii_best)
    
    inradii_cur = _inradii_fast(current, W, H)
    cur_val = np.sum(inradii_cur)
    
    T = 0.05  # initial temperature
    T_min = 0.001
    cooling = 0.995
    step_scale = 0.1
    
    while time.time() - t_start < time_budget and T > T_min:
        # Propose move
        trial = current.copy()
        k = np.random.randint(N)
        d = np.random.randint(2)
        trial[k, d] += np.random.randn() * step_scale
        trial[k, d] = np.clip(trial[k, d], 0, W if d == 0 else H)
        
        inradii_trial = _inradii_fast(trial, W, H)
        trial_val = np.sum(inradii_trial)
        
        delta = trial_val - cur_val
        if delta > 0 or np.random.random() < math.exp(delta / T):
            current = trial
            cur_val = trial_val
            if cur_val > best_val:
                best_val = cur_val
                best = trial.copy()
        
        T *= cooling
        step_scale *= 0.999
    
    return best


def _optimize_positions(positions, W, H, time_budget):
    """Full optimization: SA + L-BFGS-B iteration."""
    t_start = time.time()
    best_positions = positions.copy()
    best_val = np.sum(_inradii_fast(best_positions, W, H))
    
    current = positions.copy()
    iters = 0
    
    while time.time() - t_start < time_budget:
        iters += 1
        # SA phase
        sa_time = min(30.0, (time_budget - (time.time() - t_start)) * 0.5)
        if sa_time > 2.0:
            current = _simulated_annealing(current, W, H, sa_time)
        
        # L-BFGS-B refinement
        lbfgsb_time = min(20.0, (time_budget - (time.time() - t_start)) * 0.3)
        if lbfgsb_time > 1.0:
            current = _refine_lbfgsb(current, W, H, lbfgsb_time)
        
        # Check improvement
        inradii = _inradii_fast(current, W, H)
        val = np.sum(inradii)
        if val > best_val:
            best_val = val
            best_positions = current.copy()
        
        # Check time
        if time.time() - t_start >= time_budget - 5.0:
            break
        
        # Restart from best with perturbation
        if iters > 1:
            current = best_positions + np.random.randn(N, 2) * 0.05
            current[:, 0] = np.clip(current[:, 0], 0, W)
            current[:, 1] = np.clip(current[:, 1], 0, H)
    
    return best_positions


def construct():
    """Construct 21 disjoint circles in a perimeter-4 rectangle maximizing sum of radii."""
    t_start = time.time()
    np.random.seed(42)
    
    # Get seeds
    hex_pos, hex_W, hex_H = _seed_hex()
    grid_pos, grid_W, grid_H = _seed_grid()
    
    # Pick the better seed
    hex_val = np.sum(_inradii_fast(hex_pos, hex_W, hex_H))
    grid_val = np.sum(_inradii_fast(grid_pos, grid_W, grid_H))
    
    if hex_val >= grid_val:
        positions, W, H = hex_pos, hex_W, hex_H
    else:
        positions, W, H = grid_pos, grid_W, grid_H
    
    # Optimize
    time_remaining = TIME_LIMIT - (time.time() - t_start) - 10.0
    if time_remaining > 5.0:
        positions = _optimize_positions(positions, W, H, time_remaining)
    
    # Create circles
    circles = _make_circles(positions, W, H)
    
    # Validate
    valid, val = _validate_and_compute(circles)
    if not valid:
        # Try to fix by reducing radii slightly
        circles[:, 2] *= 0.99
        valid, val = _validate_and_compute(circles)
        if not valid:
            circles[:, 2] *= 0.95
            valid, val = _validate_and_compute(circles)
    
    if not valid:
        # Fallback to a simple valid configuration
        s = 2.0 / (7 + 3)
        circles = []
        for i in range(7):
            for j in range(3):
                if len(circles) >= N:
                    break
                circles.append([(i + 0.5) * s, (j + 0.5) * s, 0.4 * s])
        circles = np.array(circles[:N])
    
    return circles
