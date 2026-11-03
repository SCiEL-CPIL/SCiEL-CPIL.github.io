"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

START_TIME = time.time()
TIME_LIMIT = 115.0  # Leave 5s margin for the 120s limit


def _compute_radius(c):
    """Compute the maximum radius for given centers."""
    # Distance to boundaries
    boundary_dist = 0.5 - np.abs(c).max(axis=1)
    # Pairwise distances
    d = cdist(c, c)
    np.fill_diagonal(d, np.inf)
    min_pair_dist = d.min()
    return min(boundary_dist.min(), min_pair_dist / 2.0)


def _softmin_neg(c, beta=1000.0):
    """Negative soft-min of constraints (pairwise and boundary)."""
    c = c.reshape(-1, 2)
    # Boundary constraints
    boundary = 0.5 - np.abs(c).max(axis=1)
    # Pairwise distances
    d = cdist(c, c)
    np.fill_diagonal(d, np.inf)
    min_pair = d.min()
    # Use soft-min approximation: -log-sum-exp
    all_vals = np.concatenate([boundary, min_pair / 2.0 * np.ones(1)])
    # Soft min: approximate min by -1/beta * log(sum(exp(-beta * x)))
    max_val = all_vals.max()
    soft_min = max_val - np.log(np.sum(np.exp(-beta * (all_vals - max_val)))) / beta
    return -soft_min


def _initial_hex_grid(n):
    """Generate a hexagonal grid of n points in [-0.5, 0.5]^2."""
    points = []
    rows = int(np.ceil(np.sqrt(n * 2 / np.sqrt(3))))
    s = 1.0 / np.sqrt(n) * 1.1
    
    for i in range(rows + 10):
        row_y = -0.5 + (i + 0.5) * s * np.sqrt(3) / 2
        if row_y > 0.5:
            break
        offset = s / 2 if i % 2 == 0 else 0
        cols = int(np.floor(1.0 / s)) + 1
        for j in range(cols + 10):
            row_x = -0.5 + (j + 0.5) * s + offset
            if -0.5 <= row_x <= 0.5 and -0.5 <= row_y <= 0.5:
                points.append([row_x, row_y])
    
    points = np.array(points)
    
    if len(points) > n:
        indices = np.random.choice(len(points), n, replace=False)
        points = points[indices]
    
    while len(points) < n:
        points = np.vstack([points, np.random.uniform(-0.45, 0.45, size=(1, 2))])
    
    return points[:n]


def _initial_square_grid(n):
    """Generate a square grid of n points in [-0.5, 0.5]^2."""
    m = int(np.ceil(np.sqrt(n)))
    while m * m < n:
        m += 1
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:n]


def _grid_search(c_init, time_budget=60.0):
    """Discrete greedy search on fine grids with random offsets."""
    best_c = c_init.copy()
    best_r = _compute_radius(c_init)
    start = time.time()
    
    # Try different grid resolutions and offsets
    for resolution in [100, 200, 400]:
        if time.time() - start > time_budget:
            break
        
        # Generate grid points
        grid_size = resolution
        step = 1.0 / grid_size
        xs = -0.5 + np.arange(grid_size + 1) * step
        ys = -0.5 + np.arange(grid_size + 1) * step
        
        # Try several random offsets
        n_offsets = 5
        for offset_idx in range(n_offsets):
            if time.time() - start > time_budget:
                break
            
            # Random offset within one grid cell
            ox = np.random.uniform(-step/2, step/2)
            oy = np.random.uniform(-step/2, step/2)
            
            # Greedy: for each point, find best grid position
            c_trial = best_c.copy()
            improved = True
            max_iter = 10
            
            while improved and time.time() - start < time_budget:
                improved = False
                # Try to improve each point
                for i in range(N):
                    if time.time() - start > time_budget:
                        break
                    
                    # Current position
                    cx, cy = c_trial[i]
                    # Find nearest grid point
                    gx_idx = int(np.round((cx - ox + 0.5) / step))
                    gy_idx = int(np.round((cy - oy + 0.5) / step))
                    
                    # Try neighboring grid points (3x3 neighborhood)
                    best_local_r = best_r
                    best_local_pos = None
                    
                    for dx in range(-1, 2):
                        for dy in range(-1, 2):
                            ngx = gx_idx + dx
                            ngy = gy_idx + dy
                            if 0 <= ngx <= grid_size and 0 <= ngy <= grid_size:
                                new_x = -0.5 + ngx * step + ox
                                new_y = -0.5 + ngy * step + oy
                                if -0.5 <= new_x <= 0.5 and -0.5 <= new_y <= 0.5:
                                    c_trial[i] = [new_x, new_y]
                                    r_trial = _compute_radius(c_trial)
                                    if r_trial > best_local_r:
                                        best_local_r = r_trial
                                        best_local_pos = [new_x, new_y]
                    
                    if best_local_pos is not None:
                        c_trial[i] = best_local_pos
                        if best_local_r > best_r:
                            best_r = best_local_r
                            best_c = c_trial.copy()
                            improved = True
                
                if not improved:
                    break
            
            # If we improved, continue; else try next offset
    
    return best_c, best_r


def _refine(c, time_budget=30.0):
    """Refine positions using L-BFGS-B with smooth objective."""
    best_c = c.copy()
    best_r = _compute_radius(c)
    start = time.time()
    
    beta = 500.0
    for _ in range(5):
        if time.time() - start > time_budget:
            break
        
        def obj(x):
            return _softmin_neg(x, beta)
        
        try:
            res = minimize(
                obj,
                best_c.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (2 * N),
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            new_c = res.x.reshape(-1, 2)
            new_r = _compute_radius(new_c)
            if new_r > best_r:
                best_r = new_r
                best_c = new_c
        except Exception:
            pass
        
        beta *= 2
    
    return best_c, best_r


def _refine_local(c, time_budget=20.0):
    """Local refinement using coordinate descent / steepest ascent."""
    best_c = c.copy()
    best_r = _compute_radius(c)
    start = time.time()
    
    step_size = 0.001
    for iteration in range(5000):
        if time.time() - start > time_budget:
            break
        
        c_trial = best_c.copy()
        n_move = min(5, N)
        idx = np.random.choice(N, n_move, replace=False)
        c_trial[idx] += np.random.normal(0, step_size, size=(n_move, 2))
        c_trial = np.clip(c_trial, -0.5 + 1e-10, 0.5 - 1e-10)
        
        r_trial = _compute_radius(c_trial)
        if r_trial > best_r:
            best_r = r_trial
            best_c = c_trial
            step_size *= 1.1
        else:
            step_size *= 0.9
            step_size = max(step_size, 1e-6)
    
    return best_c, best_r


def construct():
    """Construct 200 circle centers in [-0.5, 0.5]^2 maximizing minimum radius."""
    global START_TIME, TIME_LIMIT
    
    # Generate multiple initial candidates
    candidates = []
    
    # Square grid
    c_square = _initial_square_grid(N)
    candidates.append(c_square)
    
    # Hex grid
    np.random.seed(42)
    c_hex = _initial_hex_grid(N)
    candidates.append(c_hex)
    
    # Slightly perturbed square grid
    np.random.seed(123)
    c_perturbed = c_square + np.random.normal(0, 0.01, size=(N, 2))
    c_perturbed = np.clip(c_perturbed, -0.49, 0.49)
    candidates.append(c_perturbed)
    
    # Evaluate initial candidates
    best_c = None
    best_r = 0
    for c in candidates:
        r = _compute_radius(c)
        if r > best_r:
            best_r = r
            best_c = c
    
    # Phase 1: Grid search on best candidate
    elapsed = time.time() - START_TIME
    time_remaining = TIME_LIMIT - elapsed - 30  # Leave 30s for polish
    
    if time_remaining > 15:
        np.random.seed(99)
        c_grid, r_grid = _grid_search(best_c, time_budget=min(time_remaining, 60.0))
        if r_grid > best_r:
            best_r = r_grid
            best_c = c_grid
    
    # Phase 2: Smooth optimization polish
    elapsed = time.time() - START_TIME
    time_remaining = TIME_LIMIT - elapsed - 10
    
    if time_remaining > 5:
        c_refined, r_refined = _refine(best_c, time_budget=min(time_remaining, 25.0))
        if r_refined > best_r:
            best_r = r_refined
            best_c = c_refined
    
    # Phase 3: Final local refinement
    elapsed = time.time() - START_TIME
    time_remaining = TIME_LIMIT - elapsed - 5
    
    if time_remaining > 3:
        c_final, r_final = _refine_local(best_c, time_budget=min(time_remaining, 10.0))
        if r_final > best_r:
            best_r = r_final
            best_c = c_final
    
    # Ensure valid output
    best_c = np.clip(best_c, -0.5 + 1e-12, 0.5 - 1e-12)
    
    if not np.all(np.isfinite(best_c)):
        return _initial_square_grid(N)
    
    return best_c
# EVOLVE-BLOCK-END
