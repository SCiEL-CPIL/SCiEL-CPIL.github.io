"""Rewritten program for packing 200 equal circles in a unit square."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist


def _compute_radius(c):
    """Compute the largest common radius for given centres."""
    n = c.shape[0]
    boundary_dists = 0.5 - np.abs(c)
    min_boundary = np.min(boundary_dists)
    
    if n < 2:
        return min_boundary
    
    dists = pdist(c, metric='euclidean')
    min_pair_dist = np.min(dists)
    
    return min(min_pair_dist / 2.0, min_boundary)


def _objective(params):
    """Minimize negative radius."""
    c = np.clip(params.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
    r = _compute_radius(c)
    if r <= 0:
        return 1.0
    return -r


def _initialize_hexagonal():
    """Initialize with a hexagonal pattern optimized for the square."""
    best_c = None
    best_r = 0
    
    for d in np.linspace(0.055, 0.085, 25):
        points = []
        row_spacing = d * np.sqrt(3) / 2
        x_offset = d / 2
        
        y = -0.5 + d / 2
        row = 0
        while y <= 0.5 - d / 2:
            x = -0.5 + d / 2
            if row % 2 == 1:
                x += x_offset
            while x <= 0.5 - d / 2:
                if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                    points.append([x, y])
                x += d
            y += row_spacing
            row += 1
        
        points = np.array(points)
        if len(points) >= N:
            c = points[:N].copy()
            r = _compute_radius(c)
            if r > best_r:
                best_r = r
                best_c = c.copy()
    
    if best_c is not None:
        return best_c
    
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()


def _initialize_grid():
    """Initialize with a uniform grid pattern."""
    best_c = None
    best_r = 0
    
    for cols in range(13, 17):
        rows = int(np.ceil(N / cols))
        if rows * cols < N:
            continue
        
        x_ticks = (np.arange(cols) + 0.5) / cols - 0.5
        y_ticks = (np.arange(rows) + 0.5) / rows - 0.5
        
        grid = np.array([(x, y) for y in y_ticks for x in x_ticks])
        if len(grid) >= N:
            c = grid[:N].copy()
            r = _compute_radius(c)
            if best_c is None or r > best_r:
                best_r = r
                best_c = c.copy()
    
    if best_c is not None:
        return best_c
    
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()


def _local_optimize(c0, max_time=25.0):
    """Local optimization using L-BFGS-B with two passes."""
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    # First pass: coarse optimization
    try:
        result = minimize(
            _objective,
            c0.flatten(),
            method='L-BFGS-B',
            bounds=[(-0.5 + 1e-12, 0.5 - 1e-12)] * (2 * N),
            options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        c_new = np.clip(result.x.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
        r_new = _compute_radius(c_new)
        if r_new > best_r:
            best_c = c_new
            best_r = r_new
    except Exception:
        pass
    
    # Second pass: fine optimization if time permits
    elapsed = time.time() - start_time
    if elapsed < max_time - 5:
        try:
            result2 = minimize(
                _objective,
                best_c.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.5 + 1e-12, 0.5 - 1e-12)] * (2 * N),
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            
            c_new = np.clip(result2.x.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
            r_new = _compute_radius(c_new)
            if r_new > best_r:
                best_c = c_new
                best_r = r_new
        except Exception:
            pass
    
    return best_c, best_r


def _perturb_and_optimize(c_base, max_time=20.0):
    """Try multiple perturbations of a base configuration and optimize each."""
    start_time = time.time()
    best_c = c_base.copy()
    best_r = _compute_radius(c_base)
    
    np.random.seed(42)
    max_trials = 3
    
    for trial in range(max_trials):
        elapsed = time.time() - start_time
        if elapsed > max_time - 5:
            break
        
        sigma = 0.005 + 0.003 * trial
        noise = np.random.normal(0, sigma, size=c_base.shape)
        c_pert = np.clip(c_base + noise, -0.5 + 1e-12, 0.5 - 1e-12)
        
        r_pert = _compute_radius(c_pert)
        if r_pert > best_r:
            best_c = c_pert.copy()
            best_r = r_pert
        
        # Optimize this perturbed configuration
        remaining = max_time - (time.time() - start_time) - 2
        if remaining > 5:
            c_opt, r_opt = _local_optimize(c_pert, max_time=remaining)
            if r_opt > best_r:
                best_c = c_opt
                best_r = r_opt
    
    return best_c, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    best_c = None
    best_r = 0.0
    
    # 1. Try hexagonal initialization
    c_hex = _initialize_hexagonal()
    r_hex = _compute_radius(c_hex)
    if r_hex > best_r:
        best_r = r_hex
        best_c = c_hex.copy()
    
    # 2. Try grid initialization
    c_grid = _initialize_grid()
    r_grid = _compute_radius(c_grid)
    if r_grid > best_r:
        best_r = r_grid
        best_c = c_grid.copy()
    
    # 3. Optimize the best initialization
    elapsed = time.time() - start_time
    if elapsed < deadline - 40:
        c_opt, r_opt = _local_optimize(best_c, max_time=min(30.0, deadline - time.time() - 35))
        if r_opt > best_r:
            best_r = r_opt
            best_c = c_opt.copy()
    
    # 4. Try perturbation-based search from the best configuration
    elapsed = time.time() - start_time
    if elapsed < deadline - 20:
        c_pert, r_pert = _perturb_and_optimize(best_c, max_time=min(15.0, deadline - time.time() - 15))
        if r_pert > best_r:
            best_r = r_pert
            best_c = c_pert.copy()
    
    # 5. Try a random restart if time permits
    elapsed = time.time() - start_time
    if elapsed < deadline - 15:
        np.random.seed(123)
        c_rand = np.random.uniform(-0.45, 0.45, size=(N, 2))
        r_rand = _compute_radius(c_rand)
        
        if r_rand > best_r:
            best_r = r_rand
            best_c = c_rand.copy()
        
        remaining = deadline - time.time() - 10
        if remaining > 5:
            c_opt, r_opt = _local_optimize(best_c, max_time=remaining)
            if r_opt > best_r:
                best_r = r_opt
                best_c = c_opt.copy()
    
    # Final safety check
    if best_c is None:
        best_c = c_hex
    
    best_c = np.clip(best_c, -0.5, 0.5)
    return best_c
# EVOLVE-BLOCK-END
