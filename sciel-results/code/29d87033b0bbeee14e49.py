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


def _soft_min_radius(c, beta=50.0):
    """Compute a smooth approximation of the minimum radius using log-sum-exp."""
    n = c.shape[0]
    
    # Boundary distances
    boundary_dists = 0.5 - np.abs(c)
    
    # Pairwise distances / 2
    dists = pdist(c, metric='euclidean') / 2.0
    
    # Combine all constraints into one array
    all_constraints = np.concatenate([boundary_dists.flatten(), dists])
    
    # Soft-min: -log-sum-exp(-beta * x) / beta
    # For numerical stability, subtract max
    x = all_constraints
    x_max = np.max(x)
    soft_min = x_max - np.log(np.sum(np.exp(-beta * (x - x_max)))) / beta
    
    return soft_min


def _objective(params):
    """Minimize negative soft-min radius (smooth surrogate)."""
    c = np.clip(params.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
    r = _soft_min_radius(c, beta=50.0)
    if r <= 0:
        return 1.0
    return -r


def _objective_exact(params):
    """Minimize negative exact radius (for final polish)."""
    c = np.clip(params.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
    r = _compute_radius(c)
    if r <= 0:
        return 1.0
    return -r


def _initialize_hexagonal():
    """Initialize with a hexagonal pattern optimized for the square."""
    best_c = None
    best_r = 0
    
    for d in np.linspace(0.06, 0.08, 15):
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


def _local_optimize(c0, max_time=25.0):
    """Local optimization using L-BFGS-B with smooth surrogate then exact polish."""
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    bounds = [(-0.5 + 1e-12, 0.5 - 1e-12)] * (2 * N)
    
    # Phase 1: Optimize smooth surrogate
    try:
        result = minimize(
            _objective,
            c0.flatten(),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        c_new = np.clip(result.x.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
        r_new = _compute_radius(c_new)
        if r_new > best_r:
            best_c = c_new
            best_r = r_new
    except Exception:
        pass
    
    # Phase 2: Polish with exact objective
    if time.time() - start_time < max_time - 10:
        try:
            result2 = minimize(
                _objective_exact,
                best_c.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            
            c_new = np.clip(result2.x.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
            r_new = _compute_radius(c_new)
            if r_new > best_r:
                best_c = c_new
                best_r = r_new
        except Exception:
            pass
    
    # Phase 3: Final high-precision polish
    if time.time() - start_time < max_time - 3:
        try:
            result3 = minimize(
                _objective_exact,
                best_c.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-12}
            )
            
            c_new = np.clip(result3.x.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
            r_new = _compute_radius(c_new)
            if r_new > best_r:
                best_c = c_new
                best_r = r_new
        except Exception:
            pass
    
    return best_c, best_r


def _polish(c, max_time=15.0):
    """Final polish using alternating coordinate updates and local optimization."""
    start_time = time.time()
    best_c = c.copy()
    best_r = _compute_radius(c)
    bounds = [(-0.5 + 1e-12, 0.5 - 1e-12)] * (2 * N)
    
    # Try multiple small perturbations and optimize
    np.random.seed(42)
    for _ in range(5):
        if time.time() - start_time > max_time - 3:
            break
        
        # Small random perturbation
        c_pert = best_c + np.random.normal(0, 0.001, size=(N, 2))
        c_pert = np.clip(c_pert, -0.5 + 1e-12, 0.5 - 1e-12)
        
        try:
            result = minimize(
                _objective_exact,
                c_pert.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            
            c_new = np.clip(result.x.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
            r_new = _compute_radius(c_new)
            if r_new > best_r:
                best_c = c_new
                best_r = r_new
        except Exception:
            pass
    
    return best_c, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    best_radius = 0.0
    best_c = None
    
    c_hex = _initialize_hexagonal()
    r_hex = _compute_radius(c_hex)
    if r_hex > best_radius:
        best_radius = r_hex
        best_c = c_hex.copy()
    
    # Primary optimization from hexagonal start
    if time.time() < deadline - 20:
        c_opt, r_opt = _local_optimize(c_hex, max_time=min(40.0, deadline - time.time() - 10))
        if r_opt > best_radius:
            best_radius = r_opt
            best_c = c_opt.copy()
    
    # Random restarts
    np.random.seed(123)
    for trial in range(4):
        if time.time() > deadline - 25:
            break
        
        c_rand = np.random.uniform(-0.45, 0.45, size=(N, 2))
        r_rand = _compute_radius(c_rand)
        
        if r_rand > best_radius:
            best_radius = r_rand
            best_c = c_rand.copy()
        
        if time.time() < deadline - 20:
            c_opt, r_opt = _local_optimize(c_rand, max_time=min(20.0, deadline - time.time() - 10))
            if r_opt > best_radius:
                best_radius = r_opt
                best_c = c_opt.copy()
    
    # Final polish on best solution
    if best_c is not None and time.time() < deadline - 5:
        c_polished, r_polished = _polish(best_c, max_time=min(10.0, deadline - time.time() - 2))
        if r_polished > best_radius:
            best_radius = r_polished
            best_c = c_polished.copy()
    
    if best_c is None:
        best_c = _initialize_hexagonal()
    
    best_c = np.clip(best_c, -0.5, 0.5)
    
    return best_c
# EVOLVE-BLOCK-END
