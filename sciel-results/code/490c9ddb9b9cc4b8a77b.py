"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
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


def _objective(params, beta=50.0):
    """Minimize negative soft-min of distances for smoother gradient."""
    c = np.clip(params.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
    
    # Pairwise distances / 2
    pair_dists = pdist(c, metric='euclidean') / 2.0
    # Boundary distances
    bound_dists = (0.5 - np.abs(c)).ravel()
    
    all_dists = np.concatenate([pair_dists, bound_dists])
    all_dists = np.maximum(all_dists, 1e-12)
    
    d_min = np.min(all_dists)
    shifted = all_dists - d_min
    
    if beta > 10000:
        return -d_min
    
    terms = np.exp(-beta * shifted)
    log_sum = np.log(np.sum(terms))
    soft_min_val = d_min - log_sum / beta
    
    return -soft_min_val


def _initialize_hexagonal():
    """Initialize with a hexagonal pattern optimized for the square."""
    best_c = None
    best_r = 0
    
    for d in np.linspace(0.06, 0.08, 20):
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
    
    # Fallback to grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()


def _local_optimize(c0, max_time=25.0):
    """Local optimization using L-BFGS-B with soft-min objective."""
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    # Phase 1: Coarse optimization
    try:
        result = minimize(
            _objective,
            c0.flatten(),
            method='L-BFGS-B',
            bounds=[(-0.5 + 1e-12, 0.5 - 1e-12)] * (2 * N),
            options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 5000}
        )
        
        c_new = np.clip(result.x.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
        r_new = _compute_radius(c_new)
        if r_new > best_r:
            best_c = c_new
            best_r = r_new
    except Exception:
        pass
    
    # Phase 2: Fine optimization if time allows
    if time.time() - start_time < max_time - 5:
        try:
            result2 = minimize(
                _objective,
                best_c.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.5 + 1e-12, 0.5 - 1e-12)] * (2 * N),
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 10000}
            )
            
            c_new = np.clip(result2.x.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
            r_new = _compute_radius(c_new)
            if r_new > best_r:
                best_c = c_new
                best_r = r_new
        except Exception:
            pass
    
    return best_c, best_r


def _perturb(c, scale=0.01):
    """Add small random perturbations."""
    c_pert = c.copy()
    c_pert += np.random.normal(0, scale, size=c.shape)
    return np.clip(c_pert, -0.5 + 1e-12, 0.5 - 1e-12)


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    best_radius = 0.0
    best_c = None
    
    # Initialize with hexagonal pattern
    c_hex = _initialize_hexagonal()
    r_hex = _compute_radius(c_hex)
    if r_hex > best_radius:
        best_radius = r_hex
        best_c = c_hex.copy()
    
    # Optimize from hexagonal start
    if time.time() < deadline - 30:
        c_opt, r_opt = _local_optimize(c_hex, max_time=min(30.0, deadline - time.time() - 10))
        if r_opt > best_radius:
            best_radius = r_opt
            best_c = c_opt.copy()
    
    # Try a few random restarts
    np.random.seed(42)
    for trial in range(5):
        if time.time() > deadline - 25:
            break
        
        c_rand = np.random.uniform(-0.45, 0.45, size=(N, 2))
        r_rand = _compute_radius(c_rand)
        
        if r_rand > best_radius:
            best_radius = r_rand
            best_c = c_rand.copy()
        
        if time.time() < deadline - 20:
            c_opt, r_opt = _local_optimize(c_rand, max_time=min(15.0, deadline - time.time() - 10))
            if r_opt > best_radius:
                best_radius = r_opt
                best_c = c_opt.copy()
    
    # Try perturbations of the best solution
    if time.time() < deadline - 20 and best_c is not None:
        for scale in [0.005, 0.002]:
            if time.time() > deadline - 15:
                break
            c_pert = _perturb(best_c, scale)
            r_pert = _compute_radius(c_pert)
            if r_pert > best_radius:
                best_radius = r_pert
                best_c = c_pert.copy()
            elif time.time() < deadline - 15:
                c_opt, r_opt = _local_optimize(c_pert, max_time=min(10.0, deadline - time.time() - 10))
                if r_opt > best_radius:
                    best_radius = r_opt
                    best_c = c_opt.copy()
    
    if best_c is None:
        best_c = _initialize_hexagonal()
    
    best_c = np.clip(best_c, -0.5, 0.5)
    
    return best_c
# EVOLVE-BLOCK-END
