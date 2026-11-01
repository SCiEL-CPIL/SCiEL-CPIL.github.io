"""Rewritten program for packing 200 equal circles in a unit square."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform


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


def _get_active_constraints(c, r_target):
    """Get indices of active (binding) constraints."""
    boundary_dists = 0.5 - np.abs(c)
    active_boundary = np.where(boundary_dists < r_target + 1e-8)[0]
    
    dists = pdist(c, metric='euclidean') / 2.0
    active_pairs = np.where(dists < r_target + 1e-8)[0]
    
    return active_boundary, active_pairs


def _soft_min_radius(c, beta):
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


def _make_objective(beta):
    """Create objective function with given beta."""
    def _objective(params):
        c = np.clip(params.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
        r = _soft_min_radius(c, beta)
        if r <= 0:
            return 1.0
        return -r
    return _objective


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
            # Try different subsets by removing edge points
            c = points[:N].copy()
            r = _compute_radius(c)
            if r > best_r:
                best_r = r
                best_c = c.copy()
            
            # Also try removing corner points first
            if len(points) > N:
                # Sort by distance from center, keep closest N
                centers = np.array([0, 0])
                dists_to_center = np.linalg.norm(points - centers, axis=1)
                sorted_idx = np.argsort(dists_to_center)
                c2 = points[sorted_idx[:N]].copy()
                r2 = _compute_radius(c2)
                if r2 > best_r:
                    best_r = r2
                    best_c = c2.copy()
    
    if best_c is not None:
        return best_c
    
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()


def _active_set_polish(c, r, max_iters=50):
    """Polish using only active constraints for speed."""
    c = c.copy()
    bounds = [(-0.5 + 1e-12, 0.5 - 1e-12)] * (2 * N)
    
    for _ in range(max_iters):
        # Get active constraints
        active_boundary, active_pairs = _get_active_constraints(c, r)
        
        # Build reduced problem
        if len(active_boundary) == 0 and len(active_pairs) == 0:
            break
        
        # Use exact objective but with tighter tolerance
        try:
            result = minimize(
                _objective_exact,
                c.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            
            c_new = np.clip(result.x.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
            r_new = _compute_radius(c_new)
            
            if r_new > r + 1e-15:
                c = c_new
                r = r_new
            else:
                break
        except Exception:
            break
    
    return c, r


def _local_optimize(c0, max_time=25.0):
    """Local optimization using L-BFGS-B with adaptive smooth surrogate then exact polish."""
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    bounds = [(-0.5 + 1e-12, 0.5 - 1e-12)] * (2 * N)
    
    # Phase 1: Optimize smooth surrogate with adaptive beta
    betas = [20.0, 50.0, 100.0]
    for beta in betas:
        if time.time() - start_time > max_time - 15:
            break
        
        try:
            obj = _make_objective(beta)
            result = minimize(
                obj,
                best_c.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 400, 'ftol': 1e-13, 'gtol': 1e-9}
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
                options={'maxiter': 600, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            
            c_new = np.clip(result2.x.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
            r_new = _compute_radius(c_new)
            if r_new > best_r:
                best_c = c_new
                best_r = r_new
        except Exception:
            pass
    
    # Phase 3: Active-set polish
    if time.time() - start_time < max_time - 5:
        try:
            c_active, r_active = _active_set_polish(best_c, best_r, max_iters=30)
            if r_active > best_r:
                best_c = c_active
                best_r = r_active
        except Exception:
            pass
    
    return best_c, best_r


def _polish(c, max_time=15.0):
    """Final polish using perturbation and active-set optimization."""
    start_time = time.time()
    best_c = c.copy()
    best_r = _compute_radius(c)
    bounds = [(-0.5 + 1e-12, 0.5 - 1e-12)] * (2 * N)
    
    # Try multiple small perturbations and optimize
    np.random.seed(42)
    for _ in range(8):
        if time.time() - start_time > max_time - 3:
            break
        
        # Small random perturbation
        c_pert = best_c + np.random.normal(0, 0.0005, size=(N, 2))
        c_pert = np.clip(c_pert, -0.5 + 1e-12, 0.5 - 1e-12)
        
        try:
            result = minimize(
                _objective_exact,
                c_pert.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 400, 'ftol': 1e-14, 'gtol': 1e-11}
            )
            
            c_new = np.clip(result.x.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
            r_new = _compute_radius(c_new)
            if r_new > best_r:
                best_c = c_new
                best_r = r_new
        except Exception:
            pass
    
    # Final active-set polish
    if time.time() - start_time < max_time - 1:
        try:
            c_active, r_active = _active_set_polish(best_c, best_r, max_iters=20)
            if r_active > best_r:
                best_c = c_active
                best_r = r_active
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
        c_opt, r_opt = _local_optimize(c_hex, max_time=min(45.0, deadline - time.time() - 10))
        if r_opt > best_radius:
            best_radius = r_opt
            best_c = c_opt.copy()
    
    # Random restarts with hexagonal perturbation
    np.random.seed(123)
    for trial in range(5):
        if time.time() > deadline - 25:
            break
        
        # Perturbed hexagonal start
        c_rand = c_hex + np.random.normal(0, 0.005, size=(N, 2))
        c_rand = np.clip(c_rand, -0.5 + 1e-12, 0.5 - 1e-12)
        r_rand = _compute_radius(c_rand)
        
        if r_rand > best_radius:
            best_radius = r_rand
            best_c = c_rand.copy()
        
        if time.time() < deadline - 20:
            c_opt, r_opt = _local_optimize(c_rand, max_time=min(25.0, deadline - time.time() - 10))
            if r_opt > best_radius:
                best_radius = r_opt
                best_c = c_opt.copy()
    
    # Final polish on best solution
    if best_c is not None and time.time() < deadline - 5:
        c_polished, r_polished = _polish(best_c, max_time=min(12.0, deadline - time.time() - 2))
        if r_polished > best_radius:
            best_radius = r_polished
            best_c = c_polished.copy()
    
    if best_c is None:
        best_c = _initialize_hexagonal()
    
    best_c = np.clip(best_c, -0.5, 0.5)
    
    return best_c
# EVOLVE-BLOCK-END
