"""Rewritten program for packing 200 equal circles in a unit square."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
from scipy.spatial import cKDTree


def _compute_radius(c):
    """Compute the largest common radius for given centres."""
    n = c.shape[0]
    if n == 0:
        return 0.0
    boundary_dists = 0.5 - np.abs(c)
    min_boundary = np.min(boundary_dists)
    
    if n < 2:
        return min_boundary
    
    dists = pdist(c, metric='euclidean')
    min_pair_dist = np.min(dists)
    
    return min(min_pair_dist / 2.0, min_boundary)


def _soft_min(values, beta):
    """Smooth approximation of min using log-sum-exp."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return 0.0
    v_min = np.min(values)
    shifted = values - v_min
    # Clip to avoid overflow
    max_shift = 50.0 / beta
    shifted = np.clip(shifted, 0, max_shift)
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    return v_min - log_sum / beta


def _get_active_pairs(c, k=10):
    """
    Get indices of active pairs (nearest neighbors) using KDTree.
    Returns a list of (i, j) tuples where i < j.
    """
    n = c.shape[0]
    tree = cKDTree(c)
    # Query k+1 to include self, so we get k neighbors
    dists, indices = tree.query(c, k=k + 1)
    
    pairs = set()
    for i in range(n):
        for j in indices[i, 1:]:  # Skip self
            if i < j:
                pairs.add((i, j))
            else:
                pairs.add((j, i))
    
    return list(pairs)


def _sparse_objective(params, beta, k_neighbors):
    """
    Objective function using sparse distances (K-NN only).
    Minimizes -soft_min(distances).
    """
    c = np.clip(params.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
    n = c.shape[0]
    
    boundary_dists = 0.5 - np.abs(c)
    
    # Get active pairs
    pairs = _get_active_pairs(c, k_neighbors)
    
    if not pairs:
        # Fallback if no pairs found (should not happen)
        return -np.min(boundary_dists)
    
    # Compute distances for active pairs
    pair_dists = np.zeros(len(pairs))
    for idx, (i, j) in enumerate(pairs):
        diff = c[i] - c[j]
        pair_dists[idx] = np.linalg.norm(diff)
    
    # Combine pair distances (halved) and boundary distances
    # Note: We want to maximize the minimum distance.
    # soft_min approximates the minimum.
    # We minimize -soft_min, so we maximize soft_min.
    
    all_dists = np.concatenate([pair_dists / 2.0, boundary_dists.ravel()])
    
    r_soft = _soft_min(all_dists, beta)
    
    if r_soft <= 0:
        return 1.0
    return -r_soft


def _local_optimize_sparse(c0, max_time=30.0):
    """
    Local optimization using sparse objective with annealing.
    """
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    # Anneal beta from low to high
    # Start with lower beta for exploration, higher for exploitation
    betas = [8.0, 20.0, 50.0, 100.0, 250.0, 500.0, 1000.0]
    k_neighbors = 12  # Number of nearest neighbors to consider
    
    for beta in betas:
        if time.time() > start_time + max_time - 3:
            break
        
        try:
            def obj(params):
                return _sparse_objective(params, beta, k_neighbors)
            
            result = minimize(
                obj,
                best_c.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.5 + 1e-12, 0.5 - 1e-12)] * (2 * N),
                options={'maxiter': 150, 'ftol': 1e-13, 'gtol': 1e-10}
            )
            
            c_new = np.clip(result.x.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
            r_new = _compute_radius(c_new)
            
            if r_new > best_r:
                best_c = c_new
                best_r = r_new
                
        except Exception:
            pass
    
    return best_c, best_r


def _slsqp_active_polish(c0, max_time=20.0):
    """
    High-precision polish using SLSQP with active constraints (K-NN).
    Uses scalar constraint function for better performance.
    """
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    # Try to increase radius slightly
    # We formulate: minimize 0 subject to dist(i,j) >= 2*r_target and boundary >= r_target
    # We iterate on r_target.
    
    # Start with current r
    r_target = best_r
    
    # Try multiple steps of increasing r_target
    for _ in range(4):
        if time.time() > start_time + max_time:
            break
            
        # Get active pairs for current config
        pairs = _get_active_pairs(best_c, k=12)
        
        if not pairs:
            break
            
        n = N
        
        def constraint_func(x):
            """Scalar constraint: min of all distances minus target."""
            c = x.reshape(-1, 2)
            # Boundary distances
            bd = 0.5 - np.abs(c)
            # Pair distances
            min_pair = np.inf
            for (i, j) in pairs:
                diff = c[i] - c[j]
                d = np.linalg.norm(diff)
                if d < min_pair:
                    min_pair = d
            # The minimum of all "radii"
            r_min = min(np.min(bd), min_pair / 2.0)
            return r_min - r_target
        
        try:
            result = minimize(
                lambda x: 0.0,
                best_c.flatten(),
                method='SLSQP',
                bounds=[(-0.5 + 1e-12, 0.5 - 1e-12)] * (2 * n),
                constraints=[
                    {'type': 'ineq', 'fun': constraint_func}
                ],
                options={'maxiter': 80, 'ftol': 1e-12, 'disp': False}
            )
            
            c_new = np.clip(result.x.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
            r_new = _compute_radius(c_new)
            
            if r_new > best_r:
                best_c = c_new
                best_r = r_new
                # Increase target slightly for next iteration
                r_target = best_r * 1.0005
            else:
                break
                
        except Exception:
            break
            
    return best_c, best_r


def _initialize_hexagonal():
    """Initialize with a hexagonal pattern."""
    best_c = None
    best_r = 0
    
    for d in np.linspace(0.055, 0.075, 20):
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
    
    # Fallback: grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()


def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin (evaluator kills at 150s)
    
    best_radius = 0.0
    best_c = None
    
    # Start with hexagonal initialization
    c_hex = _initialize_hexagonal()
    r_hex = _compute_radius(c_hex)
    if r_hex > best_radius:
        best_radius = r_hex
        best_c = c_hex.copy()
    
    # Primary optimization with sparse smooth objective
    if time.time() < deadline - 30:
        remaining = deadline - time.time() - 30
        c_opt, r_opt = _local_optimize_sparse(c_hex, max_time=min(40.0, remaining))
        if r_opt > best_radius:
            best_radius = r_opt
            best_c = c_opt.copy()
    
    # Try perturbed starts with shorter optimization
    np.random.seed(42)
    n_trials = 5
    for trial in range(n_trials):
        if time.time() > deadline - 35:
            break
        
        # Perturb the best solution found so far
        if best_c is not None:
            # Smaller perturbation for later trials
            noise_scale = 0.004 / (trial + 1)
            c_pert = best_c + np.random.normal(0, noise_scale, size=(N, 2))
            c_pert = np.clip(c_pert, -0.5 + 1e-12, 0.5 - 1e-12)
        else:
            c_pert = np.random.uniform(-0.45, 0.45, size=(N, 2))
        
        r_pert = _compute_radius(c_pert)
        if r_pert > best_radius:
            best_radius = r_pert
            best_c = c_pert.copy()
        
        if time.time() < deadline - 30:
            remaining = deadline - time.time() - 30
            c_opt, r_opt = _local_optimize_sparse(c_pert, max_time=min(18.0, remaining))
            if r_opt > best_radius:
                best_radius = r_opt
                best_c = c_opt.copy()
    
    # Final high-precision polish on the best solution found
    if best_c is not None and time.time() < deadline - 5:
        remaining = max(5.0, deadline - time.time() - 5)
        c_final, r_final = _slsqp_active_polish(best_c, max_time=remaining)
        if r_final > best_radius:
            best_radius = r_final
            best_c = c_final
    
    # Additional polish pass if time permits
    if best_c is not None and time.time() < deadline - 3:
        remaining = max(3.0, deadline - time.time() - 3)
        c_final2, r_final2 = _slsqp_active_polish(best_c, max_time=remaining)
        if r_final2 > best_radius:
            best_radius = r_final2
            best_c = c_final2
    
    if best_c is None:
        best_c = _initialize_hexagonal()
    
    best_c = np.clip(best_c, -0.5, 0.5)
    
    return best_c
# EVOLVE-BLOCK-END
