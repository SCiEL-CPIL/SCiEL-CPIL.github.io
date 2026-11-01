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
    if n == 0:
        return 0.0
    boundary_dists = 0.5 - np.abs(c)
    min_boundary = np.min(boundary_dists)
    
    if n < 2:
        return min_boundary
    
    dists = pdist(c, metric='euclidean')
    min_pair_dist = np.min(dists)
    
    return min(min_pair_dist / 2.0, min_boundary)


def _soft_min_radius(c, beta):
    """
    Compute a soft approximation of the minimum radius.
    Uses log-sum-exp to approximate the minimum of:
    - distances to boundaries
    - half pairwise distances
    """
    # Boundary terms
    b_left = 0.5 - c[:, 0]
    b_right = 0.5 + c[:, 0]
    b_bottom = 0.5 - c[:, 1]
    b_top = 0.5 + c[:, 1]
    
    # Pairwise terms
    d = pdist(c, metric='euclidean')
    p_terms = 0.5 * d
    
    # Combine all terms
    b_terms = np.concatenate([b_left, b_right, b_bottom, b_top])
    all_terms = np.concatenate([b_terms, p_terms])
    
    # Soft min calculation
    m = np.min(all_terms)
    exp_terms = np.exp(-beta * (all_terms - m))
    log_sum = np.log(np.sum(exp_terms))
    
    soft_min_val = m - (1.0 / beta) * log_sum
    
    return soft_min_val


def _initialize_hexagonal():
    """Initialize with a hexagonal pattern."""
    best_c = None
    best_r = 0
    
    for d in np.linspace(0.055, 0.075, 10):
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


def _optimize_smooth(c0, max_time=30.0):
    """Optimize using L-BFGS-B on the soft-min objective."""
    start_time = time.time()
    
    # Bounds: [-0.5, 0.5] for each coordinate
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    # Beta schedule: Start low, end high
    betas = [5.0, 10.0, 20.0, 50.0, 100.0, 200.0]
    
    for beta in betas:
        if time.time() > start_time + max_time - 5:
            break
            
        # Define objective with current beta
        def obj_func(x, beta=beta):
            c = x.reshape(-1, 2)
            r = _soft_min_radius(c, beta)
            return -r
            
        res = minimize(
            obj_func, 
            best_c.flatten(), 
            method='L-BFGS-B', 
            bounds=bounds, 
            options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 100}
        )
        
        c_opt = res.x.reshape(-1, 2)
        r_opt = _compute_radius(c_opt)
        
        if r_opt > best_r:
            best_r = r_opt
            best_c = c_opt.copy()
            
    return best_c, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    best_radius = 0.0
    best_c = None
    
    # Start with hexagonal initialization
    c_hex = _initialize_hexagonal()
    r_hex = _compute_radius(c_hex)
    if r_hex > best_radius:
        best_radius = r_hex
        best_c = c_hex.copy()
    
    # Smooth optimization with L-BFGS-B
    if time.time() < deadline - 10:
        max_time_pgo = min(120.0, deadline - time.time() - 5)
        c_opt, r_opt = _optimize_smooth(c_hex, max_time=max_time_pgo)
        if r_opt > best_radius:
            best_radius = r_opt
            best_c = c_opt.copy()
    
    # If we have time, try a second start from a slightly perturbed hexagonal grid
    if time.time() < deadline - 20:
        np.random.seed(42)
        c_pert = best_c + np.random.normal(0, 0.005, size=(N, 2))
        c_pert = np.clip(c_pert, -0.5, 0.5)
        
        max_time_pgo2 = min(50.0, deadline - time.time() - 5)
        c_opt2, r_opt2 = _optimize_smooth(c_pert, max_time=max_time_pgo2)
        if r_opt2 > best_radius:
            best_radius = r_opt2
            best_c = c_opt2.copy()
    
    if best_c is None:
        best_c = _initialize_hexagonal()
    
    # Final projection to ensure strict validity
    best_c = np.clip(best_c, -0.5, 0.5)
    
    return best_c
# EVOLVE-BLOCK-END
