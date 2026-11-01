"""Rewritten program for packing 200 equal circles in a unit square."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist


def _compute_radius(c):
    """Compute the largest common radius for given centres.
    Returns the radius value. Assumes c is (N,2) and valid."""
    n = c.shape[0]
    # Boundary distance
    boundary_dists = 0.5 - np.abs(c)
    min_boundary = np.min(boundary_dists)
    
    # Pairwise min distance
    if n < 2:
        return 0.5
    
    # Use pdist for efficiency
    dists = pdist(c, metric='euclidean')
    min_pair_dist = np.min(dists)
    
    return min(min_pair_dist / 2.0, min_boundary)


def _compute_radius_fast(c):
    """Faster radius computation that can early-exit."""
    n = c.shape[0]
    boundary_dists = 0.5 - np.abs(c)
    min_boundary = np.min(boundary_dists)
    
    if n < 2:
        return min_boundary
    
    # Use pdist but check if we can early exit
    # For now, just compute it
    dists = pdist(c, metric='euclidean')
    min_pair_dist = np.min(dists)
    
    return min(min_pair_dist / 2.0, min_boundary)


def _objective(params, c_ref=None):
    """Objective function: minimize negative radius.
    If c_ref is provided, add a small regularization to avoid degenerate solutions."""
    c = np.clip(params.reshape(-1, 2), -0.5 + 1e-10, 0.5 - 1e-10)
    radius = _compute_radius(c)
    if radius <= 0:
        return 1.0
    return -radius


def _initialize_grid():
    """Initialize with a grid pattern."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()


def _initialize_hexagonal():
    """Initialize with a hexagonal pattern."""
    # For 200 circles, use a hexagonal lattice
    # Optimal hexagonal packing has spacing d, rows offset by d/2, row spacing d*sqrt(3)/2
    # We want to fit as many as possible in [-0.5, 0.5]^2
    
    # Start with a reasonable spacing
    # Area per circle ~ (sqrt(3)/2) * d^2 for hex packing
    # 200 circles in area 1: d^2 ~ 1/(200 * sqrt(3)/2) = 1/(100*sqrt(3)) ~ 0.00577
    # d ~ 0.076
    # But we need to fit in the square, so let's try different spacings
    
    best_c = None
    best_r = 0
    
    for d in np.linspace(0.04, 0.08, 20):
        # Generate hexagonal points
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
    return _initialize_grid()


def _local_optimize(c0, max_time=30.0, tol=1e-9):
    """Local optimization using gradient-free methods."""
    from scipy.optimize import minimize
    
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    # Use Nelder-Mead for local optimization
    # But Nelder-Mead is slow for high dimensions. Let's use a different approach.
    
    # Instead, let's use a simple iterative approach:
    # 1. Find the closest pair
    # 2. Push them apart
    # 3. Repeat
    
    for iteration in range(10000):
        if time.time() - start_time > max_time:
            break
        
        c = best_c.copy()
        n = c.shape[0]
        
        # Find closest pair
        dists = pdist(c, metric='euclidean')
        min_dist = np.min(dists)
        
        # Find which pair
        # pdist returns condensed form, so we need to map back
        # For simplicity, let's use a different approach
        # Actually, let's use scipy's cKDTree for nearest neighbors
        
        from scipy.spatial import cKDTree
        tree = cKDTree(c)
        # Find nearest neighbor for each point
        # But we need the closest pair overall
        
        # Alternative: just use the fact that we have the min distance
        # and do a simple repulsion
        
        # Compute all pairwise distances and find the minimum
        # For efficiency, let's compute the gradient of the objective
        
        # Actually, let's use a simpler approach: simulated annealing-like
        # Move random points slightly and accept if it improves
        
        # Pick a random point
        i = np.random.randint(n)
        # Try a small random perturbation
        delta = np.random.randn(2) * 0.001
        new_c = c.copy()
        new_c[i] = np.clip(new_c[i] + delta, -0.5 + 1e-10, 0.5 - 1e-10)
        
        new_r = _compute_radius(new_c)
        if new_r > best_r + tol:
            best_c = new_c
            best_r = new_r
    
    return best_c, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin for safety
    
    best_radius = 0.0
    best_c = None
    
    # Try multiple initializations
    inits = []
    
    # Grid initialization
    c_grid = _initialize_grid()
    inits.append(c_grid)
    
    # Hexagonal initialization
    c_hex = _initialize_hexagonal()
    if not np.allclose(c_hex, c_grid):
        inits.append(c_hex)
    
    # Random initialization with some structure
    np.random.seed(42)
    for _ in range(3):
        c_rand = np.random.uniform(-0.45, 0.45, size=(N, 2))
        inits.append(c_rand)
    
    # For each initialization, try to optimize
    for c_init in inits:
        if time.time() > deadline - 15:
            break
        
        r_init = _compute_radius(c_init)
        if r_init > best_radius:
            best_radius = r_init
            best_c = c_init.copy()
        
        # Local optimization
        c_opt, r_opt = _local_optimize(c_init, max_time=min(30.0, deadline - time.time() - 5))
        if r_opt > best_radius:
            best_radius = r_opt
            best_c = c_opt.copy()
    
    # Ensure validity
    if best_c is None:
        best_c = _initialize_grid()
    
    # Final clip
    best_c = np.clip(best_c, -0.5, 0.5)
    
    # Verify
    r_final = _compute_radius(best_c)
    
    return best_c
# EVOLVE-BLOCK-END
