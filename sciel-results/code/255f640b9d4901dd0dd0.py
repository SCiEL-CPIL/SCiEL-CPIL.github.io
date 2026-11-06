"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

# Deadline guard: kill after 130s to be safe within 120s limit (with margin)
DEADLINE_MARGIN = 15.0


def _compute_radius(centres):
    """Compute the maximum common radius for a given set of centres."""
    if len(centres) < 2:
        return 0.0
    # Minimum distance to walls
    wall_min = min(0.5 - np.min(centres[:, 0]), 0.5 - np.max(centres[:, 0]),
                   0.5 - np.min(centres[:, 1]), 0.5 - np.max(centres[:, 1]))
    # Minimum pairwise distance / 2
    dists = pdist(centres)
    min_pair_dist = np.min(dists) if len(dists) > 0 else 1.0
    return min(wall_min, min_pair_dist / 2.0)


def _energy(centres, r_target):
    """
    Energy function to minimize: penalize pairs closer than 2*r_target
    and wall violations. Returns a scalar cost.
    """
    x = centres[:, 0]
    y = centres[:, 1]
    
    # Wall penalty: distance to each wall should be >= r_target
    dx_min = x - (-0.5)
    dx_max = 0.5 - x
    dy_min = y - (-0.5)
    dy_max = 0.5 - y
    
    # For each circle, the minimum distance to any wall
    dist_to_walls = np.minimum(
        np.minimum(dx_min, dx_max),
        np.minimum(dy_min, dy_max)
    )
    # Penalty for being too close to walls
    wall_violation = np.maximum(0.0, r_target - dist_to_walls)
    wall_energy = np.sum(wall_violation ** 2)
    
    # Pairwise penalty: distance between circles should be >= 2*r_target
    dists = pdist(centres)
    pair_violation = np.maximum(0.0, 2.0 * r_target - dists)
    pair_energy = np.sum(pair_violation ** 2)
    
    return pair_energy + wall_energy


def _initial_grid():
    """Create initial grid placement."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()


def _hexagonal_initial():
    """Create hexagonal lattice initial placement."""
    # Hexagonal packing has better density than square grid
    # For N=200, use a hexagonal arrangement
    # Side length of hexagonal packing to fit 200 circles
    # Approximate: use a grid with alternating rows offset
    
    # Use a rectangular grid with hexagonal offset
    # Number of rows and columns to approximate 200
    m = int(np.ceil(np.sqrt(N)))
    
    # Hexagonal spacing: vertical spacing is s, horizontal is s/sqrt(3)
    # Actually, for hexagonal packing:
    # Row 0: x = 0, s, 2s, ...
    # Row 1: x = s/2, 3s/2, 5s/2, ...
    # Vertical spacing: s * sqrt(3)/2
    
    # Let's use a simpler approach: create a grid and then perturb
    # to form hexagonal pattern
    
    # Alternative: use a dense hexagonal lattice clipped to square
    # Generate hexagonal points in a larger area, then clip
    
    # Hexagonal lattice parameters
    # a = side length of hexagon
    # Points at (i*a + j*a/2, j*a*sqrt(3)/2) for integer i, j
    
    # We want to fit 200 points in [-0.5, 0.5]^2
    # Let's estimate a: area per point in hexagonal packing = a^2 * sqrt(3)/2
    # Total area = 1, so a^2 * sqrt(3)/2 * 200 ≈ 1
    # a^2 ≈ 1 / (200 * sqrt(3)/2) = 1 / (100*sqrt(3)) ≈ 0.00577
    # a ≈ 0.076
    
    a = 0.076
    points = []
    max_j = int(np.ceil(1.0 / (a * np.sqrt(3) / 2))) + 2
    for j in range(-max_j, max_j + 1):
        y = j * a * np.sqrt(3) / 2
        if -0.5 <= y <= 0.5:
            offset = (j % 2) * a / 2
            max_i = int(np.ceil(1.0 / a)) + 2
            for i in range(-max_i, max_i + 1):
                x = i * a + offset
                if -0.5 <= x <= 0.5:
                    points.append([x, y])
    
    points = np.array(points)
    if len(points) >= N:
        # Randomly select N points to avoid bias
        idx = np.random.choice(len(points), N, replace=False)
        return points[idx].copy()
    else:
        # Fallback to grid
        return _initial_grid()


def _random_initial():
    """Create random initial placement inside the square."""
    return (np.random.rand(N, 2) - 0.5) * 0.9


def _local_search(centres, r_target, deadline):
    """
    Perform local search using L-BFGS-B to minimize energy
    with a target radius r_target.
    """
    start_time = time.time()
    
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    def objective(x):
        if time.time() > deadline:
            return np.inf
        c = x.reshape(-1, 2)
        return _energy(c, r_target)
    
    try:
        result = minimize(
            objective,
            centres.flatten(),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'maxfun': 20000, 'ftol': 1e-12}
        )
        if result.fun < np.inf:
            return result.x.reshape(-1, 2)
    except Exception:
        pass
    
    return centres


def _binary_search_radius(centres, deadline):
    """
    Use binary search on radius to find the maximum feasible radius
    through iterative local search.
    """
    # Initial radius estimate
    r_low = 0.03
    r_high = 0.04
    
    # First, try to find a feasible radius
    best_centres = centres.copy()
    best_radius = _compute_radius(best_centres)
    
    # Iterative approach: increase r_target and optimize
    r_target = max(best_radius * 1.05, 0.032)
    
    for iteration in range(20):
        if time.time() > deadline - 5:
            break
        
        # Try to find a configuration that works with r_target
        result = _local_search(best_centres, r_target, deadline - 5)
        
        # Check if the result is feasible
        current_radius = _compute_radius(result)
        
        if current_radius >= r_target * 0.99:
            # Success: we can achieve at least r_target
            best_centres = result.copy()
            best_radius = current_radius
            # Increase target
            r_target *= 1.03
        else:
            # Failure: decrease target slightly
            r_target *= 0.97
            # Use the result as new starting point anyway
            best_centres = result.copy()
    
    return best_centres, best_radius


def construct():
    """Construct 200 circle centres in the unit square to maximize common radius."""
    start_time = time.time()
    deadline = start_time + 120.0 - DEADLINE_MARGIN
    
    np.random.seed(42)
    
    best_centres = _initial_grid()
    best_radius = _compute_radius(best_centres)
    
    # Strategy: multiple restarts with different initializations
    # 1. Square grid
    # 2. Hexagonal lattice
    # 3. Random
    # 4. Perturbed grid
    # 5. Perturbed hexagonal
    
    initializations = [
        _initial_grid(),
        _hexagonal_initial(),
        _random_initial(),
        _initial_grid() + np.random.randn(N, 2) * 0.01,
        _hexagonal_initial() + np.random.randn(N, 2) * 0.005,
    ]
    
    # Clip all initializations to valid region
    for i, init in enumerate(initializations):
        init[:, 0] = np.clip(init[:, 0], -0.49, 0.49)
        init[:, 1] = np.clip(init[:, 1], -0.49, 0.49)
        initializations[i] = init
    
    for restart in range(len(initializations)):
        if time.time() > deadline - 10:
            break
        
        centres = initializations[restart].copy()
        
        # Run binary search radius optimization
        centres, radius = _binary_search_radius(centres, deadline - 10)
        
        if radius > best_radius:
            best_radius = radius
            best_centres = centres.copy()
        
        # If time permits, do additional refinement
        if time.time() < deadline - 5 and restart < len(initializations) - 1:
            # Use the best found so far as a new starting point
            pass
    
    # Final refinement: try to improve the best result with a slightly higher target
    if time.time() < deadline - 3:
        r_target = best_radius * 1.02
        result = _local_search(best_centres, r_target, deadline - 3)
        final_radius = _compute_radius(result)
        if final_radius > best_radius:
            best_centres = result.copy()
            best_radius = final_radius
    
    # Ensure all centres are strictly inside the square
    best_centres[:, 0] = np.clip(best_centres[:, 0], -0.5 + 1e-10, 0.5 - 1e-10)
    best_centres[:, 1] = np.clip(best_centres[:, 1], -0.5 + 1e-10, 0.5 - 1e-10)
    
    return best_centres
# EVOLVE-BLOCK-END
