"""Improved packing of 50 equal circles in a unit square."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 50
SQUARE_HALF = 0.5

def _compute_radius(centers):
    """Compute the valid packing radius for given centers."""
    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Check boundaries
    min_dist_to_wall = np.min(np.minimum(
        np.minimum(SQUARE_HALF - centers[:, 0], SQUARE_HALF - centers[:, 1]),
        np.minimum(centers[:, 0] + SQUARE_HALF, centers[:, 1] + SQUARE_HALF)
    ))
    if min_dist_to_wall < 0:
        return 0.0
    # Check pairwise distances
    d = cdist(centers, centers)
    np.fill_diagonal(d, np.inf)
    min_pair_dist = np.min(d)
    if min_pair_dist <= 0:
        return 0.0
    return min(min_pair_dist / 2.0, min_dist_to_wall)

def _objective(centers):
    """Objective to minimize: negative of the packing radius (smoothed)."""
    centers = np.asarray(centers, dtype=np.float64).reshape(-1, 2)
    # Soft min using log-sum-exp for smoothness
    d = cdist(centers, centers)
    np.fill_diagonal(d, 1e10)
    
    # Wall distances
    wall_dists = np.stack([
        SQUARE_HALF - centers[:, 0],
        SQUARE_HALF + centers[:, 0],
        SQUARE_HALF - centers[:, 1],
        SQUARE_HALF + centers[:, 1]
    ], axis=-1)
    
    # Combine all distances
    all_dists = np.concatenate([d.flatten(), wall_dists.flatten()])
    
    # Use soft-min: -log-sum-exp(-dists/beta) / beta approximates min(dists)
    beta = 0.01
    neg_dists = -all_dists / beta
    max_neg = np.max(neg_dists)
    log_sum_exp = max_neg + np.log(np.sum(np.exp(neg_dists - max_neg)))
    soft_min = -beta * log_sum_exp
    
    # We want to maximize the minimum distance, so minimize -soft_min
    # But we also need to handle the /2 for pairwise
    # Actually, let's just use a simpler approach: minimize the negative of the actual radius
    # For L-BFGS-B, we need a smooth objective
    
    # Simpler: just return the negative of the min distance (non-smooth but works with L-BFGS-B)
    min_pair = np.min(d) / 2.0
    min_wall = np.min(wall_dists)
    return -min(min_pair, min_wall)

def _objective_smooth(centers, beta=0.02):
    """Smooth objective using log-sum-exp."""
    centers = np.asarray(centers, dtype=np.float64).reshape(-1, 2)
    d = cdist(centers, centers)
    np.fill_diagonal(d, 1e10)
    
    wall_dists = np.stack([
        SQUARE_HALF - centers[:, 0],
        SQUARE_HALF + centers[:, 0],
        SQUARE_HALF - centers[:, 1],
        SQUARE_HALF + centers[:, 1]
    ], axis=-1)
    
    # For pairwise, we need to divide by 2
    pair_dists = d / 2.0
    np.fill_diagonal(pair_dists, 1e10)
    
    all_dists = np.concatenate([pair_dists.flatten(), wall_dists.flatten()])
    
    # Soft min
    neg_dists = -all_dists / beta
    max_neg = np.max(neg_dists)
    if not np.isfinite(max_neg):
        return 0.0
    log_sum_exp = max_neg + np.log(np.sum(np.exp(neg_dists - max_neg)))
    soft_min = -beta * log_sum_exp
    
    return -soft_min

def _local_search(centers, beta=0.02, max_iter=200):
    """Run L-BFGS-B local search from given centers."""
    bounds = [(-SQUARE_HALF + 0.001, SQUARE_HALF - 0.001)] * (2 * N)
    x0 = centers.flatten()
    
    result = minimize(
        _objective_smooth,
        x0,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8, 'beta': beta}
    )
    
    return result.x.reshape(N, 2)

def _grid_init():
    """Create initial grid placement."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()

def _hex_init():
    """Create initial hexagonal packing."""
    # Hexagonal packing in a square
    points = []
    r = 0.07  # approximate radius
    dx = 2 * r
    dy = 2 * r * np.sqrt(3) / 2
    
    row = 0
    while len(points) < N:
        offset = 0 if row % 2 == 0 else r
        col = 0
        while len(points) < N:
            x = -SQUARE_HALF + r + offset + col * dx
            y = -SQUARE_HALF + r + row * dy
            if -SQUARE_HALF + r <= x <= SQUARE_HALF - r and -SQUARE_HALF + r <= y <= SQUARE_HALF - r:
                points.append((x, y))
            col += 1
        row += 1
    
    return np.array(points[:N])

def construct():
    """Construct the circle centers."""
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    # Start with multiple initializations
    best_centers = None
    best_radius = 0.0
    
    # Try different initializations
    inits = [
        _grid_init(),
        _hex_init(),
    ]
    
    # Add random perturbations to grid
    rng = np.random.default_rng(42)
    for _ in range(5):
        perturbed = _grid_init() + rng.uniform(-0.02, 0.02, size=(N, 2))
        # Clip to valid range
        perturbed = np.clip(perturbed, -SQUARE_HALF + 0.001, SQUARE_HALF - 0.001)
        inits.append(perturbed)
    
    for init in inits:
        if time.time() > deadline - 10:
            break
        
        # Run local search
        centers = _local_search(init, beta=0.02, max_iter=150)
        radius = _compute_radius(centers)
        
        if radius > best_radius:
            best_radius = radius
            best_centers = centers.copy()
    
    # Try to improve with smaller beta (sharper)
    if best_centers is not None and time.time() < deadline - 5:
        for beta in [0.01, 0.005]:
            centers = _local_search(best_centers, beta=beta, max_iter=100)
            radius = _compute_radius(centers)
            if radius > best_radius:
                best_radius = radius
                best_centers = centers.copy()
    
    # Final refinement with very small beta
    if best_centers is not None and time.time() < deadline - 2:
        centers = _local_search(best_centers, beta=0.002, max_iter=50)
        radius = _compute_radius(centers)
        if radius > best_radius:
            best_radius = radius
            best_centers = centers.copy()
    
    if best_centers is None:
        # Fallback to grid
        best_centers = _grid_init()
    
    return best_centers
