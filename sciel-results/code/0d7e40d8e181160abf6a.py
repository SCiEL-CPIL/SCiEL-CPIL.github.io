"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_radius(centers):
    """Compute the valid radius for given centers."""
    # Check bounds: centers must be in [-0.5, 0.5]^3
    if not np.all(np.isfinite(centers)):
        return 0.0
    if centers.shape != (N, 3):
        return 0.0
    if np.any(np.abs(centers) > 0.5 + 1e-12):
        return 0.0
    
    # Wall clearance
    wall_clearances = 0.5 - np.abs(centers)
    min_wall = np.min(wall_clearances)
    
    if min_wall < -1e-12:
        return 0.0
    
    # Pairwise distances
    # Compute all pairwise distances efficiently
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    r = min(min_wall, min_pair)
    if r < 0:
        return 0.0
    return r

def _soft_min(x, alpha=100.0):
    """Smooth approximation of min using log-sum-exp."""
    # soft_min(x) ≈ -logsumexp(-x / alpha) * alpha
    x = np.asarray(x)
    # For numerical stability
    x_max = np.max(x)
    return -alpha * np.log(np.sum(np.exp(-(x - x_max) / alpha))) - x_max * 0  # simplified
    
def _objective(centers, alpha=50.0):
    """Objective to minimize: negative of soft-min clearance."""
    # Wall clearances
    wall_clear = 0.5 - np.abs(centers)
    min_wall = np.min(wall_clear)
    
    # Pairwise half-distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    pair_half = dists / 2.0
    
    # Use soft-min over all clearances
    # Combine wall and pair clearances
    # For efficiency, use a different approach: 
    # Minimize the negative of the minimum clearance using subgradient-like approach
    
    # Actually, let's use a direct approach: 
    # The true objective is min(min_wall, min_pair_half)
    # We want to maximize this, so minimize its negative
    
    # Use a smooth approximation
    all_clearances = np.concatenate([wall_clear.ravel(), pair_half.ravel()])
    
    # Soft min: -logsumexp(-x/alpha) * alpha
    x = all_clearances
    # Numerically stable soft-min
    x_max = np.max(x)
    soft_min_val = -alpha * np.log(np.sum(np.exp(-(x - x_max) / alpha))) - x_max * 0
    
    # Actually simpler: soft_min ≈ -alpha * log(mean(exp(-x/alpha)))
    # = -alpha * (log(sum(exp(-x/alpha))) - log(len(x)))
    # Let me redo this properly
    
    # Standard soft-min: min(x) ≈ -1/alpha * log(sum(exp(-alpha * x)))
    # For numerical stability, shift by max
    a = alpha
    x_shifted = -a * (x - x_max)
    # sum(exp(x_shifted)) / len
    log_sum = np.log(np.sum(np.exp(x_shifted))) + np.log(1.0)  # no log(len) for now
    soft_min_val = -(x_max + log_sum / a)
    
    return -soft_min_val  # We want to minimize negative of soft-min

def _initial_centers(seed=42):
    """Generate initial centers using a lattice with perturbation."""
    rng = np.random.RandomState(seed)
    
    # Use a 5x5x4 grid (100 points) with some structure
    # Actually, let's use a more space-filling approach
    # Start with a cubic lattice and perturb
    
    # 100 = 5 * 5 * 4
    m1, m2, m3 = 5, 5, 4
    x = (np.arange(m1) + 0.5) / m1 - 0.5
    y = (np.arange(m2) + 0.5) / m2 - 0.5
    z = (np.arange(m3) + 0.5) / m3 - 0.5
    
    grid = np.array([(xi, yi, zi) for zi in z for yi in y for xi in x])
    
    # Add small random perturbation to break symmetry
    perturbation = 0.05 / max(m1, m2, m3)
    grid = grid + rng.uniform(-perturbation, perturbation, size=grid.shape)
    
    # Ensure within bounds
    grid = np.clip(grid, -0.49, 0.49)
    
    return grid

def _local_search(centers, max_iter=5000, time_limit=10.0, verbose=False):
    """Local search to improve the packing."""
    start_time = time.time()
    current_centers = centers.copy()
    current_radius = _compute_radius(current_centers)
    
    # Gradient-based approach using finite differences for the min function
    # Since the min is non-smooth, we'll use a subgradient approach
    
    # Strategy: 
    # 1. Find the critical constraints (those achieving the minimum)
    # 2. Move points to relax these constraints
    
    step_size = 0.01
    best_centers = current_centers.copy()
    best_radius = current_radius
    
    for iteration in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Compute current clearances
        wall_clear = 0.5 - np.abs(current_centers)
        diff = current_centers[:, np.newaxis, :] - current_centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        pair_half = dists / 2.0
        
        current_r = min(np.min(wall_clear), np.min(pair_half))
        
        # Find critical pairs and walls
        # Wall constraints
        wall_min_val = np.min(wall_clear)
        wall_critical = np.argwhere(wall_clear <= wall_min_val + 1e-10)
        
        # Pair constraints
        pair_min_val = np.min(pair_half)
        pair_critical = np.argwhere(pair_half <= pair_min_val + 1e-10)
        # Remove diagonal
        pair_critical = [(i, j) for (i, j) in pair_critical if i != j]
        
        # Build gradient (subgradient)
        grad = np.zeros_like(current_centers)
        
        # Wall contributions
        for (i, k) in wall_critical:
            # d(0.5 - |c_ik|)/dc_ik = -sign(c_ik)
            if current_centers[i, k] > 0:
                grad[i, k] -= 1.0
            elif current_centers[i, k] < 0:
                grad[i, k] += 1.0
        
        # Pair contributions
        for (i, j) in pair_critical:
            if i >= N or j >= N:
                continue
            diff_ij = current_centers[i] - current_centers[j]
            dist_ij = np.linalg.norm(diff_ij)
            if dist_ij < 1e-12:
                # Random direction
                grad[i] += rng_dir()
                grad[j] -= rng_dir()
            else:
                unit = diff_ij / dist_ij
                # d(|ci-cj|/2)/dci = unit/2, but we want to maximize, so gradient of -dist/2 is -unit/2
                # We're maximizing min, so we want to move in direction that increases min
                # Subgradient of min at critical point: for pair (i,j), 
                # to increase |ci-cj|, move ci away from cj and cj away from ci
                grad[i] += unit * 0.5
                grad[j] -= unit * 0.5
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-12:
            # Stuck, try random perturbation
            current_centers += 0.001 * np.random.randn(*current_centers.shape)
            current_centers = np.clip(current_centers, -0.49, 0.49)
            continue
        
        grad = grad / grad_norm
        
        # Line search: try several step sizes
        best_step_r = current_r
        best_step_centers = current_centers.copy()
        
        for step in [step_size, step_size * 0.5, step_size * 0.25, step_size * 0.1]:
            trial = current_centers + step * grad
            trial = np.clip(trial, -0.49, 0.49)
            trial_r = _compute_radius(trial)
            if trial_r > best_step_r:
                best_step_r = trial_r
                best_step_centers = trial
        
        if best_step_r > current_r + 1e-12:
            current_centers = best_step_centers
            current_r = best_step_r
            if current_r > best_radius:
                best_radius = current_r
                best_centers = current_centers.copy()
            # Decrease step size slightly
            step_size *= 0.95
        else:
            # No improvement, reduce step size and try random restart
            step_size *= 0.8
            if step_size < 1e-5:
                break
            # Small random kick
            current_centers += 0.002 * np.random.randn(*current_centers.shape)
            current_centers = np.clip(current_centers, -0.49, 0.49)
    
    return best_centers, best_radius

def rng_dir():
    return np.random.randn(3) / np.sqrt(3.0)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin
    
    # Phase 1: Initial setup
    centers = _initial_centers(seed=42)
    best_centers, best_radius = _local_search(centers, max_iter=2000, time_limit=20.0)
    
    if time.time() > deadline:
        return best_centers
    
    # Phase 2: Multiple random restarts with different initializations
    rng = np.random.RandomState(123)
    
    for restart in range(5):
        if time.time() > deadline:
            break
        
        # Different initialization strategies
        if restart == 0:
            # Perturbed lattice
            init = _initial_centers(seed=100 + restart)
        elif restart == 1:
            # More random
            init = rng.uniform(-0.45, 0.45, size=(N, 3))
        elif restart == 2:
            # Fibonacci-like distribution
            init = _fibonacci_initial(N, rng)
        elif restart == 3:
            # Two-layer lattice
            init = _two_layer_initial(N, rng)
        else:
            # Random with clustering
            init = _clumped_initial(N, rng)
        
        # Local search from this initialization
        local_centers, local_radius = _local_search(init, max_iter=1500, time_limit=15.0)
        
        if local_radius > best_radius:
            best_radius = local_radius
            best_centers = local_centers
    
    # Phase 3: Final polish with tighter parameters
    if time.time() < deadline - 20.0:
        polished_centers, polished_radius = _local_search(
            best_centers, max_iter=5000, time_limit=18.0
        )
        if polished_radius > best_radius:
            best_radius = polished_radius
            best_centers = polished_centers
    
    # Ensure valid output
    best_centers = np.clip(best_centers, -0.499999, 0.499999)
    
    return best_centers

def _fibonacci_initial(n, rng):
    """Generate points using Fibonacci sphere-like distribution, then map to cube."""
    points = []
    golden_angle = np.pi * (3 - np.sqrt(5))
    for i in range(n):
        # Fibonacci sphere
        t = i / n
        z = 1 - 2 * t
        r = np.sqrt(1 - z * z)
        theta = golden_angle * i
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        points.append([x, y, z])
    
    points = np.array(points)
    # Scale to fit in cube with some margin
    points = points * 0.4
    # Add small random perturbation
    points += rng.uniform(-0.02, 0.02, size=points.shape)
    points = np.clip(points, -0.49, 0.49)
    return points

def _two_layer_initial(n, rng):
    """Two-layer lattice initialization."""
    # 50 points per layer
    m = 5  # 5x10 = 50 per layer
    x = (np.arange(5) + 0.5) / 5 - 0.5
    y = (np.arange(10) + 0.5) / 10 - 0.5
    
    layer1 = np.array([(xi, yi, -0.2) for yi in y for xi in x])
    layer2 = np.array([(xi, yi, 0.2) for yi in y for xi in x])
    
    points = np.vstack([layer1, layer2])
    points += rng.uniform(-0.03, 0.03, size=points.shape)
    points = np.clip(points, -0.49, 0.49)
    return points

def _clumped_initial(n, rng):
    """Initialize with some clustering to test escape from local optima."""
    # Create 10 clusters of 10 points each
    cluster_centers = rng.uniform(-0.3, 0.3, size=(10, 3))
    points = []
    for cc in cluster_centers:
        cluster = cc + rng.uniform(-0.1, 0.1, size=(10, 3))
        points.append(cluster)
    points = np.vstack(points)
    points = np.clip(points, -0.49, 0.49)
    return points

# EVOLVE-BLOCK-END
