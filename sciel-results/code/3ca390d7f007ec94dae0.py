"""Packing 200 equal circles in a unit square to maximize radius."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import minimize

START_TIME = time.time()
TIME_LIMIT = 140.0  # Leave 10s safety margin before 150s kill
SAFETY_MARGIN = 10.0


def _remaining_time():
    return TIME_LIMIT - SAFETY_MARGIN - (time.time() - START_TIME)


def _compute_radius(centers):
    """Compute the maximum radius for given centers."""
    if len(centers) == 0:
        return 0.0
    
    # Distance to walls: min_i min_k (0.5 - |c_ik|)
    wall_dist = np.min(0.5 - np.abs(centers), axis=None)
    
    # Pairwise distances
    n = len(centers)
    if n < 2:
        return float(wall_dist)
    
    # Compute all pairwise distances efficiently
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    return min(float(wall_dist), float(min_pair_dist / 2.0))


def _initial_grid():
    """Create initial grid arrangement with some jitter."""
    m = int(np.ceil(np.sqrt(N)))
    # Use a slightly offset grid
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()


def _initial_random():
    """Create random initial positions."""
    return np.random.uniform(-0.5, 0.5, size=(N, 2))


def _initial_honeycomb():
    """Create honeycomb-like initial arrangement."""
    # For 200 circles, try a hexagonal packing pattern
    # Approximate number of rows
    rows = 15
    cols = 14  # 15*14 = 210, take first 200
    
    centers = []
    dx = 1.0 / (cols + 0.5)
    dy = dx * np.sqrt(3) / 2
    
    for i in range(rows):
        for j in range(cols):
            if len(centers) >= N:
                break
            x = j * dx - 0.5 + dx/2
            y = i * dy - 0.5 + dy/2
            if i % 2 == 1:
                x += dx / 2
            # Scale to fit in [-0.5, 0.5]
            x = x * (cols * dx) / 1.0
            y = y * (rows * dy) / 1.0
            x = x - (cols * dx) / 2
            y = y - (rows * dy) / 2
            centers.append([x, y])
    
    return np.array(centers[:N])


def _repulsion_energy(centers, k=10.0):
    """Compute a repulsion-based energy to maximize minimum distance."""
    n = len(centers)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Energy: sum of 1/d^k for all pairs (repulsion)
    # Also add wall repulsion
    energy = np.sum(1.0 / (dists ** k))
    
    # Wall repulsion: push centers away from boundaries
    wall_penalty = np.sum(np.maximum(0, 0.45 - np.abs(centers)) ** 2)
    energy += 100.0 * wall_penalty
    
    return energy


def _optimize_with_repulsion(centers, max_iter=3000):
    """Optimize using repulsion energy minimization."""
    def neg_radius(x):
        c = x.reshape(-1, 2)
        return -_compute_radius(c)
    
    def repulsion_obj(x):
        c = x.reshape(-1, 2)
        return _repulsion_energy(c)
    
    # Try multiple optimization strategies
    best_radius = _compute_radius(centers)
    best_centers = centers.copy()
    
    # Strategy 1: Direct radius maximization with Nelder-Mead
    if _remaining_time() > 30.0:
        try:
            result = minimize(neg_radius, centers.flatten(), method='Nelder-Mead',
                              options={'maxiter': max_iter, 'xatol': 1e-10, 'fatol': 1e-12,
                                       'adaptive': True})
            opt_centers = result.x.reshape(-1, 2)
            opt_radius = _compute_radius(opt_centers)
            if opt_radius > best_radius:
                best_radius = opt_radius
                best_centers = opt_centers.copy()
        except Exception:
            pass
    
    # Strategy 2: Repulsion energy minimization
    if _remaining_time() > 20.0:
        try:
            result = minimize(repulsion_obj, best_centers.flatten(), method='L-BFGS-B',
                              options={'maxiter': max_iter // 2, 'ftol': 1e-15, 'gtol': 1e-10})
            opt_centers = result.x.reshape(-1, 2)
            # Clip to valid region
            opt_centers = np.clip(opt_centers, -0.49, 0.49)
            opt_radius = _compute_radius(opt_centers)
            if opt_radius > best_radius:
                best_radius = opt_radius
                best_centers = opt_centers.copy()
        except Exception:
            pass
    
    return best_centers, best_radius


def _local_bottleneck_opt(centers, max_iters=500):
    """Identify bottleneck pairs and locally optimize them."""
    best_radius = _compute_radius(centers)
    best_centers = centers.copy()
    
    for _ in range(max_iters):
        if _remaining_time() < 5.0:
            break
        
        # Find the minimum distance pair
        diff = best_centers[:, np.newaxis, :] - best_centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        min_idx = np.unravel_index(np.argmin(dists), dists.shape)
        i, j = min_idx[0], min_idx[1]
        min_dist = dists[i, j]
        current_radius = min(best_radius, min_dist / 2.0)
        
        # Also check wall distances
        wall_dists = 0.5 - np.abs(best_centers)
        min_wall = np.min(wall_dists)
        current_radius = min(current_radius, min_wall)
        
        # Try to move the two closest points apart
        # Compute direction to move them apart
        direction = best_centers[j] - best_centers[i]
        norm = np.linalg.norm(direction)
        if norm < 1e-10:
            direction = np.array([1.0, 0.0])
            norm = 1.0
        else:
            direction = direction / norm
        
        # Try small moves in the separating direction
        improved = False
        for step in [0.001, 0.0005, 0.0002, 0.0001]:
            new_i = best_centers[i].copy()
            new_j = best_centers[j].copy()
            new_i -= direction * step
            new_j += direction * step
            
            # Clip to bounds
            new_i = np.clip(new_i, -0.49, 0.49)
            new_j = np.clip(new_j, -0.49, 0.49)
            
            test_centers = best_centers.copy()
            test_centers[i] = new_i
            test_centers[j] = new_j
            
            test_radius = _compute_radius(test_centers)
            if test_radius > current_radius + 1e-12:
                best_centers = test_centers
                best_radius = test_radius
                improved = True
                break
        
        if not improved:
            # Try random perturbations of the bottleneck points
            for _ in range(20):
                if _remaining_time() < 5.0:
                    break
                test_centers = best_centers.copy()
                test_centers[i] += np.random.normal(0, 0.001, size=2)
                test_centers[j] += np.random.normal(0, 0.001, size=2)
                test_centers[i] = np.clip(test_centers[i], -0.49, 0.49)
                test_centers[j] = np.clip(test_centers[j], -0.49, 0.49)
                
                test_radius = _compute_radius(test_centers)
                if test_radius > current_radius + 1e-12:
                    best_centers = test_centers
                    best_radius = test_radius
                    improved = True
                    break
        
        if not improved:
            break
    
    return best_centers, best_radius


def _polish(centers, max_iters=200):
    """Final polish: small random perturbations to escape local minima."""
    best_radius = _compute_radius(centers)
    best_centers = centers.copy()
    
    for _ in range(max_iters):
        if _remaining_time() < 3.0:
            break
        
        # Random perturbation of a random subset
        n_perturb = random.randint(1, 5)
        indices = random.sample(range(N), n_perturb)
        
        test_centers = best_centers.copy()
        for idx in indices:
            test_centers[idx] += np.random.normal(0, 0.0005, size=2)
            test_centers[idx] = np.clip(test_centers[idx], -0.49, 0.49)
        
        test_radius = _compute_radius(test_centers)
        if test_radius > best_radius + 1e-12:
            best_centers = test_centers
            best_radius = test_radius
    
    return best_centers, best_radius


def construct():
    """Construct optimal circle packing."""
    random.seed(42)
    np.random.seed(42)
    
    best_radius = 0.0
    best_centers = None
    
    # Try multiple initializations
    initializations = [
        ("grid", _initial_grid()),
        ("random", _initial_random()),
        ("honeycomb", _initial_honeycomb()),
    ]
    
    for name, centers in initializations:
        if _remaining_time() < 15.0:
            break
        
        # Optimize this initialization
        opt_centers, opt_radius = _optimize_with_repulsion(centers.copy())
        
        if opt_radius > best_radius:
            best_radius = opt_radius
            best_centers = opt_centers.copy()
        
        # Local bottleneck optimization
        if _remaining_time() > 10.0:
            local_centers, local_radius = _local_bottleneck_opt(best_centers.copy())
            if local_radius > best_radius:
                best_radius = local_radius
                best_centers = local_centers.copy()
        
        # Polish
        if _remaining_time() > 5.0:
            polish_centers, polish_radius = _polish(best_centers.copy())
            if polish_radius > best_radius:
                best_radius = polish_radius
                best_centers = polish_centers.copy()
    
    # Final safety: ensure all centers are within bounds
    if best_centers is not None:
        best_centers = np.clip(best_centers, -0.5, 0.5)
    else:
        best_centers = _initial_grid()
        best_centers = np.clip(best_centers, -0.5, 0.5)
    
    return best_centers
# EVOLVE-BLOCK-END
