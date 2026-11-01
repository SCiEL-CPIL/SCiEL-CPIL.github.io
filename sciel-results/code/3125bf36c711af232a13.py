"""Optimized sphere packing using simulated annealing with deadline guard."""
import numpy as np
import time

N = 100
DEADLINE_SECONDS = 140.0  # Safety margin before 150s limit

def _compute_radius(centers):
    """Compute the maximum common radius for given centers."""
    # Boundary constraint: distance from each center to nearest cube face
    # Cube is [-0.5, 0.5]^3, so distance to face is 0.5 - |c_ik|
    boundary = np.min(np.abs(centers), axis=1)
    boundary_r = 0.5 - np.max(boundary, axis=1)  # min over k for each i
    min_boundary = np.min(boundary_r)
    
    # Pairwise distances
    # For efficiency, compute all pairwise distances
    n = len(centers)
    # Use broadcasting for pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    # Set diagonal to infinity
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    return min(min_pair, min_boundary)

def _compute_radius_fast(centers):
    """Faster radius computation using vectorized ops."""
    # Boundary
    max_abs = np.max(np.abs(centers), axis=1)
    min_boundary = 0.5 - np.min(max_abs)
    
    # Pairwise - use chunked computation for memory
    n = len(centers)
    min_pair = np.inf
    chunk = 50
    for i in range(0, n, chunk):
        end = min(i + chunk, n)
        diff = centers[i:end, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        # Only consider j > i to avoid duplicates and diagonal
        for k in range(end - i):
            row = dists[k, i+k+1:]
            if len(row) > 0:
                m = np.min(row)
                if m < min_pair:
                    min_pair = m
    min_pair /= 2.0
    return min(min_pair, min_boundary)

def _initial_configuration():
    """Generate a good initial configuration."""
    # Use a jittered grid approach
    # 100 spheres: try 5x5x4 = 100
    m1, m2, m3 = 5, 5, 4
    ticks1 = (np.arange(m1) + 0.5) / m1 - 0.5
    ticks2 = (np.arange(m2) + 0.5) / m2 - 0.5
    ticks3 = (np.arange(m3) + 0.5) / m3 - 0.5
    
    centers = np.array([(x, y, z) for z in ticks3 for y in ticks2 for x in ticks1])
    centers = centers[:N]
    
    # Add small random jitter to break symmetry
    np.random.seed(42)
    centers += np.random.uniform(-0.01, 0.01, size=centers.shape)
    
    # Clip to valid range
    centers = np.clip(centers, -0.48, 0.48)
    
    return centers

def _sa_step(centers, temp, rng):
    """Perform one simulated annealing step. Returns new centers and delta_r."""
    n = len(centers)
    i = rng.integers(0, n)
    
    old_center = centers[i].copy()
    # Propose a random move
    step_size = 0.02 * (1.0 + temp)
    new_center = old_center + rng.normal(0, step_size, size=3)
    
    # Keep within bounds
    new_center = np.clip(new_center, -0.49, 0.49)
    
    # Compute change in radius
    # Old radius contribution from sphere i
    old_r = _compute_radius_fast(centers)
    
    centers[i] = new_center
    new_r = _compute_radius_fast(centers)
    
    delta = new_r - old_r
    
    if delta > 0 or rng.random() < np.exp(delta / max(temp, 1e-10)):
        return centers, delta, True
    else:
        centers[i] = old_center
        return centers, delta, False

def construct():
    start_time = time.time()
    
    # Initial configuration
    centers = _initial_configuration()
    best_centers = centers.copy()
    best_r = _compute_radius_fast(best_centers)
    
    # Simulated annealing
    rng = np.random.default_rng(12345)
    
    T_start = 0.05
    T_end = 0.0001
    n_iter = 500000  # Will be limited by deadline
    
    # Adaptive step size based on current radius
    current_step_scale = 0.02
    
    for it in range(n_iter):
        # Deadline check
        if time.time() - start_time > DEADLINE_SECONDS:
            break
        
        # Temperature schedule (geometric cooling)
        progress = it / n_iter
        temp = T_start * (T_end / T_start) ** progress
        
        # Adjust step size based on temperature
        step_size = current_step_scale * (0.5 + temp / T_start)
        
        # Propose move
        i = rng.integers(0, N)
        old_center = centers[i].copy()
        new_center = old_center + rng.normal(0, step_size, size=3)
        new_center = np.clip(new_center, -0.49, 0.49)
        
        # Compute old radius (cache if possible)
        old_r = _compute_radius_fast(centers)
        
        centers[i] = new_center
        new_r = _compute_radius_fast(centers)
        delta = new_r - old_r
        
        # Acceptance criterion
        if delta > 0 or rng.random() < np.exp(delta / max(temp, 1e-10)):
            if new_r > best_r:
                best_r = new_r
                best_centers = centers.copy()
        else:
            centers[i] = old_center
        
        # Periodically re-evaluate and potentially restart if stuck
        if it % 10000 == 0 and it > 0:
            # Check if we've made progress
            pass
    
    # Final local refinement: try small perturbations around best
    if time.time() - start_time < DEADLINE_SECONDS - 10:
        for _ in range(10000):
            if time.time() - start_time > DEADLINE_SECONDS:
                break
            i = rng.integers(0, N)
            old_center = best_centers[i].copy()
            new_center = old_center + rng.normal(0, 0.005, size=3)
            new_center = np.clip(new_center, -0.49, 0.49)
            
            test_centers = best_centers.copy()
            test_centers[i] = new_center
            test_r = _compute_radius_fast(test_centers)
            
            if test_r > best_r:
                best_r = test_r
                best_centers = test_centers
    
    # Ensure validity: clip to valid range
    best_centers = np.clip(best_centers, -0.4999, 0.4999)
    
    return best_centers
