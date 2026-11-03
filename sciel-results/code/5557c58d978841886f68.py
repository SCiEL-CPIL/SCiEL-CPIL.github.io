"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_radius_sq(centers):
    """Compute the squared common radius (4*r^2) for speed. Returns 4*r^2."""
    # Pairwise squared distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    min_pair_dist_sq = np.min(dists_sq)
    
    # Distance to walls squared
    wall_dists = np.minimum(0.5 - centers, centers + 0.5)
    min_wall_dist = np.min(wall_dists)
    min_wall_dist_sq = min_wall_dist ** 2
    
    return min(min_pair_dist_sq, min_wall_dist_sq)

def _compute_radius(centers):
    """Compute the common radius r for given centers."""
    return np.sqrt(_compute_radius_sq(centers)) / 2.0

def _local_optimize(centers, max_time=50.0, seed=42):
    """Local optimization to maximize the minimum radius using adaptive steps."""
    start_time = time.time()
    best_centers = centers.copy()
    best_radius_sq = _compute_radius_sq(best_centers)
    
    n = len(centers)
    rng = np.random.RandomState(seed)
    
    # Adaptive step size
    step_size = 0.03
    min_step = 1e-7
    no_improve_count = 0
    max_no_improve = 30
    
    while time.time() - start_time < max_time:
        if no_improve_count > max_no_improve:
            break
        
        improved = False
        # Try multiple random directions
        for _ in range(30):
            if time.time() - start_time > max_time - 1:
                break
            
            # Random perturbation
            delta = rng.randn(n, 3) * step_size
            new_centers = best_centers + delta
            
            # Clamp to valid range
            new_centers = np.clip(new_centers, -0.5, 0.5)
            
            new_radius_sq = _compute_radius_sq(new_centers)
            
            if new_radius_sq > best_radius_sq + 1e-14:
                best_centers = new_centers
                best_radius_sq = new_radius_sq
                improved = True
                no_improve_count = 0
                # Slightly decrease step size on improvement
                step_size *= 0.97
        
        if not improved:
            no_improve_count += 1
            step_size *= 0.85
            if step_size < min_step:
                break
    
    return best_centers, best_radius_sq

def _coordinate_descent(centers, max_time=20.0):
    """Coordinate descent optimization."""
    start_time = time.time()
    best_centers = centers.copy()
    best_radius_sq = _compute_radius_sq(best_centers)
    
    n = len(centers)
    rng = np.random.RandomState(123)
    step_size = 0.02
    
    while time.time() - start_time < max_time:
        improved = False
        for _ in range(50):
            if time.time() - start_time > max_time - 1:
                break
            
            # Pick a random center and a random axis
            i = rng.randint(n)
            axis = rng.randint(3)
            
            # Try moving in both directions
            for direction in [1, -1]:
                delta = np.zeros((n, 3))
                delta[i, axis] = direction * step_size
                new_centers = best_centers + delta
                new_centers = np.clip(new_centers, -0.5, 0.5)
                
                new_radius_sq = _compute_radius_sq(new_centers)
                
                if new_radius_sq > best_radius_sq + 1e-14:
                    best_centers = new_centers
                    best_radius_sq = new_radius_sq
                    improved = True
        
        if not improved:
            step_size *= 0.9
            if step_size < 1e-7:
                break
    
    return best_centers, best_radius_sq

def _generate_cubic_grid():
    """Generate a cubic grid initial configuration."""
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].copy()

def _generate_bcc_like():
    """Generate a BCC-like initial configuration."""
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    # Use a finer grid and select points
    m2 = int(np.ceil((N * 1.5) ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m2) + 0.5) / m2 - 0.5
    # Create a 3D grid
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    # Use a simple selection: take every other point in a pattern
    # Or just use the first N points from a denser grid
    # Actually, let's use a different approach: start with cubic and add jitter
    centers = _generate_cubic_grid()
    rng = np.random.RandomState(77)
    centers = centers + rng.randn(N, 3) * 0.01
    centers = np.clip(centers, -0.5, 0.5)
    return centers

def _generate_random_repulsive():
    """Generate a random configuration with some repulsion."""
    rng = np.random.RandomState(99)
    centers = rng.uniform(-0.5, 0.5, (N, 3))
    
    # Simple repulsion: iteratively push apart close points
    for _ in range(200):
        # Find closest pairs and push them apart
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # For each point, move it away from its closest neighbor
        for i in range(N):
            j = np.argmin(dists[i])
            if dists[i, j] < 0.1:
                direction = centers[i] - centers[j]
                norm = np.linalg.norm(direction)
                if norm > 1e-10:
                    direction = direction / norm
                    centers[i] += direction * 0.01
                    centers[i] = np.clip(centers[i], -0.5, 0.5)
    
    return centers

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    # Try multiple initial configurations
    initial_configs = [
        _generate_cubic_grid(),
        _generate_bcc_like(),
        _generate_random_repulsive(),
    ]
    
    best_centers = None
    best_radius_sq = -np.inf
    
    # Optimize each initial configuration
    for idx, init_centers in enumerate(initial_configs):
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining < 5:
            break
        
        # Allocate time based on how many configs are left
        time_per_config = min(40.0, remaining / max(1, len(initial_configs) - idx))
        
        centers, radius_sq = _local_optimize(init_centers, max_time=time_per_config, seed=42 + idx)
        
        if radius_sq > best_radius_sq:
            best_radius_sq = radius_sq
            best_centers = centers.copy()
    
    # Final polish with coordinate descent
    if best_centers is not None:
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining > 5:
            best_centers, best_radius_sq = _coordinate_descent(best_centers, max_time=min(20.0, remaining))
    
    # Ensure all centers are within the cube
    if best_centers is None:
        best_centers = _generate_cubic_grid()
    
    best_centers = np.clip(best_centers, -0.5, 0.5)
    
    return best_centers
# EVOLVE-BLOCK-END
