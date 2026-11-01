import numpy as np
import time
from scipy.spatial.distance import pdist, cdist

N = 100
TIME_LIMIT = 110.0  # Safety margin before 120s kill

def _compute_radius(centers):
    """Compute the actual radius for a set of centers."""
    wall_dists = np.min(np.abs(centers), axis=1)
    min_wall = np.min(wall_dists)
    dists = pdist(centers, 'euclidean')
    min_pair = np.min(dists)
    return min(min_wall, min_pair / 2.0)

def _initial_grid():
    """Create a 5x5x4 grid arrangement (100 points)."""
    nx, ny, nz = 5, 5, 4
    xs = np.linspace(-0.5 + 0.5/nx, 0.5 - 0.5/nx, nx)
    ys = np.linspace(-0.5 + 0.5/ny, 0.5 - 0.5/ny, ny)
    zs = np.linspace(-0.5 + 0.5/nz, 0.5 - 0.5/nz, nz)
    grid = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    return grid[:N]

def construct():
    start_time = time.time()
    
    # Initial configuration
    centers = _initial_grid().copy()
    best_centers = centers.copy()
    best_radius = _compute_radius(centers)
    
    # Simulated annealing with aggressive time management
    T = 0.0005
    T_min = 1e-7
    step_size = 0.005
    min_step = 1e-5
    
    # Pre-allocate for efficiency
    while True:
        elapsed = time.time() - start_time
        if elapsed > TIME_LIMIT:
            break
        
        # Generate move: displace 1-3 random centers
        n_moves = np.random.choice([1, 2, 3], p=[0.6, 0.3, 0.1])
        indices = np.random.choice(N, size=n_moves, replace=False)
        
        old_centers = centers[indices].copy()
        disp = np.random.normal(0, step_size, size=(n_moves, 3))
        centers[indices] = old_centers + disp
        centers[indices] = np.clip(centers[indices], -0.4999, 0.4999)
        
        new_radius = _compute_radius(centers)
        
        # Metropolis acceptance
        if new_radius > best_radius:
            best_radius = new_radius
            best_centers = centers.copy()
            # Reduce step size on improvement
            step_size = max(min_step, step_size * 0.97)
        else:
            delta = best_radius - new_radius
            if delta < T and np.random.random() < np.exp(-delta / T):
                pass  # Accept
            else:
                centers[indices] = old_centers
        
        # Cooling
        T = max(T_min, T * 0.9998)
        
        # Periodic local refinement every ~500 iterations
        if np.random.random() < 0.005:
            # Find closest pair
            dist_matrix = cdist(centers, centers)
            np.fill_diagonal(dist_matrix, np.inf)
            min_pos = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
            i, j = min_pos
            
            # Move apart
            direction = centers[i] - centers[j]
            norm = np.linalg.norm(direction)
            if norm > 1e-10:
                direction /= norm
                move = 0.001 * direction
                old_i, old_j = centers[i].copy(), centers[j].copy()
                centers[i] = np.clip(old_i + move, -0.4999, 0.4999)
                centers[j] = np.clip(old_j - move, -0.4999, 0.4999)
                
                new_r = _compute_radius(centers)
                if new_r > best_radius:
                    best_radius = new_r
                    best_centers = centers.copy()
                else:
                    centers[i], centers[j] = old_i, old_j
    
    # Final local refinement
    remaining = TIME_LIMIT - (time.time() - start_time)
    if remaining > 5:
        for _ in range(min(100, int(remaining * 10))):
            if time.time() - start_time > TIME_LIMIT:
                break
            
            dist_matrix = cdist(best_centers, best_centers)
            np.fill_diagonal(dist_matrix, np.inf)
            min_pos = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
            i, j = min_pos
            
            # Try small perturbations
            for _ in range(10):
                pert = np.random.normal(0, 0.0005, size=(2, 3))
                new_i = np.clip(best_centers[i] + pert[0], -0.4999, 0.4999)
                new_j = np.clip(best_centers[j] + pert[1], -0.4999, 0.4999)
                
                test = best_centers.copy()
                test[i], test[j] = new_i, new_j
                test_r = _compute_radius(test)
                
                if test_r > best_radius:
                    best_radius = test_r
                    best_centers = test
                    break
    
    return best_centers
