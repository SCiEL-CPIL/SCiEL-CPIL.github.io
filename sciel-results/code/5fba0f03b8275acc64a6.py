"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.spatial import cKDTree

N = 200

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist
import time

_START_TIME = time.time()
_TIME_LIMIT = 140.0  # 10s safety margin before 150s kill

def _time_left():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _compute_radius(centers):
    """Compute the maximum valid radius for given centers."""
    if centers.shape[0] != N:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Check centers are inside the square [-0.5, 0.5]^2
    if np.any(centers < -0.5 - 1e-12) or np.any(centers > 0.5 + 1e-12):
        return 0.0
    
    # Distance to walls
    wall_dists = np.min(np.minimum(0.5 - centers, centers + 0.5), axis=1)
    min_wall = np.min(wall_dists)
    
    # Pairwise distances using pdist for exact min
    min_pair = np.min(pdist(centers))
    
    r = min(min_wall, min_pair / 2.0)
    return r

def _generate_hex_grid():
    """Generate a hexagonal lattice packing as initial configuration."""
    best_centers = None
    best_r = 0.0
    
    # Try different row counts
    for n_rows in range(10, 25):
        n_cols = int(np.ceil(N / n_rows))
        
        r1 = 1.0 / (2.0 * n_cols)
        r2 = 1.0 / ((n_rows - 1) * np.sqrt(3) + 2)
        r = min(r1, r2)
        s = 2 * r
        
        centers = []
        for i in range(n_rows):
            for j in range(n_cols):
                if len(centers) >= N:
                    break
                x = -0.5 + r + j * s + (i % 2) * s / 2
                y = -0.5 + r + i * s * np.sqrt(3) / 2
                centers.append([x, y])
            if len(centers) >= N:
                break
        
        centers = np.array(centers[:N])
        
        if np.any(centers < -0.5 - 1e-9) or np.any(centers > 0.5 + 1e-9):
            xs = centers[:, 0]
            ys = centers[:, 1]
            cx = (xs.max() + xs.min()) / 2
            cy = (ys.max() + ys.min()) / 2
            centers[:, 0] -= cx
            centers[:, 1] -= cy
            
            if np.any(centers < -0.5 - 1e-9) or np.any(centers > 0.5 + 1e-9):
                continue
        
        r_actual = _compute_radius(centers)
        if r_actual > best_r:
            best_r = r_actual
            best_centers = centers.copy()
    
    if best_centers is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
    
    return best_centers

def _soft_min(x, beta=50.0):
    """Soft minimum using log-sum-exp trick for numerical stability."""
    x_min = np.min(x)
    shifted = -(x - x_min) * beta
    shifted = np.clip(shifted, -500, 500)
    log_sum = np.log(np.sum(np.exp(shifted)))
    return x_min - log_sum / beta

def _gradient_refine(centers, max_iter=3000, lr=0.002):
    """
    Gradient-based refinement with improved neighbor count and adaptive beta.
    """
    centers = centers.copy()
    best_t = _compute_radius(centers)
    best_centers = centers.copy()
    
    no_improve_count = 0
    current_lr = lr
    
    for it in range(max_iter):
        if _time_left() < 15.0:
            break
        
        # Compute current radius
        d = pdist(centers)
        min_pair_dist = np.min(d)
        wall_dists = np.min(np.minimum(0.5 - centers, centers + 0.5), axis=1)
        min_wall = np.min(wall_dists)
        current_r = min(min_pair_dist / 2.0, min_wall)
        
        if current_r > best_t + 1e-12:
            best_t = current_r
            best_centers = centers.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        # Build KDTree for neighbor queries - use more neighbors for better gradient
        tree = cKDTree(centers)
        k_neighbors = min(10, N)
        dists, idx = tree.query(centers, k=k_neighbors)
        
        # Gradient: for each point, sum of repulsion from nearby neighbors + wall repulsion
        grad = np.zeros_like(centers)
        
        for i in range(N):
            # Repulsion from nearby neighbors with distance-based weighting
            for k in range(1, k_neighbors):
                j = idx[i, k]
                diff = centers[i] - centers[j]
                dist = dists[i, k]
                if dist > 1e-10:
                    # Stronger repulsion for closer pairs, with smooth falloff
                    weight = 1.0 / (dist * dist + 1e-8)
                    grad[i] += weight * diff
            
            # Wall repulsion with smooth activation
            for k in range(2):
                if centers[i, k] < 0:
                    wall_dist = 0.5 + centers[i, k]
                    if wall_dist < 0.2:
                        grad[i, k] += 1.0 / (wall_dist + 1e-6)
                else:
                    wall_dist = 0.5 - centers[i, k]
                    if wall_dist < 0.2:
                        grad[i, k] -= 1.0 / (wall_dist + 1e-6)
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-10:
            grad /= grad_norm
        
        # Update with adaptive learning rate
        centers += current_lr * grad
        centers[:, 0] = np.clip(centers[:, 0], -0.5 + 1e-10, 0.5 - 1e-10)
        centers[:, 1] = np.clip(centers[:, 1], -0.5 + 1e-10, 0.5 - 1e-10)
        
        # Adaptive learning rate: reduce when no improvement
        if no_improve_count > 50:
            current_lr *= 0.85
            no_improve_count = 0
            if current_lr < 1e-9:
                break
        elif no_improve_count == 20:
            current_lr *= 0.95
    
    return best_centers, best_t

def _multi_start_optimize(centers_list, time_per_start):
    """Run optimization on multiple starting configurations with time sharing."""
    best_centers = None
    best_r = 0.0
    
    for centers in centers_list:
        if _time_left() < 30.0:
            break
        
        start_time = time.time()
        centers_opt, r = _gradient_refine(centers, max_iter=2000, lr=0.002)
        
        if r > best_r:
            best_r = r
            best_centers = centers_opt.copy()
        
        # Check if we've exceeded time for this start
        if time.time() - start_time > time_per_start:
            continue
    
    return best_centers, best_r

def construct():
    """Construct 200 circle centers in [-0.5, 0.5]^2 maximizing the minimum radius."""
    # Generate multiple diverse initial configurations
    base_centers = _generate_hex_grid()
    
    # Create variants with small perturbations
    centers_list = [base_centers.copy()]
    
    # Add perturbed versions for diversity
    rng = np.random.RandomState(42)
    for _ in range(3):
        perturbed = base_centers.copy()
        noise = rng.normal(0, 0.005, size=perturbed.shape)
        perturbed += noise
        perturbed[:, 0] = np.clip(perturbed[:, 0], -0.45, 0.45)
        perturbed[:, 1] = np.clip(perturbed[:, 1], -0.45, 0.45)
        centers_list.append(perturbed)
    
    # Time-shared optimization across multiple starts
    total_time_available = _time_left() - 20.0  # Reserve 20s for final polish
    if total_time_available > 10.0:
        time_per_start = max(5.0, total_time_available / len(centers_list))
        centers, r = _multi_start_optimize(centers_list, time_per_start)
    else:
        centers = base_centers
        r = _compute_radius(centers)
    
    # Final high-precision polish pass
    if _time_left() > 20.0:
        # Multiple refinement passes with decreasing learning rates
        for lr in [0.001, 0.0005, 0.0002, 0.0001]:
            if _time_left() < 10.0:
                break
            centers, r_new = _gradient_refine(centers, max_iter=1000, lr=lr)
            if r_new > r:
                r = r_new
    
    # Final verification and clipping
    centers = np.clip(centers, -0.5, 0.5)
    
    return centers
# EVOLVE-BLOCK-END
