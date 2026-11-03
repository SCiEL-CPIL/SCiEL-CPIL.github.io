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
_TIME_LIMIT = 135.0  # 15s safety margin before 150s kill

def _time_left():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _compute_radius(centers):
    """Compute the maximum valid radius for given centers."""
    if centers.shape[0] != N:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(centers < -0.5 - 1e-12) or np.any(centers > 0.5 + 1e-12):
        return 0.0
    wall_dists = np.min(np.minimum(0.5 - centers, centers + 0.5), axis=1)
    min_wall = np.min(wall_dists)
    min_pair = np.min(pdist(centers))
    r = min(min_wall, min_pair / 2.0)
    return r

def _generate_hex_grid(n_rows=None, jitter=0.0):
    """Generate a hexagonal lattice packing, centered in the square."""
    if n_rows is None:
        n_rows = 14
    
    n_cols = int(np.ceil(N / n_rows))
    
    # Hexagonal spacing
    # Width: (n_cols - 1) * s + 2*r <= 1
    # Height: (n_rows - 1) * s * sqrt(3)/2 + 2*r <= 1
    # s >= 2*r
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
    
    # Center the grid
    cx = (centers[:, 0].max() + centers[:, 0].min()) / 2
    cy = (centers[:, 1].max() + centers[:, 1].min()) / 2
    centers[:, 0] -= cx
    centers[:, 1] -= cy
    
    if jitter > 0:
        centers += np.random.normal(0, jitter, centers.shape)
        centers = np.clip(centers, -0.5 + 1e-10, 0.5 - 1e-10)
    
    return centers

def _generate_best_hex():
    """Try multiple row counts and return the best hex grid."""
    best_centers = None
    best_r = 0.0
    
    for n_rows in range(10, 20):
        centers = _generate_hex_grid(n_rows=n_rows)
        r = _compute_radius(centers)
        if r > best_r:
            best_r = r
            best_centers = centers.copy()
    
    return best_centers

def _project(centers):
    """Project centers onto the feasible square."""
    centers = np.clip(centers, -0.5 + 1e-10, 0.5 - 1e-10)
    return centers

def _pgrad_momentum(centers, max_iter=5000, lr=0.005, momentum=0.9, 
                     wall_weight=5.0, min_dist_threshold=0.15):
    """
    Projected gradient descent with momentum.
    Maximizes sum of squared pairwise distances (encourages uniform distribution),
    with wall repulsion to keep points away from boundaries.
    """
    centers = centers.copy().astype(np.float64)
    velocity = np.zeros_like(centers)
    best_r = _compute_radius(centers)
    best_centers = centers.copy()
    
    tree = cKDTree(centers)
    
    for it in range(max_iter):
        if _time_left() < 25.0:
            break
        
        # Query neighbors for efficient gradient computation
        k = min(20, N)
        dists, idx = tree.query(centers, k=k)
        
        grad = np.zeros_like(centers)
        
        # Sum of squared distances gradient: grad_i = 2 * sum_j (c_i - c_j)
        # For each point, sum over its k nearest neighbors
        for i in range(N):
            for kk in range(1, k):
                j = idx[i, kk]
                diff = centers[i] - centers[j]
                grad[i] += diff
        
        # Wall repulsion: push points away from walls when close
        for i in range(N):
            for dim in range(2):
                val = centers[i, dim]
                if val < 0:
                    wall_dist = 0.5 + val
                    if wall_dist < min_dist_threshold:
                        grad[i, dim] += wall_weight / (wall_dist + 1e-6)
                else:
                    wall_dist = 0.5 - val
                    if wall_dist < min_dist_threshold:
                        grad[i, dim] -= wall_weight / (wall_dist + 1e-6)
        
        # Momentum update
        velocity = momentum * velocity + lr * grad
        centers = centers + velocity
        centers = _project(centers)
        
        # Update tree periodically
        if it % 100 == 0:
            tree = cKDTree(centers)
        
        # Track best
        if it % 50 == 0:
            r = _compute_radius(centers)
            if r > best_r + 1e-12:
                best_r = r
                best_centers = centers.copy()
    
    # Final check
    r = _compute_radius(centers)
    if r > best_r:
        best_r = r
        best_centers = centers.copy()
    
    return best_centers, best_r

def _soft_min(x, beta=50.0):
    """Soft minimum using log-sum-exp trick."""
    x_min = np.min(x)
    shifted = -(x - x_min) * beta
    shifted = np.clip(shifted, -500, 500)
    log_sum = np.log(np.sum(np.exp(shifted)))
    return x_min - log_sum / beta

def _gradient_refine(centers, max_iter=3000, lr=0.001, beta=50.0):
    """
    Gradient-based refinement using smooth surrogate.
    Maximizes the minimum of pairwise distances and wall distances.
    """
    centers = centers.copy().astype(np.float64)
    best_t = _compute_radius(centers)
    best_centers = centers.copy()
    
    no_improve_count = 0
    current_lr = lr
    
    for it in range(max_iter):
        if _time_left() < 20.0:
            break
        
        # Compute current constraint values
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
        
        # Compute gradient: repulsion from close pairs + wall repulsion
        tree = cKDTree(centers)
        dists, idx = tree.query(centers, k=8)
        
        grad = np.zeros_like(centers)
        
        for i in range(N):
            for kk in range(1, 8):
                j = idx[i, kk]
                diff = centers[i] - centers[j]
                dist = dists[i, kk]
                if dist > 1e-10:
                    # Stronger repulsion for closer pairs
                    weight = 1.0 / (dist * dist)
                    grad[i] += weight * diff
            
            # Wall repulsion
            for dim in range(2):
                val = centers[i, dim]
                if val < 0:
                    wall_dist = 0.5 + val
                    if wall_dist < 0.2:
                        grad[i, dim] += 2.0 / (wall_dist + 1e-6)
                else:
                    wall_dist = 0.5 - val
                    if wall_dist < 0.2:
                        grad[i, dim] -= 2.0 / (wall_dist + 1e-6)
        
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-10:
            grad /= grad_norm
        
        centers += current_lr * grad
        centers = _project(centers)
        
        if no_improve_count > 50:
            current_lr *= 0.9
            no_improve_count = 0
            if current_lr < 1e-8:
                break
    
    return best_centers, best_t

def _local_search(centers, max_iter=3000, step_scale=0.008):
    """Local search using perturbation of most constrained points."""
    centers = centers.copy()
    best_r = _compute_radius(centers)
    best_centers = centers.copy()
    
    tree = cKDTree(centers)
    
    for it in range(max_iter):
        if _time_left() < 30.0:
            break
        
        # Find most constrained points
        dists, idx = tree.query(centers, k=3)
        nn_dists = dists[:, 1]
        wall_dists = np.min(np.minimum(0.5 - centers, centers + 0.5), axis=1)
        
        n_perturb = 2 + (it % 3)
        candidates = np.unique(np.concatenate([
            np.argsort(nn_dists)[:n_perturb],
            np.argsort(wall_dists)[:n_perturb]
        ])[:n_perturb])
        
        improvements = 0
        for _ in range(8):
            trial = centers.copy()
            for idx in candidates:
                angle = np.random.uniform(0, 2 * np.pi)
                step = step_scale * (0.5 + np.random.rand())
                trial[idx, 0] += step * np.cos(angle)
                trial[idx, 1] += step * np.sin(angle)
            
            trial = _project(trial)
            
            r_trial = _compute_radius(trial)
            if r_trial > best_r + 1e-12:
                best_r = r_trial
                best_centers = trial.copy()
                centers = trial.copy()
                tree = cKDTree(centers)
                step_scale *= 0.97
                improvements += 1
        
        if improvements == 0:
            step_scale *= 0.995
            if step_scale < 1e-7:
                break
    
    return best_centers, best_r

def construct():
    """Construct 200 circle centers in [-0.5, 0.5]^2 maximizing the minimum radius."""
    # Start with best hexagonal grid
    centers = _generate_best_hex()
    r = _compute_radius(centers)
    
    # Phase 1: Projected gradient descent with momentum (main optimization)
    if _time_left() > 40.0:
        centers, r = _pgrad_momentum(centers, max_iter=8000, lr=0.004, momentum=0.9)
    
    # Phase 2: Local search for fine-tuning
    if _time_left() > 35.0:
        centers, r = _local_search(centers, max_iter=4000, step_scale=0.006)
    
    # Phase 3: Gradient refinement with smooth surrogate
    if _time_left() > 30.0:
        centers, r = _gradient_refine(centers, max_iter=4000, lr=0.001, beta=50.0)
    
    # Phase 4: Final high-precision polish
    if _time_left() > 25.0:
        for lr in [0.0005, 0.0002, 0.0001]:
            if _time_left() < 22.0:
                break
            centers, r_new = _gradient_refine(centers, max_iter=2000, lr=lr, beta=80.0)
            if r_new > r:
                r = r_new
    
    # Final verification and clipping
    centers = np.clip(centers, -0.5, 0.5)
    
    return centers
# EVOLVE-BLOCK-END
