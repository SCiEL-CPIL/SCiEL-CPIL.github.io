"""Rewritten program for packing 200 equal circles in a unit square."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist, cdist
from scipy.spatial import cKDTree


def _compute_radius(c):
    """Compute the largest common radius for given centres."""
    n = c.shape[0]
    if n == 0:
        return 0.0
    boundary_dists = 0.5 - np.abs(c)
    min_boundary = np.min(boundary_dists)
    
    if n < 2:
        return min_boundary
    
    dists = pdist(c, metric='euclidean')
    min_pair_dist = np.min(dists)
    
    return min(min_pair_dist / 2.0, min_boundary)


def _active_constraints(c, k=5):
    """Find the k nearest neighbors for each point (active constraints)."""
    n = c.shape[0]
    tree = cKDTree(c)
    dists, indices = tree.query(c, k=k+1)
    # indices[i][0] is the point itself, indices[i][1:] are neighbors
    return dists[:, 1:], indices[:, 1:]


def _gradient_min_dist(c, k=5):
    """Compute gradient of the minimum distance (soft version focusing on active constraints)."""
    n = c.shape[0]
    grad = np.zeros_like(c)
    
    # Boundary gradients
    boundary_dists = 0.5 - np.abs(c)
    min_boundary_idx = np.argmin(boundary_dists)
    
    # Pairwise distances using k-nearest neighbors
    tree = cKDTree(c)
    dists, indices = tree.query(c, k=k+1)
    
    min_pair_dist = np.inf
    min_pair_i, min_pair_j = -1, -1
    
    for i in range(n):
        for j_idx in range(1, k+1):
            j = indices[i, j_idx]
            if j > i:  # avoid double counting
                d = dists[i, j_idx]
                if d < min_pair_dist:
                    min_pair_dist = d
                    min_pair_i, min_pair_j = i, j
    
    # Determine which constraint is active
    if min_pair_dist / 2.0 < np.min(boundary_dists):
        # Pairwise constraint is active
        r = min_pair_dist / 2.0
        if r > 0:
            diff = c[min_pair_i] - c[min_pair_j]
            norm = np.linalg.norm(diff)
            if norm > 1e-15:
                unit = diff / norm
                grad[min_pair_i] += unit
                grad[min_pair_j] -= unit
    else:
        # Boundary constraint is active
        r = np.min(boundary_dists)
        i = min_boundary_idx
        for dim in range(2):
            if c[i, dim] > 0:
                grad[i, dim] -= 1.0
            elif c[i, dim] < 0:
                grad[i, dim] += 1.0
    
    return grad, r


def _project_to_square(c, margin=1e-12):
    """Project points into the square [-0.5, 0.5]^2 with margin."""
    return np.clip(c, -0.5 + margin, 0.5 - margin)


def _projected_gradient_optimize(c0, max_time=30.0, k=5):
    """Optimize using projected gradient method on the min-distance objective."""
    start_time = time.time()
    c = c0.copy()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    # Adaptive step size
    step_size = 0.01
    min_step = 1e-6
    max_step = 0.05
    
    iter_count = 0
    no_improve_count = 0
    
    while time.time() < start_time + max_time:
        iter_count += 1
        
        # Compute gradient and current radius
        grad, r = _gradient_min_dist(c, k=k)
        
        if np.linalg.norm(grad) < 1e-12:
            break
        
        # Take a step in the direction that increases the minimum distance
        # We want to maximize r, so we move in the direction of grad
        c_new = c + step_size * grad
        c_new = _project_to_square(c_new)
        
        r_new = _compute_radius(c_new)
        
        if r_new > best_r + 1e-12:
            best_c = c_new.copy()
            best_r = r_new
            c = c_new.copy()
            no_improve_count = 0
            # Increase step size slightly if we improved
            step_size = min(step_size * 1.05, max_step)
        else:
            no_improve_count += 1
            # Decrease step size
            step_size = max(step_size * 0.9, min_step)
            
            if no_improve_count > 50:
                # Try a different step size
                step_size = 0.01
                no_improve_count = 0
    
    return best_c, best_r


def _high_precision_polish(c0, max_time=20.0):
    """High-precision polish using very small steps on the exact objective."""
    start_time = time.time()
    c = c0.copy()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    step_size = 1e-4
    k = 3  # Only look at closest neighbors for precision
    
    while time.time() < start_time + max_time:
        grad, r = _gradient_min_dist(c, k=k)
        
        if np.linalg.norm(grad) < 1e-14:
            break
        
        # Try multiple step sizes
        improved = False
        for ss in [step_size, step_size * 0.5, step_size * 2.0]:
            c_trial = c + ss * grad
            c_trial = _project_to_square(c_trial)
            r_trial = _compute_radius(c_trial)
            
            if r_trial > best_r + 1e-14:
                best_c = c_trial.copy()
                best_r = r_trial
                c = c_trial.copy()
                step_size = ss
                improved = True
                break
        
        if not improved:
            step_size *= 0.5
            if step_size < 1e-8:
                break
    
    return best_c, best_r


def _initialize_hexagonal():
    """Initialize with a hexagonal pattern."""
    best_c = None
    best_r = 0
    
    for d in np.linspace(0.055, 0.075, 20):
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
    
    # Fallback: grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    best_radius = 0.0
    best_c = None
    
    # Start with hexagonal initialization
    c_hex = _initialize_hexagonal()
    r_hex = _compute_radius(c_hex)
    if r_hex > best_radius:
        best_radius = r_hex
        best_c = c_hex.copy()
    
    # Projected gradient optimization from hexagonal start
    if time.time() < deadline - 30:
        max_time_pgo = min(40.0, deadline - time.time() - 15)
        c_opt, r_opt = _projected_gradient_optimize(c_hex, max_time=max_time_pgo)
        if r_opt > best_radius:
            best_radius = r_opt
            best_c = c_opt.copy()
    
    # High-precision polish
    if time.time() < deadline - 10:
        max_time_polish = min(15.0, deadline - time.time() - 5)
        c_polished, r_polished = _high_precision_polish(best_c, max_time=max_time_polish)
        if r_polished > best_radius:
            best_radius = r_polished
            best_c = c_polished.copy()
    
    # Try a few perturbed starts with projected gradient
    np.random.seed(42)
    n_trials = 3
    for trial in range(n_trials):
        if time.time() > deadline - 20:
            break
        
        # Perturb the best solution found so far
        if best_c is not None:
            c_pert = best_c + np.random.normal(0, 0.003, size=(N, 2))
            c_pert = _project_to_square(c_pert)
        else:
            c_pert = np.random.uniform(-0.45, 0.45, size=(N, 2))
            c_pert = _project_to_square(c_pert)
        
        r_pert = _compute_radius(c_pert)
        if r_pert > best_radius:
            best_radius = r_pert
            best_c = c_pert.copy()
        
        if time.time() < deadline - 15:
            max_time_pgo = min(15.0, deadline - time.time() - 10)
            c_opt, r_opt = _projected_gradient_optimize(c_pert, max_time=max_time_pgo)
            if r_opt > best_radius:
                best_radius = r_opt
                best_c = c_opt.copy()
        
        # Quick polish if time allows
        if time.time() < deadline - 8:
            max_time_polish = min(5.0, deadline - time.time() - 5)
            c_polished, r_polished = _high_precision_polish(best_c, max_time=max_time_polish)
            if r_polished > best_radius:
                best_radius = r_polished
                best_c = c_polished.copy()
    
    if best_c is None:
        best_c = _initialize_hexagonal()
    
    best_c = np.clip(best_c, -0.5, 0.5)
    
    return best_c
# EVOLVE-BLOCK-END
