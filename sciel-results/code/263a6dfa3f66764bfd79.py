import numpy as np
import time
from scipy.optimize import minimize

N = 300

def _compute_radius(x):
    """
    x: (N, 2) array of points.
    Returns the maximum common radius.
    """
    # Pairwise distances
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    min_dist = np.min(dist)
    
    # Boundary distances
    norms = np.linalg.norm(x, axis=1)
    min_boundary = np.min(1.0 - norms)
    
    return min(min_dist / 2.0, min_boundary)

def _objective_and_grad(x, active_tol=1e-6):
    """
    Objective: -radius(x)
    Gradient: -grad(radius(x))
    """
    x = x.reshape(-1, 2)
    N = len(x)
    
    # Compute distances
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    
    # Boundary
    norms = np.linalg.norm(x, axis=1)
    boundary = 1.0 - norms
    
    r_pair = dist / 2.0
    r_bound = boundary
    
    min_pair = np.min(r_pair)
    min_bound = np.min(r_bound)
    
    r = min(min_pair, min_bound)
    
    # Gradient calculation
    grad = np.zeros_like(x)
    
    # Active pair constraints
    tri_idx = np.triu_indices(N, k=1)
    pair_vals = r_pair[tri_idx]
    active_pairs = np.where(pair_vals <= min_pair + active_tol)[0]
    
    if len(active_pairs) > 0:
        i_idx = tri_idx[0][active_pairs]
        j_idx = tri_idx[1][active_pairs]
        
        diffs = x[i_idx] - x[j_idx]
        dists = dist[i_idx, j_idx]
        
        mask_non_zero = dists > 1e-10
        diffs = diffs[mask_non_zero]
        dists = dists[mask_non_zero]
        i_idx = i_idx[mask_non_zero]
        j_idx = j_idx[mask_non_zero]
        
        if len(diffs) > 0:
            grads_i = diffs / (2.0 * dists)[:, np.newaxis]
            grads_j = -grads_i
            
            for k in range(len(i_idx)):
                grad[i_idx[k]] += grads_i[k]
                grad[j_idx[k]] += grads_j[k]
                
    # Active boundary constraints
    active_bounds = np.where(boundary <= min_bound + active_tol)[0]
    if len(active_bounds) > 0:
        x_active = x[active_bounds]
        norms_active = norms[active_bounds]
        mask_non_zero = norms_active > 1e-10
        x_active = x_active[mask_non_zero]
        norms_active = norms_active[mask_non_zero]
        active_bounds = active_bounds[mask_non_zero]
        
        if len(x_active) > 0:
            grads_b = -x_active / norms_active[:, np.newaxis]
            for k in range(len(active_bounds)):
                grad[active_bounds[k]] += grads_b[k]
                
    obj = -r
    grad = -grad
    
    return obj, grad.ravel()

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # 1. Initialization: Hexagonal Lattice
    s = np.sqrt(2.0 * np.pi / (N * np.sqrt(3.0)))
    
    k = int(1.0 / s) + 3
    rows = [((i + 0.5 * (j % 2)) * s, j * s * np.sqrt(3.0) / 2.0)
            for i in range(-k, k + 1) for j in range(-k, k + 1)]
    points = np.array(rows)
    radius = np.hypot(points[:, 0], points[:, 1])
    points, radius = points[radius <= 1.0 - s / 2.0], radius[radius <= 1.0 - s / 2.0]
    
    if len(points) >= N:
        points = points[np.argsort(radius, kind="stable")[:N]]
    else:
        while len(points) < N:
            s *= 0.95
            k = int(1.0 / s) + 3
            rows = [((i + 0.5 * (j % 2)) * s, j * s * np.sqrt(3.0) / 2.0)
                    for i in range(-k, k + 1) for j in range(-k, k + 1)]
            points = np.array(rows)
            radius = np.hypot(points[:, 0], points[:, 1])
            points, radius = points[radius <= 1.0 - s / 2.0], radius[radius <= 1.0 - s / 2.0]
            if len(points) >= N:
                points = points[np.argsort(radius, kind="stable")[:N]]
            if len(points) < N:
                points = np.random.rand(N, 2) * 0.8 - 0.4
                break

    x = points.copy()
    best_x = x.copy()
    best_r = _compute_radius(x)
    
    # 2. Local Optimization Loop
    for _ in range(10):
        if time.time() > deadline - 10.0:
            break
            
        def obj_func(x_flat):
            return _objective_and_grad(x_flat)[0]
            
        def grad_func(x_flat):
            return _objective_and_grad(x_flat)[1]
            
        res = minimize(
            obj_func, 
            x, 
            jac=grad_func, 
            method='L-BFGS-B', 
            options={'maxiter': 500, 'gtol': 1e-8, 'maxls': 50}
        )
        
        x_new = res.x.reshape(-1, 2)
        r_new = _compute_radius(x_new)
        
        if r_new > best_r:
            best_r = r_new
            best_x = x_new.copy()
            x = x_new.copy()
            
        if time.time() > deadline - 20.0:
            break
            
        noise = np.random.normal(0, 0.001, x.shape)
        x = x + noise
        
    return best_x
