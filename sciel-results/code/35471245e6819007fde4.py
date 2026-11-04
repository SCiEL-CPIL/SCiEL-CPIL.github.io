import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import time

N = 100

def _compute_radius(centers):
    """
    Computes the exact maximum common radius for a given set of centers.
    Centers: (N, 3) array.
    Cube: [-0.5, 0.5]^3.
    """
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
        
    # Wall constraints: distance from center to nearest wall
    max_abs = np.max(np.abs(centers), axis=1)
    wall_dists = 0.5 - max_abs
    min_wall_dist = np.min(wall_dists)
    
    if min_wall_dist < -1e-10: # Allow small numerical error
        return 0.0

    # Pairwise constraints
    dists = cdist(centers, centers)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    r = min(min_pair_dist / 2.0, min_wall_dist)
    return max(0.0, r)

def _soft_min_obj_grad(x, beta):
    """
    Computes the negative smooth minimum radius and its gradient.
    Objective: maximize r = min(constraints).
    We minimize -r_smooth = (1/beta) * log(sum(exp(-beta * d_i)))
    
    d_i are the constraints:
    - Pairwise distances / 2
    - Wall distances
    """
    centers = x.reshape(-1, 3)
    n = len(centers)
    
    # 1. Compute pairwise distances (upper triangle)
    # To save memory and time, we can compute this in chunks or use cdist if N is small enough.
    # N=100 is small enough for cdist.
    dists = cdist(centers, centers)
    np.fill_diagonal(dists, np.inf)
    
    i_idx, j_idx = np.triu_indices(n, k=1)
    pair_dists = dists[i_idx, j_idx] / 2.0
    pair_dists_sq = (dists[i_idx, j_idx] ** 2) / 4.0
    
    # 2. Compute wall distances
    max_abs = np.max(np.abs(centers), axis=1)
    wall_dists = 0.5 - max_abs
    
    # Combine all distances
    all_dists = np.concatenate([pair_dists, wall_dists])
    num_pairs = len(pair_dists)
    num_walls = len(wall_dists)
    
    # Shift to prevent overflow in exp
    min_d = np.min(all_dists)
    if min_d < 1e-12:
        # If any distance is near zero, the objective is dominated by it.
        # Return a large penalty and gradient pointing away from the collision.
        # For stability, we just return the current value and a zero gradient if too close,
        # but L-BFGS-B handles bounds.
        # Let's use a small epsilon to avoid log(0) or exp issues.
        min_d = 1e-12
        
    shifted = all_dists - min_d
    # exp(-beta * shifted) is stable if shifted >= 0
    exp_terms = np.exp(-beta * shifted)
    sum_exp = np.sum(exp_terms)
    
    if sum_exp < 1e-300:
        return 0.0, np.zeros_like(x)
    
    weights = exp_terms / sum_exp
    soft_min_val = min_d - np.log(sum_exp) / beta
    
    # Gradient calculation
    grad = np.zeros_like(x)
    
    # 1. Gradient of pairwise distances
    # d(pair_dist)/dc_i = -(c_i - c_j) / (2 * d_ij)
    # d(pair_dist)/dc_j = (c_i - c_j) / (2 * d_ij)
    # Note: pair_dist = d_ij / 2. So d(pair_dist)/dc_i = -0.5 * (c_i - c_j) / d_ij
    # Wait, d_ij = |c_i - c_j|. d(d_ij)/dc_i = (c_i - c_j) / d_ij.
    # d(pair_dist)/dc_i = 0.5 * (c_i - c_j) / d_ij.
    
    # We need to accumulate: grad += -weights * d(all_dists)/dx
    # Since we are minimizing -soft_min, the gradient of the objective is - (gradient of soft_min).
    # grad_soft_min = sum(weights_k * grad(d_k)/dx)
    # grad_obj = - grad_soft_min
    
    # Let's compute grad_soft_min first.
    grad_soft = np.zeros_like(centers)
    
    # Pairwise part
    # Vectorized update is tricky due to the sparse nature of the gradient for each pair.
    # Loop is acceptable for N=100 (4950 pairs).
    for k in range(num_pairs):
        w = weights[k]
        if w < 1e-10:
            continue
        i = i_idx[k]
        j = j_idx[k]
        d_ij = 2.0 * pair_dists[k]
        if d_ij < 1e-10:
            continue
        vec = centers[i] - centers[j]
        # d(pair_dist)/dc_i = -0.5 * vec / d_ij
        # d(pair_dist)/dc_j = 0.5 * vec / d_ij
        grad_soft[i] += w * (-0.5 * vec / d_ij)
        grad_soft[j] += w * (0.5 * vec / d_ij)
        
    # Wall part
    # wall_dist = 0.5 - max(|x|, |y|, |z|)
    # d(wall_dist)/dc_i[k] = -sign(c_i[k]) if |c_i[k]| is the max
    for i in range(n):
        w = weights[num_pairs + i]
        if w < 1e-10:
            continue
        max_val = max_abs[i]
        if max_val < 1e-10:
            continue
        # Find the axis with the max absolute value
        # If there are ties, we can distribute the gradient or pick one. 
        # The subgradient is the convex hull. Picking one is sufficient for L-BFGS.
        # Let's pick the first one.
        for k in range(3):
            if np.isclose(np.abs(centers[i, k]), max_val, atol=1e-12):
                grad_soft[i, k] += w * (-np.sign(centers[i, k]))
                break
                
    # Objective is -soft_min
    obj_val = -soft_min_val
    grad_val = -grad_soft.flatten()
    
    return obj_val, grad_val

def _optimize_start(centers, beta_start=10.0, beta_end=1000.0, max_iter=200, time_budget=10.0):
    """
    Optimizes the centers starting from the given configuration.
    Uses an adaptive beta to sharpen the minimum.
    """
    start_time = time.time()
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    # We will run L-BFGS-B with a fixed beta for a while, then increase beta.
    # Or we can just use a high beta from the start if the initialization is good.
    # Let's try a few stages.
    
    current_centers = centers.copy()
    best_radius = _compute_radius(current_centers)
    
    # Stage 1: Coarse optimization with lower beta
    # Stage 2: Fine optimization with higher beta
    
    betas = [50.0, 200.0, 500.0, 1000.0]
    
    for beta in betas:
        if time.time() > start_time + time_budget - 1.0:
            break
            
        def obj_grad(x):
            return _soft_min_obj_grad(x, beta)
        
        try:
            res = minimize(
                obj_grad,
                current_centers.flatten(),
                jac=True,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter, 'disp': False, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            new_centers = res.x.reshape(-1, 3)
            # Clip to ensure validity
            new_centers = np.clip(new_centers, -0.499999, 0.499999)
            
            new_radius = _compute_radius(new_centers)
            if new_radius > best_radius:
                best_radius = new_radius
                current_centers = new_centers
        except Exception:
            pass
            
    return current_centers, best_radius

def construct():
    start_time = time.time()
    deadline = start_time + 130.0 # 20s safety margin
    
    best_radius = 0.0
    best_centers = None
    
    # Generate initial configurations
    inits = []
    
    # 1. Grid 5x4x5
    def make_grid(nx, ny, nz):
        xs = np.linspace(-0.5 + 0.5/nx, 0.5 - 0.5/nx, nx)
        ys = np.linspace(-0.5 + 0.5/ny, 0.5 - 0.5/ny, ny)
        zs = np.linspace(-0.5 + 0.5/nz, 0.5 - 0.5/nz, nz)
        
        grid = []
        for z in zs:
            for y in ys:
                for x in xs:
                    grid.append([x, y, z])
        return np.array(grid)
        
    # Standard grid
    inits.append(make_grid(5, 4, 5))
    inits.append(make_grid(4, 5, 5))
    inits.append(make_grid(5, 5, 4))
    
    # Perturbed grids
    rng = np.random.RandomState(42)
    base_grid = make_grid(5, 4, 5)
    for scale in [0.01, 0.02, 0.03]:
        noisy = base_grid + rng.normal(0, scale, size=(N, 3))
        noisy = np.clip(noisy, -0.45, 0.45)
        inits.append(noisy)
        
    # Random initialization
    for i in range(5):
        rc = rng.uniform(-0.45, 0.45, size=(N, 3))
        inits.append(rc)
        
    # Lattice-like initialization
    # Try to place points on a finer grid and select the best N
    def make_lattice_init(target_r):
        spacing = 2.0 * target_r * 0.95
        if spacing < 0.05:
            spacing = 0.05
        n_pts_per_dim = int(1.0 / spacing) + 1
        if n_pts_per_dim < 3:
            n_pts_per_dim = 3
        if n_pts_per_dim > 15:
            n_pts_per_dim = 15
            
        xs = np.linspace(-0.5 + spacing/2, 0.5 - spacing/2, n_pts_per_dim)
        lattice = []
        for z in xs:
            for y in xs:
                for x in xs:
                    lattice.append([x, y, z])
        lattice = np.array(lattice)
        
        # Filter points that are too close to walls
        max_abs = np.max(np.abs(lattice), axis=1)
        lattice = lattice[max_abs <= 0.5 - target_r]
        
        if len(lattice) < N:
            return None
            
        # Greedy selection: pick points that are farthest from already-selected
        selected = []
        remaining = np.arange(len(lattice))
        
        while len(selected) < N and len(remaining) > 0:
            if len(selected) == 0:
                # Start from center
                dists_to_center = np.sum(lattice[remaining]**2, axis=1)
                idx = remaining[np.argmin(dists_to_center)]
            else:
                # Pick point that maximizes minimum distance to selected
                min_dists = np.full(len(remaining), np.inf)
                for sel_idx in selected[-10:]:
                    d = np.sqrt(np.sum((lattice[remaining] - lattice[sel_idx])**2, axis=1))
                    min_dists = np.minimum(min_dists, d)
                idx = remaining[np.argmax(min_dists)]
            
            selected.append(idx)
            remaining = remaining[remaining != idx]
            
            if len(remaining) < N - len(selected):
                break
        
        if len(selected) < N:
            return None
            
        return lattice[selected]

    for tr in [0.10, 0.105, 0.11]:
        lat_init = make_lattice_init(tr)
        if lat_init is not None:
            inits.append(lat_init)
            
    # Optimize each initialization
    for init in inits:
        if time.time() > deadline - 15.0:
            break
            
        # Time budget per start
        time_left = deadline - 15.0 - time.time()
        if time_left <= 0:
            break
            
        # Allocate time based on number of remaining inits
        # Simpler: fixed time budget per init, reduced if running low
        budget_per_init = min(15.0, time_left / max(1, len(inits)))
        
        optimized_centers, radius = _optimize_start(init, time_budget=budget_per_init)
        
        if radius > best_radius:
            best_radius = radius
            best_centers = optimized_centers.copy()
            
    # Final polish on the best result
    if best_centers is not None and time.time() < deadline - 5.0:
        polish_budget = deadline - 5.0 - time.time()
        # Use a very high beta for final polish
        def final_obj_grad(x):
            return _soft_min_obj_grad(x, 2000.0)
            
        bounds = [(-0.5, 0.5)] * (N * 3)
        try:
            res = minimize(
                final_obj_grad,
                best_centers.flatten(),
                jac=True,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'disp': False, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            polished = res.x.reshape(-1, 3)
            polished = np.clip(polished, -0.499999, 0.499999)
            r_polished = _compute_radius(polished)
            if r_polished > best_radius:
                best_radius = r_polished
                best_centers = polished
        except Exception:
            pass

    if best_centers is None:
        best_centers = make_grid(5, 4, 5)
    else:
        best_centers = np.clip(best_centers, -0.49999, 0.49999)
        if not np.all(np.isfinite(best_centers)):
            best_centers = make_grid(5, 4, 5)
            
    return best_centers
