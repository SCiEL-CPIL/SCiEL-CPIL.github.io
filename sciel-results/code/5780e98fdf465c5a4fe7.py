import numpy as np
import time
from scipy.optimize import minimize

N = 100
HALF = 0.5
DEADLINE_MARGIN = 10.0

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
    # The distance from a point (x,y,z) to the boundary of the cube [-0.5, 0.5]^3
    # is min(0.5 - x, 0.5 - y, 0.5 - z, -0.5 - x, -0.5 - y, -0.5 - z)
    # Which is equivalent to 0.5 - max(|x|, |y|, |z|)
    max_abs = np.max(np.abs(centers), axis=1)
    wall_dists = 0.5 - max_abs
    min_wall_dist = np.min(wall_dists)
    
    if min_wall_dist < -1e-10: # Allow small numerical error
        return 0.0

    # Pairwise constraints
    # Using broadcasting for clarity and speed with N=100
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    r = min(min_pair_dist / 2.0, min_wall_dist)
    return max(0.0, r)

def _adam_step(x, grad, state, lr, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    Performs one step of Adam optimization.
    x: current position
    grad: gradient
    state: dict with 'm', 'v', 't'
    lr: learning rate
    """
    t = state['t'] + 1
    state['t'] = t
    
    # Update moments
    m = beta1 * state['m'] + (1 - beta1) * grad
    v = beta2 * state['v'] + (1 - beta2) * (grad ** 2)
    
    # Bias correction
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    
    # Update
    x_new = x - lr * m_hat / (np.sqrt(v_hat) + eps)
    
    state['m'] = m
    state['v'] = v
    
    return x_new

def _soft_min_obj_grad(x, beta):
    """
    Computes the negative soft-minimum of effective distances and its gradient.
    Objective: maximize r = min(constraints).
    We minimize -r_smooth.
    
    Constraints:
    - Pairwise distances / 2
    - Wall distances
    """
    centers = x.reshape(-1, 3)
    n = len(centers)
    
    # 1. Compute pairwise distances
    # Using broadcasting
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    i_idx, j_idx = np.triu_indices(n, k=1)
    pair_dists = dists[i_idx, j_idx] / 2.0
    pair_dists_sq = (dists[i_idx, j_idx] ** 2) / 4.0
    
    # 2. Compute wall distances
    # wall_dist_i = 0.5 - max(|x_i|, |y_i|, |z_i|)
    max_abs = np.max(np.abs(centers), axis=1)
    wall_dists = 0.5 - max_abs
    
    # Combine all distances
    all_dists = np.concatenate([pair_dists, wall_dists])
    num_pairs = len(pair_dists)
    num_walls = len(wall_dists)
    
    # Shift to prevent overflow in exp
    min_d = np.min(all_dists)
    if min_d < 1e-12:
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
    grad_soft = np.zeros_like(centers)
    
    # 1. Gradient of pairwise distances
    # d(pair_dist)/dc_i = -0.5 * (c_i - c_j) / d_ij
    # d(pair_dist)/dc_j = 0.5 * (c_i - c_j) / d_ij
    # Vectorized accumulation
    valid = pair_dists > 1e-10
    if np.any(valid):
        i_v = i_idx[valid]
        j_v = j_idx[valid]
        d_v = pair_dists[valid]
        w_v = weights[valid]
        
        vecs = centers[i_v] - centers[j_v]
        # coeff for grad_i is -0.5 / d_ij, for grad_j is 0.5 / d_ij
        # Note: pair_dist = d_ij / 2. d(pair_dist)/dc_i = 0.5 * (c_i - c_j) / d_ij
        # Wait, d(d_ij)/dc_i = (c_i - c_j) / d_ij.
        # d(pair_dist)/dc_i = 0.5 * (c_i - c_j) / d_ij.
        
        coeffs = w_v / (2.0 * d_v) # w * 0.5 / d_ij
        
        # grad_soft[i] += coeff * vec
        np.add.at(grad_soft, i_v, coeffs[:, np.newaxis] * vecs)
        # grad_soft[j] -= coeff * vec
        np.add.at(grad_soft, j_v, -coeffs[:, np.newaxis] * vecs)
        
    # 2. Gradient of wall distances
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
        # The subgradient is the convex hull. Picking one is sufficient.
        # Let's pick the first one.
        for k in range(3):
            if np.isclose(np.abs(centers[i, k]), max_val, atol=1e-12):
                grad_soft[i, k] += w * (-np.sign(centers[i, k]))
                break
                
    # Objective is -soft_min
    obj_val = -soft_min_val
    grad_val = -grad_soft.flatten()
    
    return obj_val, grad_val

def _optimize_adam(centers, beta_start=10.0, beta_end=500.0, max_iter=2000, time_budget=10.0):
    """
    Optimizes the centers using Adam with adaptive beta.
    """
    start_time = time.time()
    
    current_centers = centers.copy()
    best_radius = _compute_radius(current_centers)
    
    # Initialize Adam state
    state = {
        'm': np.zeros_like(current_centers.flatten()),
        'v': np.zeros_like(current_centers.flatten()),
        't': 0
    }
    
    # Schedule beta
    # We'll use a geometric schedule for beta
    num_steps = max_iter
    if num_steps <= 1:
        return current_centers, best_radius
        
    betas = np.geomspace(beta_start, beta_end, num_steps)
    
    # Learning rate schedule
    # Start with a higher LR, decay it
    base_lr = 0.01
    
    for it in range(num_steps):
        if time.time() > start_time + time_budget:
            break
            
        beta = betas[it]
        
        # Decay learning rate
        # lr = base_lr * (0.999 ** it)
        lr = base_lr / (1.0 + 0.1 * it)
        
        x_flat = current_centers.flatten()
        obj, grad = _soft_min_obj_grad(x_flat, beta)
        
        # Check for NaN/Inf
        if not np.all(np.isfinite(grad)):
            break
            
        x_new_flat = _adam_step(x_flat, grad, state, lr)
        
        # Clip to bounds
        x_new_flat = np.clip(x_new_flat, -0.499999, 0.499999)
        
        new_centers = x_new_flat.reshape(-1, 3)
        new_radius = _compute_radius(new_centers)
        
        if new_radius > best_radius:
            best_radius = new_radius
            current_centers = new_centers
            
    return current_centers, best_radius

def _optimize_lbfgs(centers, beta, max_iter=100, time_budget=5.0):
    """
    Optimizes using L-BFGS-B for final polishing.
    """
    start_time = time.time()
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    def obj_grad(x):
        return _soft_min_obj_grad(x, beta)
        
    try:
        res = minimize(
            obj_grad,
            centers.flatten(),
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'disp': False, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        
        new_centers = res.x.reshape(-1, 3)
        new_centers = np.clip(new_centers, -0.499999, 0.499999)
        return new_centers
    except Exception:
        return centers

def _generate_initial_positions():
    """
    Generate initial positions using FCC lattice structure.
    """
    inits = []
    
    # 1. FCC Lattice
    # FCC unit cell vectors
    a = 0.12 # approximate spacing
    # Basis vectors for FCC
    # (0,0,0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)
    
    # Let's generate a grid of FCC points
    # We want roughly 100 points in [-0.5, 0.5]^3
    # Volume = 1.0. Number of points = 100.
    # Density = 100.
    # FCC packing fraction ~ 0.74.
    # Let's just generate a grid of points and filter.
    
    # Grid 5x4x5
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
        
    # FCC-like initialization
    # Generate points on FCC lattice
    # Lattice constant 'a' such that ~100 points fit in 1x1x1 cube
    # Number of points in FCC per unit volume = 4/a^3
    # 100 = 4/a^3 => a^3 = 0.04 => a ~ 0.34
    # But we need to fit in the cube, so let's try a few values.
    
    def make_fcc_init(a):
        # Generate points
        points = []
        # Range of unit cells
        # Let's go from -2 to 2 in each dimension
        for i in range(-2, 3):
            for j in range(-2, 3):
                for k in range(-2, 3):
                    # FCC basis
                    # (i, j, k)
                    points.append([i*a, j*a, k*a])
                    # (i+0.5, j+0.5, k)
                    points.append([(i+0.5)*a, (j+0.5)*a, k*a])
                    # (i+0.5, j, k+0.5)
                    points.append([(i+0.5)*a, j*a, (k+0.5)*a])
                    # (i, j+0.5, k+0.5)
                    points.append([i*a, (j+0.5)*a, (k+0.5)*a])
                    
        points = np.array(points)
        
        # Filter to cube
        mask = (np.abs(points) <= 0.45).all(axis=1)
        points = points[mask]
        
        if len(points) < N:
            return None
            
        # Select N points that maximize minimum distance
        # Greedy farthest point sampling
        selected = []
        remaining = np.arange(len(points))
        
        # Start from center
        dists_to_center = np.sum(points[remaining]**2, axis=1)
        idx = remaining[np.argmin(dists_to_center)]
        selected.append(idx)
        remaining = remaining[remaining != idx]
        
        while len(selected) < N and len(remaining) > 0:
            # Pick point that maximizes minimum distance to selected
            min_dists = np.full(len(remaining), np.inf)
            for sel_idx in selected:
                d = np.sqrt(np.sum((points[remaining] - points[sel_idx])**2, axis=1))
                min_dists = np.minimum(min_dists, d)
            idx = remaining[np.argmax(min_dists)]
            selected.append(idx)
            remaining = remaining[remaining != idx]
            
        if len(selected) < N:
            return None
            
        return points[selected]
        
    for a in [0.1, 0.12, 0.15, 0.2]:
        fcc_init = make_fcc_init(a)
        if fcc_init is not None:
            inits.append(fcc_init)
            
    return inits

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s safety margin
    
    best_radius = 0.0
    best_centers = None
    
    # Generate initial configurations
    inits = _generate_initial_positions()
    
    # Optimize each initialization
    for idx, init in enumerate(inits):
        if time.time() > deadline - 5.0:
            break
            
        time_left = deadline - 5.0 - time.time()
        if time_left <= 0:
            break
            
        # Allocate time based on number of remaining inits
        # More time for earlier inits? Or equal?
        # Let's give each init a fixed budget, but cap it.
        # 14 inits total. 140s / 14 ~ 10s per init.
        # But Adam is fast. Let's try 10s per init for Adam, then 5s for L-BFGS.
        
        # 1. Adam optimization
        budget_adam = min(10.0, time_left * 0.7)
        optimized_centers, radius = _optimize_adam(init, time_budget=budget_adam)
        
        # 2. L-BFGS polishing
        if time.time() < deadline - 5.0:
            budget_lbfgs = min(5.0, deadline - 5.0 - time.time())
            if budget_lbfgs > 0:
                polished_centers = _optimize_lbfgs(optimized_centers, beta=500.0, time_budget=budget_lbfgs)
                polished_radius = _compute_radius(polished_centers)
                if polished_radius > radius:
                    optimized_centers = polished_centers
                    radius = polished_radius
                    
        if radius > best_radius:
            best_radius = radius
            best_centers = optimized_centers.copy()
            
    # Final polish on the best result
    if best_centers is not None and time.time() < deadline - 2.0:
        polish_budget = deadline - 2.0 - time.time()
        if polish_budget > 0:
            # Use a very high beta for final polish
            polished_centers = _optimize_lbfgs(best_centers, beta=1000.0, time_budget=polish_budget)
            polished_radius = _compute_radius(polished_centers)
            if polished_radius > best_radius:
                best_radius = polished_radius
                best_centers = polished_centers

    if best_centers is None:
        best_centers = inits[0]
    else:
        best_centers = np.clip(best_centers, -0.49999, 0.49999)
        if not np.all(np.isfinite(best_centers)):
            best_centers = inits[0]
            
    return best_centers

if __name__ == "__main__":
    # Quick test
    centers = construct()
    r = _compute_radius(centers)
    print(f"Radius: {r}")
