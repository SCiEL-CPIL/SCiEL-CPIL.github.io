import numpy as np
import time
from scipy.optimize import minimize

N = 300

def _compute_score(points):
    """Compute the actual packing radius for given points."""
    if points.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    
    # Check if all points are inside unit circle
    radii = np.hypot(points[:, 0], points[:, 1])
    if np.any(radii > 1.0):
        return 0.0
    
    # Compute pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    min_wall_dist = np.min(1.0 - radii)
    
    return min(min_pair_dist / 2.0, min_wall_dist)

def _soft_min_and_grad(points, beta=50.0):
    """
    Compute soft-min of all constraints (pair distances and wall distances)
    and its gradient. Uses log-sum-exp for numerical stability.
    
    Returns: (soft_min_value, gradient)
    """
    n = points.shape[0]
    
    # Pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    dists = np.sqrt(dists_sq)
    
    # Wall distances: 1 - |c_i|
    radii = np.hypot(points[:, 0], points[:, 1])
    wall_dists = 1.0 - radii
    
    # Combine all constraints into one vector
    # Pair constraints: dists[i,j] for i < j (each pair counted once)
    # Wall constraints: wall_dists[i]
    
    # For soft-min, we want to minimize the minimum of all constraints
    # soft_min(x) = -log(sum(exp(-beta * x))) / beta
    # gradient: d(soft_min)/dx_i = -sum_j exp(-beta*x_j) * (-beta) * (dx_j/dx_i) / sum_k exp(-beta*x_k)
    #           = sum_j exp(-beta*x_j) * (dx_j/dx_i) / sum_k exp(-beta*x_k)
    
    # To avoid overflow, use log-sum-exp trick
    # Let's work with the negative: maximize soft_max(-x) = log(sum(exp(beta*(-x)))) / beta
    # Actually, let's just compute directly with careful scaling
    
    # All constraint values
    # Pair constraints (upper triangle only to avoid double counting)
    iu = np.triu_indices(n, k=1)
    pair_dists = dists[iu]
    
    # Wall constraints
    wall = wall_dists
    
    # Combine
    all_constraints = np.concatenate([pair_dists, wall])
    
    # Soft-min via log-sum-exp
    # soft_min = -log(sum(exp(-beta * c))) / beta
    neg_beta_c = -beta * all_constraints
    # Subtract max for stability
    max_neg_beta_c = np.max(neg_beta_c)
    exp_terms = np.exp(neg_beta_c - max_neg_beta_c)
    sum_exp = np.sum(exp_terms)
    
    soft_min_val = -(np.log(sum_exp) + max_neg_beta_c) / beta
    
    # Gradient
    # d(soft_min)/d(points) 
    # soft_min = -log(sum(exp(-beta*c))) / beta
    # d(soft_min)/dc_k = -1/beta * (1/sum_exp) * exp(-beta*c_k) * (-beta) = exp(-beta*c_k) / sum_exp
    # Then d(soft_min)/d(points) = sum_k (d(soft_min)/dc_k) * (dc_k/d(points))
    
    weights = exp_terms / sum_exp
    
    # Gradient from pair constraints
    grad = np.zeros_like(points)
    
    # For each pair (i,j), constraint c = dist(i,j)
    # dc/d(points[i]) = (points[i] - points[j]) / dist
    # dc/d(points[j]) = (points[j] - points[i]) / dist
    
    # weights[:len(pair_dists)] correspond to pairs in iu
    pair_weights = weights[:len(pair_dists)]
    
    # Accumulate gradient from pairs
    # For pair (i,j) with weight w:
    # grad[i] += w * (points[i] - points[j]) / dist[i,j]
    # grad[j] += w * (points[j] - points[i]) / dist[i,j]
    
    i_idx, j_idx = iu
    pair_w = pair_weights
    safe_dists = np.where(pair_dists < 1e-10, 1e-10, pair_dists)
    
    # Vectorized accumulation using np.add.at
    dir_ij = (points[i_idx] - points[j_idx]) / safe_dists[:, np.newaxis]
    
    np.add.at(grad, i_idx, pair_w[:, np.newaxis] * dir_ij)
    np.add.at(grad, j_idx, -pair_w[:, np.newaxis] * dir_ij)
    
    # Gradient from wall constraints
    # c_i = 1 - |points[i]|
    # dc_i/d(points[i]) = -points[i] / |points[i]|
    wall_weights = weights[len(pair_dists):]
    safe_radii = np.where(radii < 1e-10, 1e-10, radii)
    wall_grad = -wall_weights[:, np.newaxis] * points / safe_radii[:, np.newaxis]
    grad += wall_grad
    
    return soft_min_val, grad

def _objective_and_grad(x, beta=50.0):
    """Objective function and gradient for scipy.optimize.minimize."""
    points = x.reshape(-1, 2)
    val, grad = _soft_min_and_grad(points, beta)
    # We want to maximize soft_min, so minimize -soft_min
    return -val, -grad

def _hex_lattice_init():
    """Generate hexagonal lattice initialization."""
    r_init = 0.05
    spacing = 2.0 * r_init
    points = []
    k = int(1.0 / spacing) + 5
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            points.append([x, y])
    
    points = np.array(points)
    dists = np.hypot(points[:, 0], points[:, 1])
    points = points[dists <= 1.0 - r_init]
    
    if len(points) > N:
        dists = np.hypot(points[:, 0], points[:, 1])
        idx = np.argsort(dists)
        points = points[idx[:N]]
    elif len(points) < N:
        points = np.random.uniform(-0.9, 0.9, (N, 2))
    
    if len(points) < N:
        extra = N - len(points)
        new_pts = np.random.uniform(-0.5, 0.5, (extra, 2))
        points = np.vstack((points, new_pts))
    else:
        points = points[:N]
    
    points += np.random.uniform(-1e-5, 1e-5, points.shape)
    return points

def _radial_ring_init():
    """Generate radial ring initialization."""
    # Place circles in concentric rings
    # Center: 1 circle
    # Ring k: approximately 6k circles at radius ~ k * spacing
    points = []
    points.append([0.0, 0.0])
    
    ring = 1
    while len(points) < N:
        # Number of circles in this ring
        n_in_ring = 6 * ring
        if len(points) + n_in_ring > N:
            n_in_ring = N - len(points)
        
        # Radius for this ring (approximate)
        r_ring = ring * 0.11
        
        # Check if ring fits in unit circle
        if r_ring > 0.95:
            break
            
        for i in range(n_in_ring):
            angle = 2.0 * np.pi * i / n_in_ring
            x = r_ring * np.cos(angle)
            y = r_ring * np.sin(angle)
            points.append([x, y])
        
        ring += 1
    
    points = np.array(points)
    if len(points) < N:
        # Fill remaining with random points
        extra = N - len(points)
        new_pts = np.random.uniform(-0.8, 0.8, (extra, 2))
        points = np.vstack((points, new_pts))
    else:
        points = points[:N]
    
    points += np.random.uniform(-1e-5, 1e-5, points.shape)
    return points

def _local_optimize(points, beta_start=20.0, beta_end=200.0, max_iter=1000, time_limit=None):
    """
    Local optimization using L-BFGS-B with increasing beta (annealing).
    """
    start_time = time.time()
    
    # Start with low beta (smooth) and increase to high beta (sharp)
    n_steps = 5
    betas = np.geomspace(beta_start, beta_end, n_steps)
    
    current_points = points.copy()
    
    for beta in betas:
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        
        # Define objective and gradient for this beta
        def obj_grad(x):
            p = x.reshape(-1, 2)
            val, grad = _soft_min_and_grad(p, beta)
            return -val, -grad
        
        # Use L-BFGS-B
        x0 = current_points.flatten()
        
        # Run optimization
        result = minimize(
            lambda x: obj_grad(x)[0],
            x0,
            jac=lambda x: obj_grad(x)[1],
            method='L-BFGS-B',
            options={'maxiter': max_iter // n_steps, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        current_points = result.x.reshape(-1, 2)
        
        # Ensure points stay inside unit circle
        radii = np.hypot(current_points[:, 0], current_points[:, 1])
        mask = radii > 1.0
        if np.any(mask):
            current_points[mask] /= radii[mask, np.newaxis]
    
    return current_points

def _force_directed_refine(points, r_target, max_steps=5000, step_size=0.01):
    """
    Force-directed refinement to push radius toward target.
    """
    current_points = points.copy()
    step = step_size
    
    for _ in range(max_steps):
        # Compute current radius
        diff = current_points[:, np.newaxis, :] - current_points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        radii = np.hypot(current_points[:, 0], current_points[:, 1])
        min_wall_dist = np.min(1.0 - radii)
        
        current_r = min(min_pair_dist / 2.0, min_wall_dist)
        
        if current_r >= r_target:
            break
        
        # Compute forces
        safe_dists = np.where(dists < 1e-8, 1e-8, dists)
        violation = np.maximum(0, 2.0 * r_target - dists)
        unit_vec = diff / safe_dists[:, :, np.newaxis]
        forces = np.sum(violation[:, :, np.newaxis] * unit_vec, axis=1)
        
        # Wall forces
        wall_violation = np.maximum(0, radii - (1.0 - r_target))
        safe_radii = np.where(radii < 1e-8, 1e-8, radii)
        wall_forces = -wall_violation[:, np.newaxis] * (current_points / safe_radii[:, np.newaxis])
        
        total_forces = forces + wall_forces
        current_points += step * total_forces
        
        # Keep inside unit circle
        radii = np.hypot(current_points[:, 0], current_points[:, 1])
        mask = radii > 1.0
        if np.any(mask):
            current_points[mask] /= radii[mask, np.newaxis]
        
        # Anneal step size
        step *= 0.999
    
    return current_points

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin for 150s limit
    
    # Try multiple initializations and keep the best
    best_points = None
    best_score = 0.0
    
    # Initialization strategies
    inits = []
    
    # Hexagonal lattice
    hex_pts = _hex_lattice_init()
    inits.append(hex_pts)
    
    # Radial rings
    ring_pts = _radial_ring_init()
    inits.append(ring_pts)
    
    # Random
    rand_pts = np.random.uniform(-0.9, 0.9, (N, 2))
    inits.append(rand_pts)
    
    # For each initialization, run optimization
    for init_pts in inits:
        if time.time() > deadline:
            break
        
        # Phase 1: Smooth optimization with L-BFGS-B
        time_remaining = deadline - time.time()
        if time_remaining < 10:
            break
        
        optimized_pts = _local_optimize(
            init_pts, 
            beta_start=10.0, 
            beta_end=100.0,
            max_iter=500,
            time_limit=time_remaining * 0.5
        )
        
        # Phase 2: Force-directed refinement to push radius higher
        time_remaining = deadline - time.time()
        if time_remaining < 5:
            break
        
        # Estimate current radius
        diff = optimized_pts[:, np.newaxis, :] - optimized_pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        radii = np.hypot(optimized_pts[:, 0], optimized_pts[:, 1])
        min_wall_dist = np.min(1.0 - radii)
        current_r = min(min_pair_dist / 2.0, min_wall_dist)
        
        # Try to push to slightly higher radius
        target_r = current_r * 1.005
        refined_pts = _force_directed_refine(
            optimized_pts,
            r_target=target_r,
            max_steps=2000,
            step_size=0.005
        )
        
        # Phase 3: Final polish with high beta
        time_remaining = deadline - time.time()
        if time_remaining < 3:
            break
        
        polished_pts = _local_optimize(
            refined_pts,
            beta_start=50.0,
            beta_end=200.0,
            max_iter=300,
            time_limit=time_remaining * 0.8
        )
        
        # Evaluate
        score = _compute_score(polished_pts)
        
        if score > best_score:
            best_score = score
            best_points = polished_pts.copy()
    
    # If no optimization ran, use the best initial
    if best_points is None:
        best_points = hex_pts
    
    # Final validation
    radii = np.hypot(best_points[:, 0], best_points[:, 1])
    mask = radii > 1.0
    if np.any(mask):
        best_points[mask] /= radii[mask, np.newaxis]
    
    return best_points
