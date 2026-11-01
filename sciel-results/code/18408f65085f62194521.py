"""Solver for packing 50 equal spheres in a unit cube [-0.5, 0.5]^3.

Strategy:
  1. Multiple diverse seeds (FCC-like, BCC-like, grid, random)
  2. Repulsion annealing to escape local minima
  3. Lloyd's algorithm (Voronoi centroid moves) to equalize clearance
  4. Smooth polish using L-BFGS-B with soft-min objective
  5. Final high-precision local search
"""
import numpy as np
import time
from scipy.optimize import minimize, linprog
from scipy.spatial import Voronoi
from scipy.spatial.distance import cdist

N = 50


def _compute_radius(c):
    """Compute the common radius for centres c (N,3) in cube [-0.5,0.5]^3."""
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    # Wall distances
    wall_dists = 0.5 - np.abs(c)
    min_wall = np.min(wall_dists)
    return min(min_pair / 2.0, min_wall)


def _softmin_radius_and_grad(c, beta=200.0):
    """Compute soft-min radius and gradient using log-sum-exp."""
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dists_sq, np.inf)
    dists = np.sqrt(dists_sq)
    
    # Wall distances
    wall_dists = 0.5 - np.abs(c)
    
    # Soft-min of all pair distances and wall distances
    # We want to maximize the minimum, so we minimize the negative soft-min
    all_dists = np.concatenate([dists[~np.isinf(dists)].flatten(), 
                                 (wall_dists.flatten())])
    
    # For pairs, the contribution is dist/2; for walls, it's dist
    pair_dists = dists[~np.isinf(dists)]
    pair_contrib = pair_dists / 2.0
    wall_contrib = wall_dists.flatten()
    
    all_contrib = np.concatenate([pair_contrib.flatten(), wall_contrib])
    
    # Soft-min: -log-sum-exp(-contrib / beta) * beta
    max_val = np.max(all_contrib)
    exp_terms = np.exp(-all_contrib / beta + max_val / beta)
    sum_exp = np.sum(exp_terms)
    softmin_val = max_val - beta * np.log(sum_exp)
    
    # Gradient w.r.t. c
    # d(softmin)/d(c_i) = -sum_j w_j * d(contrib_j)/d(c_i)
    # where w_j = exp(-contrib_j/beta + max_val/beta) / sum_exp
    
    weights = exp_terms / sum_exp
    pair_weights = weights[:len(pair_contrib)].reshape(-1)
    wall_weights = weights[len(pair_contrib):].reshape(N, 3)
    
    # Pairwise gradient contribution
    # For pair (i,j), contrib = |c_i - c_j|/2
    # d(contrib)/d(c_i) = (c_i - c_j) / (2 * |c_i - c_j|) = (c_i - c_j) / (2 * dist)
    # So d(softmin)/d(c_i) includes -w_ij * (c_i - c_j) / (2 * dist_ij)
    
    grad = np.zeros_like(c)
    
    # Process pairs
    i_idx, j_idx = np.triu_indices(N, k=1)
    for idx in range(len(i_idx)):
        i, j = i_idx[idx], j_idx[idx]
        w = pair_weights[idx]
        d = c[i] - c[j]
        dist = np.sqrt(np.sum(d * d))
        if dist > 1e-12:
            # d(softmin)/d(c_i) += -w * (c_i - c_j) / (2 * dist)
            # d(softmin)/d(c_j) += -w * (c_j - c_i) / (2 * dist) = w * (c_i - c_j) / (2 * dist)
            grad[i] += -w * d / (2.0 * dist)
            grad[j] += w * d / (2.0 * dist)
    
    # Wall gradient contribution
    # For point i, wall contrib = 0.5 - |c_ik| for each k
    # d(contrib)/d(c_ik) = -sign(c_ik)
    # So d(softmin)/d(c_ik) += -w_ik * (-sign(c_ik)) = w_ik * sign(c_ik)
    signs = np.sign(c)
    grad += wall_weights * signs
    
    # We want to maximize softmin, so objective = -softmin
    # Gradient of objective = -gradient of softmin
    return -softmin_val, -grad


def _epigraph_objective_and_grad(x, beta=200.0):
    """Epigraph formulation: minimize -t subject to t <= min(dist/2, wall_dist).
    
    Using smooth approximation: objective = -t + penalty for violations.
    x = [c_1, ..., c_N, t]
    """
    c = x[:N*3].reshape(N, 3)
    t = x[N*3]
    
    # Compute soft-min radius
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dists_sq, np.inf)
    dists = np.sqrt(dists_sq)
    wall_dists = 0.5 - np.abs(c)
    
    pair_contrib = (dists[~np.isinf(dists)] / 2.0).flatten()
    wall_contrib = wall_dists.flatten()
    all_contrib = np.concatenate([pair_contrib, wall_contrib])
    
    # Soft-min
    max_val = np.max(all_contrib)
    exp_terms = np.exp(-all_contrib / beta + max_val / beta)
    sum_exp = np.sum(exp_terms)
    softmin_val = max_val - beta * np.log(sum_exp)
    
    # Objective: minimize (t - softmin_val)^2 + (t - 0.136)^2 * 0.01
    # This encourages t to be close to the actual min distance
    obj = (t - softmin_val) ** 2 + 0.01 * (t - 0.136) ** 2
    
    # Gradient
    # d(obj)/dt = 2*(t - softmin_val) + 0.02*(t - 0.136)
    grad_t = 2.0 * (t - softmin_val) + 0.02 * (t - 0.136)
    
    # d(obj)/dc = -2*(t - softmin_val) * d(softmin_val)/dc
    # We need d(softmin_val)/dc
    weights = exp_terms / sum_exp
    pair_weights = weights[:len(pair_contrib)].reshape(-1)
    wall_weights = weights[len(pair_contrib):].reshape(N, 3)
    
    d_softmin_dc = np.zeros_like(c)
    i_idx, j_idx = np.triu_indices(N, k=1)
    for idx in range(len(i_idx)):
        i, j = i_idx[idx], j_idx[idx]
        w = pair_weights[idx]
        d = c[i] - c[j]
        dist = np.sqrt(np.sum(d * d))
        if dist > 1e-12:
            d_softmin_dc[i] += -w * d / (2.0 * dist)
            d_softmin_dc[j] += w * d / (2.0 * dist)
    
    signs = np.sign(c)
    d_softmin_dc += wall_weights * signs
    
    grad_c = -2.0 * (t - softmin_val) * d_softmin_dc
    
    grad = np.concatenate([grad_c.flatten(), [grad_t]])
    return obj, grad


def _repulsion_anneal(c0, n_steps=2000, start_temp=0.1, end_temp=0.001, time_limit=20.0):
    """Simulated annealing with repulsion energy."""
    c = c0.copy()
    start_time = time.time()
    
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    for step in range(n_steps):
        if time.time() - start_time > time_limit:
            break
        
        # Temperature schedule
        temp = start_temp * (end_temp / start_temp) ** (step / n_steps)
        
        # Propose a move: pick random point, perturb it
        i = np.random.randint(N)
        delta = np.random.normal(0, temp, 3)
        c_new = c.copy()
        c_new[i] += delta
        
        # Keep within bounds
        c_new[i] = np.clip(c_new[i], -0.499, 0.499)
        
        # Compute energy (negative radius)
        r_old = _compute_radius(c)
        r_new = _compute_radius(c_new)
        
        # Accept if better, or with probability exp(-(r_old - r_new)/temp)
        if r_new > r_old:
            c = c_new
            if r_new > best_r:
                best_r = r_new
                best_c = c.copy()
        else:
            delta_e = r_old - r_new
            if np.random.random() < np.exp(-delta_e / max(temp, 1e-10)):
                c = c_new
    
    return best_c, best_r


def _lloyd_step(c, n_iters=50, time_limit=10.0):
    """Lloyd's algorithm: move each point to the centroid of its Voronoi cell."""
    c = c.copy()
    start_time = time.time()
    
    for it in range(n_iters):
        if time.time() - start_time > time_limit:
            break
        
        try:
            # Compute Voronoi diagram
            vor = Voronoi(c)
            
            # For each point, find the centroid of its Voronoi cell
            for i in range(N):
                # Find the region for point i
                region_idx = vor.point_region[i]
                region = vor.regions[region_idx]
                
                if len(region) == 0 or -1 in region:
                    # Unbounded region, skip or handle carefully
                    continue
                
                # Get vertices of the region
                vertices = vor.vertices[region]
                
                # Compute centroid
                centroid = np.mean(vertices, axis=0)
                
                # Move point towards centroid
                c[i] = 0.5 * c[i] + 0.5 * centroid
            
            # Keep within bounds
            c = np.clip(c, -0.499, 0.499)
            
        except Exception:
            # Voronoi computation failed, break
            break
    
    return c


def _smooth_polish(c, time_limit=15.0):
    """Polish using L-BFGS-B with smooth objective."""
    x0 = np.concatenate([c.flatten(), [0.136]])
    
    # Bounds for c and t
    bounds = [(-0.5, 0.5)] * (N * 3) + [(0.0, 0.5)]
    
    start_time = time.time()
    
    def fun(x):
        if time.time() - start_time > time_limit:
            raise RuntimeError("Time limit exceeded")
        val, grad = _epigraph_objective_and_grad(x, beta=300.0)
        return val
    
    def jac(x):
        if time.time() - start_time > time_limit:
            raise RuntimeError("Time limit exceeded")
        val, grad = _epigraph_objective_and_grad(x, beta=300.0)
        return grad
    
    try:
        result = minimize(fun, x0, jac=jac, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
        c_opt = result.x[:N*3].reshape(N, 3)
        c_opt = np.clip(c_opt, -0.4999, 0.4999)
        return c_opt
    except Exception:
        return c


def _local_search(c, n_trials=500, time_limit=10.0):
    """Local search with small perturbations."""
    best_c = c.copy()
    best_r = _compute_radius(c)
    start_time = time.time()
    
    for trial in range(n_trials):
        if time.time() - start_time > time_limit:
            break
        
        # Random perturbation
        scale = 0.01 * (1.0 - trial / n_trials) + 0.001
        c_trial = c + np.random.normal(0, scale, c.shape)
        c_trial = np.clip(c_trial, -0.4999, 0.4999)
        
        r_trial = _compute_radius(c_trial)
        if r_trial > best_r:
            best_r = r_trial
            best_c = c_trial
    
    return best_c


def _seed_fcc():
    """FCC-like seed."""
    # FCC lattice points in a cube
    a = 1.0 / 4.0
    points = []
    for i in range(4):
        for j in range(4):
            for k in range(4):
                # FCC: corners + face centers
                points.append([i*a, j*a, k*a])
                points.append([(i+0.5)*a, (j+0.5)*a, k*a])
                points.append([(i+0.5)*a, j*a, (k+0.5)*a])
                points.append([i*a, (j+0.5)*a, (k+0.5)*a])
    
    points = np.array(points)
    # Center and scale to fit in [-0.5, 0.5]^3
    points = (points - 0.5) * 0.9  # Scale down slightly
    points = points[:N]
    
    # Add small perturbation
    np.random.seed(123)
    points += np.random.normal(0, 0.01, points.shape)
    points = np.clip(points, -0.49, 0.49)
    
    return points


def _seed_bcc():
    """BCC-like seed."""
    a = 1.0 / 4.0
    points = []
    for i in range(4):
        for j in range(4):
            for k in range(4):
                points.append([i*a, j*a, k*a])
                points.append([(i+0.5)*a, (j+0.5)*a, (k+0.5)*a])
    
    points = np.array(points)
    points = (points - 0.5) * 0.9
    points = points[:N]
    
    np.random.seed(456)
    points += np.random.normal(0, 0.01, points.shape)
    points = np.clip(points, -0.49, 0.49)
    
    return points


def _seed_grid():
    """Grid seed."""
    m = int(np.ceil(N ** (1.0 / 3.0)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    c = grid[:N].copy()
    
    np.random.seed(789)
    c += np.random.normal(0, 0.01, c.shape)
    c = np.clip(c, -0.49, 0.49)
    
    return c


def _seed_random():
    """Random seed."""
    np.random.seed(101)
    c = np.random.uniform(-0.45, 0.45, (N, 3))
    return c


def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    # Try multiple seeds and strategies
    seeds = [
        ("fcc", _seed_fcc),
        ("bcc", _seed_bcc),
        ("grid", _seed_grid),
        ("random", _seed_random),
    ]
    
    best_c = None
    best_r = 0.0
    
    for name, seed_func in seeds:
        elapsed = time.time() - start_time
        if elapsed > deadline:
            break
        
        # Get seed
        c0 = seed_func()
        r0 = _compute_radius(c0)
        
        if r0 > best_r:
            best_r = r0
            best_c = c0.copy()
        
        # Repulsion annealing
        elapsed = time.time() - start_time
        if elapsed < deadline - 40:
            c_anneal, r_anneal = _repulsion_anneal(c0, n_steps=1500, time_limit=15.0)
            if r_anneal > best_r:
                best_r = r_anneal
                best_c = c_anneal.copy()
        
        # Lloyd's algorithm
        elapsed = time.time() - start_time
        if elapsed < deadline - 30:
            c_lloyd = _lloyd_step(c_anneal, n_iters=30, time_limit=8.0)
            r_lloyd = _compute_radius(c_lloyd)
            if r_lloyd > best_r:
                best_r = r_lloyd
                best_c = c_lloyd.copy()
        
        # Smooth polish
        elapsed = time.time() - start_time
        if elapsed < deadline - 20:
            c_polish = _smooth_polish(c_lloyd, time_limit=12.0)
            r_polish = _compute_radius(c_polish)
            if r_polish > best_r:
                best_r = r_polish
                best_c = c_polish.copy()
        
        # Local search
        elapsed = time.time() - start_time
        if elapsed < deadline - 10:
            c_local = _local_search(c_polish, n_trials=300, time_limit=8.0)
            r_local = _compute_radius(c_local)
            if r_local > best_r:
                best_r = r_local
                best_c = c_local.copy()
    
    # Final verification
    if best_c is None or not np.all(np.isfinite(best_c)):
        # Fallback to simple grid
        m = int(np.ceil(N ** (1.0 / 3.0)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_c = grid[:N].copy()
    
    # Ensure all points are strictly inside the cube
    best_c = np.clip(best_c, -0.4999, 0.4999)
    
    return best_c
