"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

def _compute_radius(centers):
    """Compute the largest common radius for given centers."""
    if not np.all(np.isfinite(centers)):
        return 0.0
    if centers.shape != (N, 3):
        return 0.0
    
    # Distance from each center to nearest wall
    wall_dists = np.min(np.array([
        centers[:, 0] + 0.5,
        -centers[:, 0] + 0.5,
        centers[:, 1] + 0.5,
        -centers[:, 1] + 0.5,
        centers[:, 2] + 0.5,
        -centers[:, 2] + 0.5,
    ]), axis=1)
    
    if np.any(wall_dists < -1e-10):
        return 0.0
    
    min_wall = np.min(wall_dists)
    
    # Pairwise distances
    dists = cdist(centers, centers)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    r = 0.5 * min(min_pair, min_wall)
    return r if r > 0 else 0.0

def _soft_min(x, beta):
    """Compute soft-min using log-sum-exp trick."""
    # soft_min(x) ≈ -log(sum(exp(-beta*x)))/beta
    # For numerical stability, subtract max
    x_shifted = x - np.max(x)
    log_sum = np.log(np.sum(np.exp(-beta * x_shifted)))
    return -(np.max(x) - log_sum / beta)

def _objective_and_grad(x_flat, beta):
    """Objective: negative soft-min of all distances (pairwise + wall).
    Returns (value, gradient) for maximization via minimization of negative."""
    centers = x_flat.reshape(N, 3)
    
    # Wall distances
    wall_dists = np.array([
        centers[:, 0] + 0.5,
        -centers[:, 0] + 0.5,
        centers[:, 1] + 0.5,
        -centers[:, 1] + 0.5,
        centers[:, 2] + 0.5,
        -centers[:, 2] + 0.5,
    ])  # shape (6, N)
    
    # Pairwise distances
    dists = cdist(centers, centers)
    np.fill_diagonal(dists, np.inf)
    # Get upper triangle distances
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu]
    
    # Combine all distances
    all_dists = np.concatenate([wall_dists.ravel(), pair_dists])
    
    # Compute soft-min
    x_shifted = all_dists - np.max(all_dists)
    exp_vals = np.exp(-beta * x_shifted)
    log_sum = np.log(np.sum(exp_vals))
    soft_min_val = -(np.max(all_dists) - log_sum / beta)
    
    # Gradient of soft-min w.r.t. all_dists
    # d(soft_min)/d(all_dists[k]) = exp(-beta*(all_dists[k] - max)) / sum(exp(-beta*(all_dists - max)))
    weights = exp_vals / np.sum(exp_vals)
    
    grad_dists = np.zeros_like(all_dists)
    grad_dists[:] = weights
    
    # Now backpropagate to centers
    grad_centers = np.zeros((N, 3))
    
    # Wall distance gradients
    # wall_dists[0] = centers[:,0] + 0.5 -> grad w.r.t. centers[:,0] = +1
    # wall_dists[1] = -centers[:,0] + 0.5 -> grad w.r.t. centers[:,0] = -1
    # Similarly for y and z
    for axis in range(3):
        w_pos = grad_dists[axis * N:(axis+1)*N]
        w_neg = grad_dists[(axis+3)*N:(axis+4)*N]
        grad_centers[:, axis] += w_pos - w_neg
    
    # Pairwise distance gradients
    # d(|ci-cj|)/d(ci) = (ci-cj)/|ci-cj|
    # d(|ci-cj|)/d(cj) = (cj-ci)/|ci-cj|
    for idx in range(len(iu[0])):
        i, j = iu[0][idx], iu[1][idx]
        w = grad_dists[N*6 + idx]
        diff = centers[i] - centers[j]
        d = np.linalg.norm(diff)
        if d > 1e-12:
            grad_centers[i] += w * diff / d
            grad_centers[j] -= w * diff / d
    
    # Objective is negative soft-min (we want to maximize soft-min)
    value = -soft_min_val
    grad = -grad_centers.ravel()
    
    return value, grad

def _smooth_optimize(centers, beta_schedule=None, max_iters=200, tol=1e-10):
    """Optimize centers using smooth surrogate with adaptive beta."""
    if beta_schedule is None:
        beta_schedule = [10.0, 50.0, 100.0, 200.0, 500.0]
    
    x = centers.copy().ravel()
    
    for beta in beta_schedule:
        result = minimize(
            _objective_and_grad,
            x,
            args=(beta,),
            jac=True,
            method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (N * 3),
            options={'maxiter': max_iters, 'ftol': tol, 'gtol': tol, 'disp': False}
        )
        x = result.x
    
    return x.reshape(N, 3)

def _local_refine(centers):
    """Refine a packing by pushing spheres apart when they overlap."""
    centers = centers.copy()
    n = len(centers)
    
    for iteration in range(50):
        improved = False
        dists = cdist(centers, centers)
        np.fill_diagonal(dists, np.inf)
        
        for i in range(n):
            for j in range(i+1, n):
                d = dists[i, j]
                if d < 0.01:
                    diff = centers[i] - centers[j]
                    norm = np.linalg.norm(diff)
                    if norm > 1e-10:
                        push = 0.001 * diff / norm
                        centers[i] += push
                        centers[j] -= push
                        improved = True
        
        centers = np.clip(centers, -0.5, 0.5)
        
        if not improved:
            break
    
    return centers

def construct():
    """Construct centers for N equal spheres in unit cube."""
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s limit
    
    # Initial: grid placement
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    best_centers = grid[:N].copy()
    best_radius = _compute_radius(best_centers)
    
    rng = np.random.RandomState(42)
    
    # Phase 1: Random restarts with smooth optimization
    num_restarts = 15
    for restart in range(num_restarts):
        if time.time() > deadline - 30.0:
            break
        
        # Random initial positions
        centers = rng.uniform(-0.4, 0.4, size=(N, 3))
        
        # Smooth optimization with adaptive beta
        centers = _smooth_optimize(centers, beta_schedule=[10.0, 50.0, 100.0], max_iters=100)
        
        # Local refinement
        centers = _local_refine(centers)
        
        radius = _compute_radius(centers)
        if radius > best_radius:
            best_radius = radius
            best_centers = centers.copy()
    
    # Phase 2: High-precision polish on best solution
    if time.time() < deadline - 10.0:
        best_centers = _smooth_optimize(best_centers, beta_schedule=[100.0, 300.0, 500.0, 1000.0], max_iters=50)
        best_centers = _local_refine(best_centers)
        final_radius = _compute_radius(best_centers)
        if final_radius > best_radius:
            best_radius = final_radius
    
    # Final safety: ensure all centers are strictly inside
    best_centers = np.clip(best_centers, -0.499999, 0.499999)
    
    return best_centers
# EVOLVE-BLOCK-END
