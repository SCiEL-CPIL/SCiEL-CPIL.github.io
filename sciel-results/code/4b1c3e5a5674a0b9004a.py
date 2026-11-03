import numpy as np
from scipy.optimize import minimize
import time

N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _compute_radius(centers):
    n = centers.shape[0]
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists) / 2.0
    min_boundary_dist = np.min(0.5 - np.abs(centers))
    return min(min_pair_dist, min_boundary_dist)

def _soft_min_objective_and_grad(x, beta):
    centers = x.reshape(N, 3)
    
    # Pairwise terms
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    dists_safe = np.where(dists < 1e-10, 1e-10, dists)
    
    v_pair = dists_safe / 2.0
    mask = ~np.eye(N, dtype=bool)
    
    exp_pair = np.exp(-beta * v_pair)
    np.fill_diagonal(exp_pair, 0)
    
    S_pair = 0.5 * np.sum(exp_pair)
    
    W = exp_pair / (2.0 * dists_safe)
    np.fill_diagonal(W, 0)
    
    sum_W = np.sum(W, axis=1)
    W_c = W.dot(centers)
    
    G_pair = 0.5 * (centers * sum_W[:, np.newaxis] - W_c)
    
    # Boundary terms
    v_bound = 0.5 - np.abs(centers)
    exp_bound = np.exp(-beta * v_bound)
    S_bound = np.sum(exp_bound)
    
    signs = np.sign(centers)
    G_bound = -exp_bound * signs
    
    S_total = S_pair + S_bound
    G_total = G_pair + G_bound
    
    grad = G_total / S_total
    obj = (1.0 / beta) * np.log(S_total)
    
    return obj, grad.ravel()

def _local_polish(centers, max_iter=100, tol=1e-10):
    """Subgradient ascent on the true nonsmooth radius."""
    best_r = _compute_radius(centers)
    best_c = centers.copy()
    lr = 0.01
    
    for it in range(max_iter):
        # Compute subgradient of the radius
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        # Find the closest pair
        min_pair_idx = np.unravel_index(np.argmin(dists), dists.shape)
        min_pair_dist = dists[min_pair_idx] / 2.0
        
        # Find the closest boundary
        bound_dists = 0.5 - np.abs(centers)
        min_bound_idx = np.unravel_index(np.argmin(bound_dists), bound_dists.shape)
        min_bound_dist = bound_dists[min_bound_idx]
        
        # Determine which constraint is active
        if min_pair_dist < min_bound_dist:
            # Pair constraint active
            i, j = min_pair_idx
            if i > j:
                i, j = j, i
            direction = centers[i] - centers[j]
            norm = np.linalg.norm(direction)
            if norm > 1e-10:
                direction /= norm
            else:
                direction = np.random.randn(3)
                direction /= np.linalg.norm(direction)
            
            grad = np.zeros_like(centers)
            grad[i] += direction
            grad[j] -= direction
        else:
            # Boundary constraint active
            i, k = min_bound_idx
            direction = np.zeros(3)
            direction[k] = np.sign(centers[i, k])
            if centers[i, k] == 0:
                direction[k] = 1.0
            
            grad = np.zeros_like(centers)
            grad[i, k] = -direction[k]
        
        # Move in the direction of the subgradient
        new_centers = centers + lr * grad
        new_centers = np.clip(new_centers, -0.5, 0.5)
        
        new_r = _compute_radius(new_centers)
        
        if new_r > best_r + tol:
            centers = new_centers
            best_r = new_r
            best_c = centers.copy()
            lr *= 0.95  # Slightly reduce learning rate
        else:
            lr *= 0.8  # Reduce learning rate more aggressively
            if lr < 1e-8:
                break
    
    return best_c, best_r

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    best_radius = 0.0
    best_centers = None
    
    # Initial best from seed (grid)
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    initial_centers = grid[:N]
    best_radius = _compute_radius(initial_centers)
    best_centers = initial_centers.copy()
    
    # Optimization loop
    num_restarts = 20
    betas = [5, 10, 20, 50, 100, 200]
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Initialize
        if r == 0:
            x0 = best_centers.ravel()
        else:
            # Random initialization within the cube
            if r % 2 == 0:
                x0 = best_centers.ravel() + np.random.normal(0, 0.05, N*3)
                x0 = np.clip(x0, -0.5, 0.5)
            else:
                x0 = np.random.uniform(-0.4, 0.4, N*3)
                
        # Annealing over betas
        current_x = x0
        for beta in betas:
            if time.time() > deadline:
                break
                
            try:
                res = minimize(
                    _soft_min_objective_and_grad,
                    current_x,
                    args=(beta,),
                    method='L-BFGS-B',
                    jac=True,
                    bounds=[(-0.5, 0.5)] * (N * 3),
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                current_x = res.x
            except Exception:
                pass
                
        # Evaluate true radius
        centers = current_x.reshape(N, 3)
        centers = np.clip(centers, -0.5, 0.5)
        
        r_curr = _compute_radius(centers)
        
        if r_curr > best_radius:
            best_radius = r_curr
            best_centers = centers.copy()
            
    return best_centers
# EVOLVE-BLOCK-END
