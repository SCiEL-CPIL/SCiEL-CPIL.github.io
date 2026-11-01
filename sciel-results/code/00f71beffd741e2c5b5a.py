import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import time

N = 100

def _compute_radius(centers):
    """
    Compute the largest common radius for the given centers.
    Centers are in [-0.5, 0.5]^3.
    """
    # Distance to walls
    dists_to_walls = 0.5 - np.abs(centers)
    min_wall_dist = np.min(dists_to_walls)
    
    # Pairwise distances
    dists = cdist(centers, centers, metric='euclidean')
    
    # Zero out diagonal
    np.fill_diagonal(dists, np.inf)
    
    min_pair_dist = np.min(dists)
    
    return min(min_wall_dist, min_pair_dist / 2.0)

def _soft_min_objective(centers_flat, beta, iu):
    """
    Objective function for L-BFGS-B.
    Minimizes the soft-min approximation of the radius.
    r(p) approx -1/beta * ln(sum(exp(-beta * x_k)))
    where x_k are the constraints (distances).
    We want to maximize r, so we minimize -r.
    """
    centers = centers_flat.reshape(N, 3)
    
    # Wall constraints
    wall_dists = 0.5 - np.abs(centers)
    # Pairwise constraints (dist/2)
    dists = cdist(centers, centers, metric='euclidean')
    np.fill_diagonal(dists, np.inf)
    pair_dists = dists / 2.0
    
    # Combine all constraints
    # We can do this in a vectorized way or just concatenate
    # For numerical stability, use log-sum-exp
    # x = [wall_dists, pair_dists[upper_triangle]]
    
    # To avoid huge arrays, we can compute log-sum-exp directly
    # But for simplicity and speed with N=100, concatenation is fine
    # Number of pairs: 100*99/2 = 4950
    # Number of wall dists: 300
    # Total ~ 5250
    
    # Get upper triangle indices
    iu_idx = np.triu_indices(N, k=1)
    pair_vals = pair_dists[iu_idx]
    
    all_vals = np.concatenate([wall_dists.flatten(), pair_vals])
    
    # Soft min: -1/beta * log(sum(exp(-beta * x)))
    # Numerically stable version:
    # Let m = min(all_vals)
    # sum(exp(-beta * (x - m))) * exp(-beta * m)
    # log(sum(exp(-beta * x))) = -beta * m + log(sum(exp(-beta * (x - m))))
    
    m = np.min(all_vals)
    shifted = -beta * (all_vals - m)
    # Clip to avoid overflow in exp
    shifted = np.clip(shifted, -700, 700)
    log_sum_exp = np.log(np.sum(np.exp(shifted)))
    
    soft_min_val = -1.0 / beta * (-beta * m + log_sum_exp)
    
    # We want to maximize soft_min_val, so minimize -soft_min_val
    return -soft_min_val

def _soft_min_grad(centers_flat, beta, iu):
    """
    Gradient of the soft-min objective.
    """
    centers = centers_flat.reshape(N, 3)
    
    wall_dists = 0.5 - np.abs(centers)
    dists = cdist(centers, centers, metric='euclidean')
    np.fill_diagonal(dists, np.inf)
    pair_dists = dists / 2.0
    
    iu_idx = np.triu_indices(N, k=1)
    pair_vals = pair_dists[iu_idx]
    
    all_vals = np.concatenate([wall_dists.flatten(), pair_vals])
    
    m = np.min(all_vals)
    shifted = -beta * (all_vals - m)
    shifted = np.clip(shifted, -700, 700)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    
    # Gradient of soft_min w.r.t x_k:
    # d/dx_k [-1/beta * log(sum(exp(-beta * x_j)))]
    # = -1/beta * 1/sum(exp(-beta*x_j)) * (-beta * exp(-beta*x_k))
    # = exp(-beta*x_k) / sum(exp(-beta*x_j))
    # So d(soft_min)/dx_k = exp(-beta*x_k) / sum_exp
    # We are minimizing -soft_min, so grad = -d(soft_min)/dx_k
    # = -exp(-beta*x_k) / sum_exp
    
    weights = -exp_shifted / sum_exp
    
    grad_centers = np.zeros_like(centers)
    
    # Wall gradients
    # x_k = 0.5 - |c_ik|
    # d(x_k)/d(c_ik) = -sign(c_ik)
    wall_weights = weights[:N*3]
    for i in range(N):
        for k in range(3):
            idx = i*3 + k
            grad_centers[i, k] += wall_weights[idx] * (-np.sign(centers[i, k]))
    
    # Pair gradients
    # x_k = dist(i,j)/2
    # d(x_k)/d(c_i) = (c_i - c_j) / (2 * dist(i,j))
    # d(x_k)/d(c_j) = (c_j - c_i) / (2 * dist(i,j)) = -(c_i - c_j) / (2 * dist(i,j))
    
    pair_weights = weights[N*3:]
    i_idx, j_idx = iu_idx
    for idx in range(len(pair_weights)):
        i = i_idx[idx]
        j = j_idx[idx]
        d = centers[i] - centers[j]
        dist = np.linalg.norm(d)
        if dist < 1e-12:
            continue
        # Gradient contribution to c_i
        # w * (d / (2*dist))
        grad_centers[i] += pair_weights[idx] * d / (2.0 * dist)
        # Gradient contribution to c_j
        # w * (-d / (2*dist))
        grad_centers[j] += pair_weights[idx] * (-d / (2.0 * dist))
        
    return grad_centers.flatten()

def _relax(centers, beta, max_iter=100, ftol=1e-12, deadline=None):
    """
    Optimize centers to maximize soft-min radius.
    """
    x0 = centers.flatten()
    
    res = minimize(
        _soft_min_objective,
        x0,
        args=(beta, None),
        jac=lambda x, b, iu: _soft_min_grad(x, b, iu),
        method='L-BFGS-B',
        bounds=[(-0.5, 0.5)] * (N * 3),
        options={'maxiter': max_iter, 'ftol': ftol, 'disp': False}
    )
    
    if res.success or res.fun < 1e-6:
        return res.x.reshape(N, 3)
    return x0.reshape(N, 3)

def construct():
    start_time = time.time()
    timeout = 140.0 # Safety margin for 150s limit
    
    # 1. Initialization
    # Grid 5x5x4
    x = np.linspace(-0.4, 0.4, 5)
    y = np.linspace(-0.4, 0.4, 5)
    z = np.linspace(-0.4, 0.4, 4)
    
    centers = np.array([(xi, yi, zi) for zi in z for yi in y for xi in x])
    # Add jitter
    centers += np.random.uniform(-0.01, 0.01, size=(N, 3))
    
    # Simple force-directed relaxation to break symmetry
    for _ in range(50):
        if time.time() - start_time > timeout: break
        forces = np.zeros_like(centers)
        for i in range(N):
            for j in range(N):
                if i == j: continue
                d = centers[i] - centers[j]
                dist = np.linalg.norm(d)
                if dist < 0.15: # Repel if close
                    forces[i] += d / (dist**3 + 1e-6)
            # Wall repulsion
            for k in range(3):
                if centers[i, k] > 0.4: forces[i, k] -= 1.0
                elif centers[i, k] < -0.4: forces[i, k] += 1.0
        centers += 0.001 * forces
        # Project back to box
        centers = np.clip(centers, -0.5, 0.5)
    
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    # 2. Optimization with increasing beta (continuation method)
    # Start with low beta for global search, increase for local precision
    
    betas = [10.0, 50.0, 100.0, 200.0, 500.0, 1000.0]
    
    for beta in betas:
        if time.time() - start_time > timeout - 10:
            break
            
        # Relax with current beta
        optimized_centers = _relax(best_centers, beta, max_iter=100, deadline=timeout)
        
        # Compute actual radius
        r_actual = _compute_radius(optimized_centers)
        
        if r_actual > best_radius:
            best_radius = r_actual
            best_centers = optimized_centers
        else:
            # If no improvement, keep previous best
            pass
            
    # 3. Final Polish
    # Use a very high beta for final precision
    if time.time() - start_time < timeout - 5:
        final_centers = _relax(best_centers, 2000.0, max_iter=200, ftol=1e-14, deadline=timeout)
        r_final = _compute_radius(final_centers)
        if r_final > best_radius:
            best_radius = r_final
            best_centers = final_centers
    
    # Ensure centers are strictly inside
    best_centers = np.clip(best_centers, -0.4999, 0.4999)
    
    return best_centers
