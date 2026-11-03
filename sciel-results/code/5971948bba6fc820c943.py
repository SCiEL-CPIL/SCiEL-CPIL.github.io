"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _compute_radius(centers):
    """Compute the exact maximum common radius for a given set of centers."""
    n = len(centers)
    if n < 2:
        return np.min(0.5 - np.abs(centers))
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    wall_dists = 0.5 - np.abs(centers)
    min_wall_dist = np.min(wall_dists)
    
    return min(min_pair_dist / 2.0, min_wall_dist)


def _soft_min_objective_and_grad(centers, target_r, beta):
    """
    Objective: maximize the minimum of (pair distances / 2, wall distances).
    We use a soft-min formulation:
    soft_min = -(1/beta) * log( sum_{i<j} exp(-beta * d_ij/2) + sum_i exp(-beta * w_i) )
    
    We want to maximize soft_min, equivalently minimize -soft_min.
    
    The gradient with respect to centers is computed analytically.
    """
    n = len(centers)
    if n < 2:
        wall_dists = 0.5 - np.abs(centers)
        # soft_min = -(1/beta) * log(sum(exp(-beta * w_i)))
        log_terms = -beta * wall_dists
        log_terms = np.clip(log_terms, -500, 500)
        log_sum = np.logaddexp.reduce(log_terms)
        obj = -(1.0/beta) * log_sum
        # grad: d(obj)/d(centers_k) = (1/beta) * (sum_i exp(-beta*w_i) * (-beta) * dw_i/dx_k) / sum_i exp(-beta*w_i)
        # dw_i/dx_k = -sign(x_ik) for the k-th coordinate of sphere i
        weights = np.exp(log_terms - log_sum)  # normalized weights
        grad = np.zeros_like(centers)
        for i in range(n):
            for k in range(3):
                sign = np.sign(centers[i, k])
                if sign == 0:
                    sign = 1.0
                grad[i, k] = -weights[i] * sign
        return obj, grad
    
    # Pairwise
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Wall distances
    wall_dists = 0.5 - np.abs(centers)
    
    # We want to maximize min(d_ij/2, w_i)
    # soft_min = -(1/beta) * log( sum_{i<j} exp(-beta * d_ij/2) + sum_i exp(-beta * w_i) )
    
    # Compute log terms
    iu = np.triu_indices(n, 1)
    pair_dists_half = dists[iu] / 2.0
    
    # Clip to avoid overflow
    log_pair_terms = -beta * pair_dists_half
    log_pair_terms = np.clip(log_pair_terms, -500, 500)
    log_wall_terms = -beta * wall_dists
    log_wall_terms = np.clip(log_wall_terms, -500, 500)
    
    # Combine: log(sum(exp(log_pair_terms)) + sum(exp(log_wall_terms)))
    # Use log-sum-exp trick
    max_log = max(np.max(log_pair_terms), np.max(log_wall_terms))
    sum_pair = np.sum(np.exp(log_pair_terms - max_log))
    sum_wall = np.sum(np.exp(log_wall_terms - max_log))
    log_sum = max_log + np.log(sum_pair + sum_wall)
    
    obj = -(1.0/beta) * log_sum
    
    # Gradient
    # d(obj)/d(centers) = (1/beta) * [ sum_{i<j} exp(-beta*d_ij/2) * (-beta/2) * d(d_ij)/d(centers) 
    #                                 + sum_i exp(-beta*w_i) * (-beta) * d(w_i)/d(centers) ] / sum_total
    
    # Normalized weights
    total = sum_pair + sum_wall
    w_pair = np.exp(log_pair_terms - max_log) / total
    w_wall = np.exp(log_wall_terms - max_log) / total
    
    grad = np.zeros_like(centers)
    
    # Pair contributions
    # For each pair (i,j), d(d_ij)/d(c_i) = -(c_i - c_j)/d_ij = (c_j - c_i)/d_ij
    # d(d_ij/2)/d(c_i) = (c_j - c_i)/(2*d_ij)
    # Contribution to grad[i] from pair (i,j): w_pair[k] * (-1/2) * (c_j - c_i)/d_ij
    # Wait, let me redo this carefully.
    
    # obj = -(1/beta) * log(S) where S = sum_{i<j} exp(-beta*d_ij/2) + sum_i exp(-beta*w_i)
    # d(obj)/d(c_i) = -(1/beta) * (1/S) * dS/d(c_i)
    # dS/d(c_i) = sum_{j != i} exp(-beta*d_ij/2) * (-beta/2) * d(d_ij)/d(c_i) 
    #            + exp(-beta*w_i) * (-beta) * d(w_i)/d(c_i)
    # d(d_ij)/d(c_i) = (c_i - c_j)/d_ij
    # d(w_i)/d(c_i) = -sign(c_i)
    
    # So: d(obj)/d(c_i) = -(1/beta) * (1/S) * [ sum_{j != i} exp(-beta*d_ij/2) * (-beta/2) * (c_i - c_j)/d_ij 
    #                                              + exp(-beta*w_i) * (-beta) * (-sign(c_i)) ]
    #               = (1/(2*S)) * sum_{j != i} exp(-beta*d_ij/2) * (c_i - c_j)/d_ij
    #                 + (1/S) * exp(-beta*w_i) * sign(c_i)
    
    # Using normalized weights: exp(-beta*d_ij/2)/S = w_pair[k] for the k-th pair
    # and exp(-beta*w_i)/S = w_wall[i]
    
    # Pair part: for each pair (i,j), add w_pair[k] * (c_i - c_j)/(2*d_ij) to grad[i]
    #            and w_pair[k] * (c_j - c_i)/(2*d_ij) to grad[j]
    
    for idx, (i, j) in enumerate(zip(iu[0], iu[1])):
        d = dists[i, j]
        if d < 1e-10:
            continue
        w = w_pair[idx]
        vec = (centers[i] - centers[j]) / d
        grad[i] += w * vec / 2.0
        grad[j] -= w * vec / 2.0
    
    # Wall part: for each sphere i, add w_wall[i] * sign(c_i) to grad[i]
    signs = np.sign(centers)
    signs[signs == 0] = 1.0
    grad += w_wall[:, np.newaxis] * signs
    
    return obj, grad


def _soft_min_optimize(centers_init, target_r, beta, max_iter, deadline):
    """Optimize using L-BFGS-B with analytic gradients."""
    def objective_and_grad(x):
        c = x.reshape(-1, 3)
        obj, grad = _soft_min_objective_and_grad(c, target_r, beta)
        return obj, grad.flatten()
    
    bounds = [(-0.4999, 0.4999)] * (3 * N)
    
    try:
        res = minimize(
            objective_and_grad,
            centers_init.flatten(),
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-6, 'disp': False,
                     'maxls': 50}
        )
        opt_centers = res.x.reshape(-1, 3)
    except Exception:
        opt_centers = centers_init
    
    if time.time() > deadline:
        return centers_init, _compute_radius(centers_init)
    
    r = _compute_radius(opt_centers)
    return opt_centers, r


def _radius_increase_search(centers_init, start_radius, deadline):
    """
    Iteratively increase the target radius and optimize.
    """
    best_centers = centers_init.copy()
    best_radius = start_radius
    
    target_r = best_radius * 1.01
    beta = 50.0
    max_iter = 300
    
    while time.time() < deadline - 5.0:
        remaining = deadline - time.time() - 3.0
        if remaining < 2.0:
            break
        
        centers, r = _soft_min_optimize(
            best_centers, target_r, beta, max_iter, 
            time.time() + remaining
        )
        
        if r > best_radius:
            best_radius = r
            best_centers = centers
            target_r = best_radius * 1.005
            beta = min(beta * 1.1, 200.0)
        else:
            # Try perturbation
            rng = np.random.RandomState(int(time.time()*1000) % 100000)
            perturbed = best_centers + rng.randn(N, 3) * 0.005
            perturbed = np.clip(perturbed, -0.4999, 0.4999)
            
            centers_p, r_p = _soft_min_optimize(
                perturbed, target_r, beta, max_iter,
                time.time() + remaining
            )
            
            if r_p > best_radius:
                best_radius = r_p
                best_centers = centers_p
                target_r = best_radius * 1.005
            else:
                if time.time() > deadline - 10.0:
                    break
                target_r = best_radius * 1.01
                beta = max(beta * 0.8, 10.0)
    
    return best_centers, best_radius


def _polish(centers, deadline):
    """Final high-precision polish pass."""
    if time.time() > deadline - 2.0:
        return centers, _compute_radius(centers)
    
    current_r = _compute_radius(centers)
    target_r = current_r * 1.001
    
    centers, r = _soft_min_optimize(
        centers, target_r, 200.0, 500,
        deadline - 1.0
    )
    
    if r > current_r:
        return centers, r
    return centers, current_r


def _grid_init(n):
    """Generate a grid initial configuration."""
    m = int(np.ceil(n ** (1.0/3.0)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:n]


def _random_init(n, seed=None):
    """Generate a random initial configuration."""
    rng = np.random.RandomState(seed)
    return rng.uniform(-0.45, 0.45, size=(n, 3))


def construct():
    """Construct the optimal packing of N spheres in the unit cube."""
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Initial setup: try multiple starts
    best_centers = _grid_init(N)
    best_radius = _compute_radius(best_centers)
    
    for seed in range(5):
        if time.time() > deadline - 30.0:
            break
        c = _random_init(N, seed)
        r = _compute_radius(c)
        if r > best_radius:
            best_radius = r
            best_centers = c
    
    # Main optimization loop
    best_centers, best_radius = _radius_increase_search(best_centers, best_radius, deadline)
    
    # Final polish
    if time.time() < deadline - 3.0:
        best_centers, best_radius = _polish(best_centers, deadline)
    
    # Ensure validity
    best_centers = np.clip(best_centers, -0.499999, 0.499999)
    
    return best_centers
# EVOLVE-BLOCK-END
