import numpy as np
from scipy.optimize import minimize
import time

N = 50

def _calc_score(x):
    """Calculate the packing score (max common radius) for centers x."""
    x = np.asarray(x).reshape(N, 2)
    # Pairwise distances
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    # Boundary distances
    radii = np.sqrt(np.sum(x ** 2, axis=1))
    min_bound = np.min(1.0 - radii)
    return min(min_pair, min_bound) / 2.0

def _hex_lattice_init():
    """Generate initial hexagonal lattice points, sorted by radius for better centering."""
    spacing = 0.22
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
        if len(points) >= N:
            # Sort by radius to get a more centered packing
            idx = np.argsort(radius, kind="stable")[:N]
            return points[idx]
        spacing *= 0.97

def _soft_min_energy_and_grad(x, alpha):
    """
    Compute the soft-min energy and gradient.
    We want to maximize the minimum distance, so we minimize the soft-min.
    x: (N, 2) array of centers.
    alpha: temperature parameter for soft-min.
    Returns: (energy, gradient)
    """
    x = x.reshape(N, 2)
    
    # Pairwise distances (upper triangle)
    i_idx, j_idx = np.triu_indices(N, k=1)
    diff_vec = x[i_idx] - x[j_idx]
    d_pairs = np.sqrt(np.sum(diff_vec ** 2, axis=1))
    
    # Boundary distances
    radii = np.sqrt(np.sum(x ** 2, axis=1))
    b_dists = 1.0 - radii
    
    # Combine all distances
    all_dists = np.concatenate([d_pairs, b_dists])
    n_pairs = len(d_pairs)
    n_bounds = N
    
    # Soft-min: E = (1/alpha) * log(sum(exp(-alpha * d_k)))
    # This is approximately min(d_k) when alpha is large.
    exponents = -alpha * all_dists
    max_exp = np.max(exponents)
    exponents_shifted = exponents - max_exp
    weights = np.exp(exponents_shifted)
    sum_weights = np.sum(weights)
    
    if sum_weights <= 0 or not np.isfinite(sum_weights):
        return 0.0, np.zeros_like(x)
    
    energy = (1.0 / alpha) * (np.log(sum_weights) + max_exp)
    
    # Gradient computation
    grad = np.zeros_like(x)
    
    # Pairwise part
    w_pairs = weights[:n_pairs]
    mask = d_pairs > 1e-12
    grad_unit = np.zeros_like(diff_vec)
    grad_unit[mask] = diff_vec[mask] / d_pairs[mask, np.newaxis]
    
    weighted_grad = w_pairs[:, np.newaxis] * grad_unit
    inv_sum = 1.0 / sum_weights
    
    np.add.at(grad, i_idx, -weighted_grad * inv_sum)
    np.add.at(grad, j_idx, weighted_grad * inv_sum)
    
    # Boundary part
    w_bounds = weights[n_pairs:n_pairs + n_bounds]
    radii_safe = np.where(radii < 1e-12, 1e-12, radii)
    grad_bounds = -x / radii_safe[:, np.newaxis]
    
    grad -= w_bounds[:, np.newaxis] * grad_bounds * inv_sum
    
    return energy, grad

def construct():
    start_time = time.time()
    deadline = 130.0  # Safety margin (evaluator kills at 150s)
    
    # Initial guess
    x0 = _hex_lattice_init()
    
    best_x = x0.copy()
    best_score = _calc_score(best_x)
    
    # Progressive alpha schedule: start low for broad exploration, increase for precision
    alphas = [3.0, 5.0, 10.0, 20.0, 40.0, 80.0]
    
    current_x = x0.copy()
    
    for alpha in alphas:
        if time.time() > deadline - 10.0:  # Leave time for final polish
            break
            
        try:
            res = minimize(
                _soft_min_energy_and_grad,
                current_x,
                args=(alpha,),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
            )
            x_new = res.x.reshape(N, 2)
            
            # Validate: check all finite, inside container
            if not np.all(np.isfinite(x_new)):
                continue
            if np.any(np.linalg.norm(x_new, axis=1) >= 1.0):
                continue
                
            score = _calc_score(x_new)
            if score > best_score:
                best_score = score
                best_x = x_new.copy()
                current_x = x_new.copy()
        except Exception:
            pass
    
    # Final high-precision polish with very high alpha
    remaining = deadline - time.time()
    if remaining > 5.0:
        try:
            res = minimize(
                _soft_min_energy_and_grad,
                current_x,
                args=(200.0,),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 30}
            )
            x_new = res.x.reshape(N, 2)
            if np.all(np.isfinite(x_new)) and np.all(np.linalg.norm(x_new, axis=1) < 1.0):
                score = _calc_score(x_new)
                if score > best_score:
                    best_score = score
                    best_x = x_new.copy()
        except Exception:
            pass
    
    # Return the best valid solution
    return best_x
