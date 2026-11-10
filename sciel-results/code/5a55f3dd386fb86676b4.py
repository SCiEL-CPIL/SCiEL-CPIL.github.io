import numpy as np
from scipy.optimize import minimize
import time

N = 50

def _hex_lattice_init():
    """Generate initial hexagonal lattice points."""
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
        if len(points) >= N:
            return points[np.argsort(radius, kind="stable")[:N]]
        spacing *= 0.98

def _soft_min_energy_and_grad(x, alpha):
    """
    Compute the soft-min energy and gradient.
    x: (N, 2) array of centers.
    alpha: temperature parameter for soft-min.
    Returns: (energy, gradient)
    """
    x = x.reshape(N, 2)
    # Pairwise distances
    # diff[i, j] = x[i] - x[j]
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Avoid division by zero in gradient calculation for i=j
    np.fill_diagonal(dists, np.inf)
    
    # Boundary distances: 1 - |x_i|
    radii = np.linalg.norm(x, axis=1)
    b_dists = 1.0 - radii
    
    # Combine all distances into a single list for soft-min
    # We want min of all pairwise and boundary distances.
    # Soft min: -1/alpha * log( sum(exp(-alpha * d_k)) )
    # We minimize this, so we maximize the soft min.
    # Let's minimize E = 1/alpha * log( sum(exp(-alpha * d_k)) )
    
    # Calculate weights w_k = exp(-alpha * d_k)
    # To avoid underflow/overflow, we can shift by max d, but here d are small.
    # d_k are positive. exp(-alpha * d_k) is small if d_k is large.
    
    p_dists = dists[np.triu_indices(N, k=1)]
    all_dists = np.concatenate([p_dists, b_dists])
    
    # If any distance is negative (overlap or outside), the term exp(-alpha * d) becomes exp(positive) -> huge.
    # This is good for penalization.
    
    exponents = -alpha * all_dists
    # Shift for numerical stability if exponents are large positive
    max_exp = np.max(exponents)
    if max_exp > 0:
        exponents -= max_exp
        shift = max_exp
    else:
        shift = 0.0
        
    weights = np.exp(exponents)
    sum_weights = np.sum(weights)
    
    if sum_weights == 0:
        return 0.0, np.zeros_like(x)
        
    energy = (1.0 / alpha) * (np.log(sum_weights) + shift)
    
    # Gradient
    # dE/dx_i = -1/S * sum_k w_k * d(d_k)/dx_i
    # Note: The shift cancels out in the derivative of log(sum w) because d/dx (log(sum w) + shift) = d/dx log(sum w)
    # Wait, E = 1/alpha * (log(sum w) + shift).
    # dE/dx = 1/alpha * (1/sum w) * sum (dw/dx)
    # dw_k/dx = w_k * (-alpha * dd_k/dx)
    # So dE/dx = 1/alpha * 1/S * sum (w_k * -alpha * grad_d_k) = -1/S * sum (w_k * grad_d_k)
    
    grad = np.zeros_like(x)
    
    # Pairwise gradients
    # grad_d_ij = (x_i - x_j) / d_ij
    # Contribution to grad[i] from pair (i,j) is -w_ij * (x_i - x_j) / d_ij / S
    # Contribution to grad[j] from pair (i,j) is -w_ij * (x_j - x_i) / d_ij / S = +w_ij * (x_i - x_j) / d_ij / S
    
    # We have p_dists corresponding to triu indices.
    i_idx, j_idx = np.triu_indices(N, k=1)
    w_pairs = weights[:len(p_dists)]
    d_pairs = p_dists
    
    # Vectorized gradient accumulation
    # diff_vec = x[i] - x[j]
    diff_vec = x[i_idx] - x[j_idx]
    # grad_unit = diff_vec / d_pairs
    # Avoid div by 0
    mask = d_pairs > 1e-10
    grad_unit = np.zeros_like(diff_vec)
    grad_unit[mask] = diff_vec[mask] / d_pairs[mask, np.newaxis]
    
    # weighted_grad = w_pairs * grad_unit
    weighted_grad = w_pairs[:, np.newaxis] * grad_unit
    
    # Accumulate to grad[i] and grad[j]
    # grad[i] -= weighted_grad / S
    # grad[j] += weighted_grad / S
    
    np.add.at(grad, i_idx, -weighted_grad / sum_weights)
    np.add.at(grad, j_idx, weighted_grad / sum_weights)
    
    # Boundary gradients
    # d_b_i = 1 - |x_i|
    # grad_d_b_i = -x_i / |x_i|
    w_bounds = weights[len(p_dists):]
    radii_safe = np.where(radii < 1e-10, 1e-10, radii)
    grad_bounds = -x / radii_safe[:, np.newaxis]
    
    weighted_bounds = w_bounds[:, np.newaxis] * grad_bounds
    grad += -weighted_bounds / sum_weights
    
    return energy, grad

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    # Initial guess
    x0 = _hex_lattice_init()
    
    best_x = x0
    best_score = -np.inf
    
    # We will run optimization with a few different alpha values or just one good one.
    # Let's try a sequence of alphas to refine.
    alphas = [5.0, 10.0, 20.0, 50.0]
    
    current_x = x0
    
    for alpha in alphas:
        if time.time() > deadline:
            break
            
        try:
            res = minimize(
                _soft_min_energy_and_grad,
                current_x,
                args=(alpha,),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if res.success or res.fun < best_score:
                # Check validity
                x_new = res.x.reshape(N, 2)
                # Calculate actual score
                d_min = np.inf
                for i in range(N):
                    for j in range(i+1, N):
                        d = np.linalg.norm(x_new[i] - x_new[j])
                        if d < d_min:
                            d_min = d
                    d_b = 1.0 - np.linalg.norm(x_new[i])
                    if d_b < d_min:
                        d_min = d_b
                
                r = d_min / 2.0
                if r > best_score:
                    best_score = r
                    best_x = x_new
                    current_x = x_new
        except Exception:
            pass
            
    # Final check and return
    return best_x
