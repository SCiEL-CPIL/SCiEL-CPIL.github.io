"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14
D = 3

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

def get_score(pts):
    """Calculate the squared ratio of max to min pairwise distance."""
    dists = pdist(pts)
    if np.any(dists < 1e-10):
        return 1e10
    min_d = np.min(dists)
    max_d = np.max(dists)
    if min_d == 0:
        return 1e10
    return (max_d / min_d) ** 2

def energy_and_grad(pts_flat):
    """
    Compute the repulsive-attraction energy and its gradient.
    Energy = sum(1/d_ij^2) + alpha * sum(d_ij^2)
    This encourages equalizing all pairwise distances.
    """
    pts = pts_flat.reshape(N, D)
    # Compute pairwise distances and differences
    # diff[i, j] = pts[i] - pts[j]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=-1)
    # Avoid division by zero
    dists_sq = np.maximum(dists_sq, 1e-12)
    
    # Mask diagonal (self-distance)
    mask = ~np.eye(N, dtype=bool)
    
    # Energy terms
    # E_rep = sum(1/d^2) over i<j
    # Note: sum over i!=j is 2 * sum over i<j
    E_rep = np.sum(1.0 / dists_sq[mask]) / 2.0
    
    # E_attr = sum(d^2) over i<j
    E_attr = np.sum(dists_sq[mask]) / 2.0
    
    alpha = 1.0
    E_total = E_rep + alpha * E_attr
    
    # Gradient
    # Grad_i = sum_{j != i} ( -2 * (p_i - p_j) / d_ij^4 + 2 * alpha * (p_i - p_j) )
    # dists_sq is (N, N)
    # diff is (N, N, D)
    
    # Term 1: -2 * diff / dists_sq^2
    # dists_sq^2 is (N, N)
    inv_d4 = 1.0 / (dists_sq ** 2)
    # Multiply diff (N, N, D) by inv_d4 (N, N, 1)
    grad_rep = -2.0 * diff * inv_d4[:, :, np.newaxis]
    
    # Sum over j (axis 1)
    # grad_rep_sum[i] = sum_j grad_rep[i, j]
    grad_rep_sum = np.sum(grad_rep, axis=1)
    
    # Term 2: 2 * alpha * diff
    grad_attr = 2.0 * alpha * diff
    grad_attr_sum = np.sum(grad_attr, axis=1)
    
    grad_total = grad_rep_sum + grad_attr_sum
    
    return E_total, grad_total.flatten()

def construct():
    rng = np.random.default_rng(42)
    
    best_pts = None
    best_score = float('inf')
    
    # Run from multiple random initializations
    # We can also try a centered random init
    for k in range(20):
        if k == 0:
            # Start with a random sphere-like distribution
            init_pts = rng.normal(0, 1, size=(N, D))
        else:
            init_pts = rng.normal(0, 1, size=(N, D))
        
        init_flat = init_pts.flatten()
        
        # Optimize the energy function
        # L-BFGS-B is efficient for this smooth function
        try:
            res = minimize(
                fun=lambda x: energy_and_grad(x)[0],
                x0=init_flat,
                jac=lambda x: energy_and_grad(x)[1],
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            opt_pts = res.x.reshape(N, D)
            score = get_score(opt_pts)
            
            if score < best_score:
                best_score = score
                best_pts = opt_pts
        except Exception as e:
            pass
            
    return best_pts

# EVOLVE-BLOCK-END
