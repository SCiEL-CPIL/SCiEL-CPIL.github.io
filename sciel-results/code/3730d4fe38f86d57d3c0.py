"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _pairwise_distances(c):
    """Compute all pairwise distances between N points."""
    # c has shape (N, 2)
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    return np.sqrt(np.sum(diff**2, axis=-1))

def _objective_and_grad(c):
    """
    Objective: maximize the minimum distance (which is 2*r).
    We minimize the negative of a smooth approximation of the minimum distance.
    Using log-sum-exp for smoothness:
    min_dist_smooth = -beta * log(sum(exp(-dist^2 / beta)))  ... wait, that's for min.
    Actually, we want to maximize min_dist.
    Let's minimize f(c) = -min_dist.
    A smooth approximation of -min_i d_i is:
    f(c) = beta * log(sum(exp(-d_i^2 / (2*beta))))  ... no.
    
    Let's use: f(c) = - (1/beta) * log( sum_{i<j} exp(-d_{ij}^2 / (2*beta)) )
    As beta -> 0, this approaches -min_{i<j} d_{ij}.
    We want to minimize this, which is equivalent to maximizing min d_{ij}.
    
    Wait, if d_min is the minimum distance, we want to maximize it.
    Let g(c) = min_{i<j} d_{ij}.
    We want to maximize g(c).
    Equivalent to minimizing -g(c).
    
    Smooth approximation of -min_{i<j} d_{ij}:
    h(c) = (1/beta) * log( sum_{i<j} exp(-d_{ij} / beta) )  ... no, distances can be 0.
    
    Better: use squared distances to avoid issues with 0.
    Let D_{ij} = d_{ij}^2.
    min D_{ij} is what we want to maximize (since sqrt is monotonic).
    Smooth min: m(D) = (1/beta) * log( sum exp(-D_{ij} / beta) )
    As beta -> 0, m(D) -> min D_{ij}.
    So we want to maximize m(D).
    Equivalent to minimizing -m(D).
    
    Objective: f(c) = -m(D) = -(1/beta) * log( sum_{i<j} exp(-D_{ij} / beta) )
    
    Gradient:
    df/dc_k = -d/dc_k [ -(1/beta) log(sum exp(-D/beta)) ]
    = (1/beta) * [ (1/sum_exp) * sum_j (d/dx_k exp(-D_{kj}/beta)) ]
    = (1/beta) * [ (1/sum_exp) * sum_j exp(-D_{kj}/beta) * (-1/beta) * dD_{kj}/dx_k ]
    = -(1/beta^2) * [ (1/sum_exp) * sum_j exp(-D_{kj}/beta) * (x_k - x_j) * 2 ]  ... wait, D = ||x_k - x_j||^2
    dD/dx_k = 2(x_k - x_j)
    
    Let's simplify. Let w_j = exp(-D_{kj}/beta) / sum_m exp(-D_{km}/beta) for all j != k.
    Then df/dx_k = -(2/beta) * sum_{j != k} w_j (x_k - x_j)
    
    This is a weighted average of vectors from x_k to x_j, scaled by -2/beta.
    """
    beta = 0.01  # Temperature for log-sum-exp
    c = c.reshape(N, 2)
    
    # Compute squared distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    D = np.sum(diff**2, axis=-1)
    np.fill_diagonal(D, np.inf)  # Ignore self-distances
    
    # For numerical stability, subtract max
    D_flat = D.ravel()
    D_max = np.max(D_flat[D_flat < np.inf])
    
    # Compute weights
    exp_terms = np.exp(-(D - D_max) / beta)
    np.fill_diagonal(exp_terms, 0)
    
    sum_exp = np.sum(exp_terms, axis=1)
    weights = exp_terms / sum_exp[:, np.newaxis]
    
    # Objective value
    log_sum = np.log(sum_exp + 1e-300)
    f_val = -np.mean(log_sum - D_max / beta)  # This is -m(D) averaged? No, we want the global min.
    
    # Actually, the log-sum-exp approximates the min over all pairs.
    # The formula above computes for each k, the min over j. But we want the global min over all pairs.
    # Let's recompute for the global min.
    
    D_flat_valid = D[D < np.inf].ravel()
    D_max_global = np.max(D_flat_valid)
    exp_global = np.exp(-(D - D_max_global) / beta)
    np.fill_diagonal(exp_global, 0)
    
    sum_global = np.sum(exp_global)
    log_sum_global = np.log(sum_global + 1e-300)
    
    # Smooth min of D
    smooth_min_D = D_max_global - beta * log_sum_global
    
    # We want to maximize smooth_min_D, so minimize -smooth_min_D
    f_val = -smooth_min_D
    
    # Gradient
    # df/dx_k = -(2/beta) * sum_{j != k} w_j (x_k - x_j)
    # where w_j = exp(-(D_kj - D_max_global)/beta) / sum_global
    
    # This is not quite right because the weights are global, not per-k.
    # Let's derive properly.
    # f = -[D_max - beta * log(sum_{i<j} exp(-(D_ij - D_max)/beta))]
    #   = -D_max + beta * log(sum_{i<j} exp(-(D_ij - D_max)/beta))
    
    # df/dx_k = beta * [1/sum_exp] * sum_{j != k} exp(-(D_kj - D_max)/beta) * (-1/beta) * dD_kj/dx_k
    #         = -[1/sum_exp] * sum_{j != k} exp(-(D_kj - D_max)/beta) * 2(x_k - x_j)
    
    grad = np.zeros_like(c)
    for k in range(N):
        for j in range(N):
            if k != j:
                w = exp_global[k, j] / sum_global
                grad[k] += -2 * w * (c[k] - c[j])
    
    return f_val, grad

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_score = -np.inf
    best_centers = None
    
    # Initial guess: hexagonal lattice
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
        if len(points) >= N:
            init_centers = points[np.argsort(radius, kind="stable")[:N]]
            break
        spacing *= 0.98
    
    # Multi-start L-BFGS-B
    n_restarts = 20
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        if restart == 0:
            c0 = init_centers.copy()
        else:
            # Add random perturbation
            c0 = init_centers + np.random.normal(0, 0.1, (N, 2))
            # Ensure inside unit circle
            r = np.hypot(c0[:, 0], c0[:, 1])
            mask = r > 0.9
            c0[mask] *= 0.9 / r[mask, np.newaxis]
        
        # Run L-BFGS-B
        try:
            result = minimize(
                _objective_and_grad,
                c0.ravel(),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            c_opt = result.x.reshape(N, 2)
            
            # Evaluate actual score
            r = np.hypot(c_opt[:, 0], c_opt[:, 1])
            if np.any(r >= 1.0):
                continue
            
            # Compute min distance
            D = _pairwise_distances(c_opt)
            np.fill_diagonal(D, np.inf)
            min_dist = np.min(D)
            min_r = min(min_dist / 2.0, 1.0 - np.max(r))
            
            if min_r > best_score:
                best_score = min_r
                best_centers = c_opt.copy()
                
        except Exception:
            continue
    
    # If no valid result, return initial guess
    if best_centers is None:
        return init_centers
    
    return best_centers
# EVOLVE-BLOCK-END
