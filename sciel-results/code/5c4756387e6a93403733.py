"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _hexagonal_seed(n):
    """Generate a hexagonal lattice seed for n circles."""
    spacing = 2.0 / (np.sqrt(n) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        mask = radius <= 1.0 - spacing / 2.0
        points = points[mask]
        if len(points) >= n:
            idx = np.argsort(radius[mask], kind="stable")[:n]
            return points[idx]
        spacing *= 0.98

def _compute_score(points):
    """Compute the packing score: min of pairwise dist/2 and boundary dist."""
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    radii = np.sqrt(np.sum(points ** 2, axis=1))
    min_boundary = np.min(1.0 - radii)
    return min(min_pair, min_boundary)

def _soft_min_objective(x, beta):
    """
    Objective function for scipy.optimize.minimize.
    We want to maximize r = min(d_ij/2, 1-|c_i|).
    This is equivalent to minimizing -r.
    We use a soft-min approximation:
    soft_min(y) = -1/beta * log(sum(exp(-beta * y_k)))
    We want to minimize -soft_min(y) = 1/beta * log(sum(exp(-beta * y_k)))
    y_k are the distances: d_ij/2 for all i<j, and 1-|c_i| for all i.
    """
    pts = x.reshape(-1, 2)
    
    # Pairwise distances / 2
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=-1)
    # Avoid log(0) or issues with diagonal
    np.fill_diagonal(dists_sq, np.inf)
    # We only need upper triangle
    iu = np.triu_indices(N, k=1)
    pair_dists = np.sqrt(dists_sq[iu]) / 2.0
    
    # Boundary distances
    radii = np.sqrt(np.sum(pts ** 2, axis=1))
    bound_dists = 1.0 - radii
    
    # Combine all distances
    # To avoid underflow in exp(-beta * y), we can shift by min
    y = np.concatenate([pair_dists, bound_dists])
    
    # Soft min calculation
    # For numerical stability: soft_min(y) = min(y) - 1/beta * log(sum(exp(-beta*(y - min(y)))))
    y_min = np.min(y)
    # If y is very large, exp(-beta*(y-y_min)) is fine.
    # If y is very small, it's fine.
    # We want to minimize -soft_min(y) = -y_min + 1/beta * log(sum(exp(-beta*(y - y_min))))
    
    # Clip y - y_min to avoid overflow in exp if beta is large and differences are large
    # But y - y_min >= 0.
    # exp(-beta * diff) <= 1. So sum is <= N*(N-1)/2 + N.
    # log(sum) is safe.
    
    diffs = y - y_min
    # beta * diffs can be large, exp(-large) -> 0.
    # exp(-0) = 1.
    exp_terms = np.exp(-beta * diffs)
    log_sum = np.log(np.sum(exp_terms))
    
    obj = -y_min + (1.0 / beta) * log_sum
    return obj

def _gradient(x, beta):
    """
    Gradient of the soft-min objective.
    f(x) = -soft_min(y(x)) = y_min(x) - 1/beta * log(sum(exp(-beta*(y_k(x) - y_min(x)))))
    This is complicated due to y_min.
    Alternative: Use the direct derivative of the log-sum-exp form without subtracting min first,
    or use finite differences if analytic is too complex.
    Given time constraints, let's try to derive it properly or use a simpler smooth function.
    
    Actually, let's use the property:
    g(x) = 1/beta * log( sum_k exp(-beta * y_k(x)) )
    dg/dx = 1/beta * 1/sum(exp) * sum( exp(-beta*y_k) * (-beta * dy_k/dx) )
    dg/dx = - sum( w_k * dy_k/dx )  where w_k = exp(-beta*y_k) / sum(exp(-beta*y_j))
    
    This is a weighted average of the gradients of the individual distances, with weights proportional to exp(-beta*y_k).
    This is much easier to compute.
    """
    pts = x.reshape(-1, 2)
    
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=-1)
    np.fill_diagonal(dists_sq, np.inf)
    
    # We need gradients for all terms in the sum.
    # Terms:
    # 1. Pairwise: y_k = 0.5 * ||p_i - p_j||
    #    dy_k/dp_i = 0.5 * (p_i - p_j) / ||p_i - p_j||
    #    dy_k/dp_j = 0.5 * (p_j - p_i) / ||p_i - p_j||
    # 2. Boundary: y_k = 1 - ||p_i||
    #    dy_k/dp_i = -p_i / ||p_i||
    
    # Compute weights
    # To avoid underflow, shift by min y
    iu = np.triu_indices(N, k=1)
    pair_dists = np.sqrt(dists_sq[iu]) / 2.0
    radii = np.sqrt(np.sum(pts ** 2, axis=1))
    bound_dists = 1.0 - radii
    
    y_all = np.concatenate([pair_dists, bound_dists])
    y_min = np.min(y_all)
    
    # Weights w_k = exp(-beta * (y_k - y_min))
    # Sum of weights S = sum(exp(-beta * (y_k - y_min)))
    # Normalized weights p_k = w_k / S
    
    exp_pair = np.exp(-beta * (pair_dists - y_min))
    exp_bound = np.exp(-beta * (bound_dists - y_min))
    
    S = np.sum(exp_pair) + np.sum(exp_bound)
    
    w_pair = exp_pair / S
    w_bound = exp_bound / S
    
    grad = np.zeros_like(x)
    
    # Pairwise contributions
    # For each pair (i, j) in iu, with weight w_pair[k]
    # grad[p_i] += w_pair[k] * 0.5 * (p_i - p_j) / d_ij
    # grad[p_j] += w_pair[k] * 0.5 * (p_j - p_i) / d_ij
    
    i_idx, j_idx = iu
    # d_ij is 2 * pair_dists
    d_ij = 2.0 * pair_dists
    # Avoid division by zero
    d_ij_safe = np.where(d_ij > 1e-12, d_ij, 1.0)
    
    # Vectorized update
    # diff_ij = p_i - p_j
    diff_ij = pts[i_idx] - pts[j_idx]
    
    # Unit vectors
    unit_vecs = diff_ij / d_ij_safe[:, np.newaxis]
    
    # Gradients for i
    grad_i = w_pair[:, np.newaxis] * 0.5 * unit_vecs
    # Gradients for j
    grad_j = w_pair[:, np.newaxis] * 0.5 * (-unit_vecs)
    
    # Accumulate
    np.add.at(grad, i_idx, grad_i)
    np.add.at(grad, j_idx, grad_j)
    
    # Boundary contributions
    # For each i, with weight w_bound[i]
    # grad[p_i] += w_bound[i] * (-p_i / ||p_i||)
    radii_safe = np.where(radii > 1e-12, radii, 1.0)
    unit_radii = pts / radii_safe[:, np.newaxis]
    
    grad_bound = w_bound[:, np.newaxis] * (-unit_radii)
    
    grad += grad_bound
    
    return grad

def construct():
    start_time = time.time()
    timeout = 140.0  # Safety margin before 150s kill
    
    # 1. Generate initial seed
    centers = _hexagonal_seed(N)
    x0 = centers.flatten()
    
    best_centers = centers.copy()
    best_r = _compute_score(centers)
    
    # 2. Smooth Optimization with L-BFGS-B
    # We try a few different beta values, increasing beta for sharper approximation
    betas = [10.0, 50.0, 200.0]
    
    bounds = [(-1.0, 1.0)] * (2 * N)
    
    for beta in betas:
        if time.time() - start_time > timeout - 20:
            break
            
        try:
            res = minimize(
                _soft_min_objective,
                x0,
                args=(beta,),
                method='L-BFGS-B',
                jac=_gradient,
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 1000}
            )
            
            if res.success or res.fun < _soft_min_objective(x0, beta):
                new_centers = res.x.reshape(-1, 2)
                new_r = _compute_score(new_centers)
                
                if new_r > best_r:
                    best_r = new_r
                    best_centers = new_centers
                    x0 = res.x
        except Exception:
            pass

    # 3. Final Polish with Exact Hill Climbing
    # Use the best found centers as start for hill climbing
    centers = best_centers.copy()
    current_r = best_r
    
    # Precompute distance matrix
    def calc_dist_matrix(c):
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        return np.linalg.norm(diff, axis=-1)
    
    dist_mat = calc_dist_matrix(centers)
    np.fill_diagonal(dist_mat, np.inf)
    
    boundary_dists = 1.0 - np.linalg.norm(centers, axis=1)
    
    step_size = 0.01
    no_improve_count = 0
    
    while time.time() - start_time < timeout:
        i = np.random.randint(N)
        
        for _ in range(10):
            delta = np.random.normal(0, step_size, size=2)
            new_c = centers[i] + delta
            r_new_c = np.linalg.norm(new_c)
            if r_new_c > 1.0:
                continue
            
            old_row = dist_mat[i].copy()
            new_dists = np.linalg.norm(centers - new_c, axis=1)
            new_dists[i] = np.inf
            dist_mat[i] = new_dists
            dist_mat[:, i] = new_dists
            
            min_pair = np.min(dist_mat) / 2.0
            boundary_dists[i] = 1.0 - r_new_c
            min_bound = np.min(boundary_dists)
            
            candidate_r = min(min_pair, min_bound)
            
            if candidate_r > current_r + 1e-12:
                centers[i] = new_c
                current_r = candidate_r
                no_improve_count = 0
                
                if current_r > best_r:
                    best_r = current_r
                    best_centers = centers.copy()
                
                step_size *= 1.02
            else:
                dist_mat[i] = old_row
                dist_mat[:, i] = old_row
                boundary_dists[i] = 1.0 - np.linalg.norm(centers[i])
                no_improve_count += 1
        
        if no_improve_count > 50:
            step_size *= 0.9
            no_improve_count = 0
            if step_size < 0.0005:
                break

    # Final validation
    radii = np.linalg.norm(best_centers, axis=1)
    if np.any(radii >= 1.0):
        best_centers = best_centers * (1.0 - 1e-8) / radii

    return best_centers
# EVOLVE-BLOCK-END
