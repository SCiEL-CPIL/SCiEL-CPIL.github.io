"""Improved program for packing 300 circles in a unit circle using smooth soft-min optimization."""
import numpy as np
import time
from scipy.optimize import minimize

N = 300

# EVOLVE-BLOCK-START

def _compute_score(points):
    """Compute the minimum radius for a given set of points."""
    n = len(points)
    if n < 2:
        return min(1.0 - np.linalg.norm(points[0])) if n == 1 else 0.0
    
    # Compute all pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists) / 2.0
    
    # Compute distance to boundary
    boundary_dists = 1.0 - np.linalg.norm(points, axis=1)
    min_boundary = np.min(boundary_dists)
    
    return min(min_pair_dist, min_boundary)

def _hex_lattice_init(n, seed=42):
    """
    Initialize points in a hexagonal lattice clipped to the unit circle.
    """
    rng = np.random.default_rng(seed)
    points = []
    # Estimate spacing. r ~ 1/sqrt(N). Diameter ~ 2r.
    # Hex lattice spacing s = 2r.
    # Area of hexagon per point ~ s^2 * sqrt(3)/2.
    # N * s^2 * sqrt(3)/2 ~ pi.
    # s ~ sqrt(2*pi / (N*sqrt(3))).
    s = np.sqrt(2.0 * np.pi / (n * np.sqrt(3.0)))
    
    # Generate a large grid of hex lattice points
    # Hex lattice coordinates:
    # x = s * (i + 0.5 * (j % 2))
    # y = s * j * sqrt(3) / 2
    max_k = int(1.0 / (s * 0.5)) + 2
    
    candidates = []
    for j in range(-max_k, max_k + 1):
        for i in range(-max_k, max_k + 1):
            x = s * (i + 0.5 * (j % 2))
            y = s * j * np.sqrt(3.0) / 2.0
            if x*x + y*y <= 1.0:
                candidates.append([x, y])
    
    candidates = np.array(candidates)
    
    if len(candidates) < n:
        # If we didn't generate enough, shrink spacing slightly and retry
        s *= 0.9
        max_k = int(1.0 / (s * 0.5)) + 3
        candidates = []
        for j in range(-max_k, max_k + 1):
            for i in range(-max_k, max_k + 1):
                x = s * (i + 0.5 * (j % 2))
                y = s * j * np.sqrt(3.0) / 2.0
                if x*x + y*y <= 1.0:
                    candidates.append([x, y])
        candidates = np.array(candidates)
        
    if len(candidates) >= n:
        # Sort by distance to origin to pick the most central ones
        radii = np.linalg.norm(candidates, axis=1)
        indices = np.argsort(radii)[:n]
        points = candidates[indices]
    else:
        # Fallback: random points
        points = rng.random((n, 2)) * 2 - 1
        norms = np.linalg.norm(points, axis=1)
        mask = norms > 1.0
        points[mask] = points[mask] / norms[mask, np.newaxis] * 0.9

    # Add small random perturbation to break symmetry
    points += rng.normal(0, 1e-4, points.shape)
    
    return points

def _soft_min_objective_and_grad(x, beta):
    """
    Compute the soft-min objective and its gradient.
    We want to maximize min(constraint distances), so we minimize -soft_min.
    """
    p = x.reshape(-1, 2)
    n = p.shape[0]
    
    # Calculate distances
    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    
    # Flatten distances for vectorization
    # Pair distances: upper triangle
    iu = np.triu_indices(n, k=1)
    pair_dists = dists[iu] / 2.0
    
    # Boundary distances
    norms = np.linalg.norm(p, axis=1)
    bound_dists = 1.0 - norms
    
    # Combine all constraint distances
    # We want to compute soft_min(c) = -1/beta * log(sum(exp(-beta*c)))
    # Objective: -soft_min(c) = 1/beta * log(sum(exp(-beta*c)))
    
    z_pairs = -beta * pair_dists
    z_bounds = -beta * bound_dists
    
    # Max values for stability
    max_z_pair = np.max(z_pairs)
    max_z_bound = np.max(z_bounds)
    max_z = max(max_z_pair, max_z_bound)
    
    # exp terms
    exp_pairs = np.exp(z_pairs - max_z)
    exp_bounds = np.exp(z_bounds - max_z)
    
    sum_exp = np.sum(exp_pairs) + np.sum(exp_bounds)
    
    # log_sum_exp = max_z + log(sum_exp)
    log_sum_exp = max_z + np.log(sum_exp)
    
    # Objective value: -soft_min = 1/beta * log_sum_exp
    val = log_sum_exp / beta
    
    # Gradient
    # grad(soft_min) = (1/sum_exp) * sum_k exp(-beta*c_k) * grad(c_k)
    # grad(-soft_min) = -grad(soft_min)
    # We want grad of objective = grad(-soft_min)
    
    # Weights
    w_pairs = exp_pairs / sum_exp
    w_bounds = exp_bounds / sum_exp
    
    grad = np.zeros_like(p)
    
    # Pair contributions
    # c_{ij} = 0.5 * |p_i - p_j|
    # grad_{p_i} c_{ij} = 0.5 * (p_i - p_j) / |p_i - p_j|
    # grad_{p_j} c_{ij} = 0.5 * (p_j - p_i) / |p_i - p_j|
    
    # Vectorized pair gradient
    i_idx, j_idx = iu
    
    # Avoid division by zero
    safe_pair_dists = np.where(pair_dists > 1e-12, pair_dists, 1.0)
    
    # Coefficient for each pair
    # w_k * 0.25 / pair_dists[k]
    coeff = w_pairs * 0.25 / safe_pair_dists
    
    # Vector from j to i
    vecs = p[i_idx] - p[j_idx]
    
    # Accumulate gradients
    # grad[p_i] += coeff * vecs
    # grad[p_j] -= coeff * vecs
    
    np.add.at(grad, i_idx, coeff[:, np.newaxis] * vecs)
    np.add.at(grad, j_idx, -coeff[:, np.newaxis] * vecs)
    
    # Boundary contributions
    # c_i = 1 - |p_i|
    # grad_{p_i} c_i = - p_i / |p_i|
    
    safe_norms = np.where(norms > 1e-12, norms, 1.0)
    bound_grad_vecs = -p / safe_norms[:, np.newaxis]
    
    # grad[p_i] += w_bounds[i] * bound_grad_vecs[i]
    grad += w_bounds[:, np.newaxis] * bound_grad_vecs
    
    # The objective is -soft_min, so gradient is -grad(soft_min)
    # Wait, my derivation:
    # f = -soft_min
    # grad(f) = -grad(soft_min)
    # grad(soft_min) = sum_k p_k * grad(c_k) where p_k = exp(-beta*c_k)/sum(exp(-beta*c))
    # So grad(f) = -sum_k p_k * grad(c_k)
    
    # But in my code above, I computed:
    # For pairs: grad[p_i] += w_pairs * 0.5 * (p_i - p_j) / dists[i,j]
    # This is grad(soft_min) contribution.
    # So I need to negate it.
    
    grad = -grad
    
    return val, grad

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # 1. Initialization
    points = _hex_lattice_init(N)
    best_points = points.copy()
    best_score = _compute_score(points)
    
    # 2. Optimization using L-BFGS-B with smooth soft-min objective
    # We'll use a sequence of increasing beta values to gradually sharpen the soft-min
    
    # Precompute upper triangle indices for efficiency
    iu = np.triu_indices(N, k=1)
    
    def objective_and_grad(x, beta):
        return _soft_min_objective_and_grad(x, beta)
    
    # Try a few beta values
    beta_values = [10.0, 50.0, 100.0, 200.0]
    
    for beta in beta_values:
        if time.time() > deadline:
            break
            
        def obj(x):
            val, _ = _soft_min_objective_and_grad(x, beta)
            return val
        
        def grad(x):
            _, g = _soft_min_objective_and_grad(x, beta)
            return g
        
        # Run L-BFGS-B
        try:
            result = minimize(
                obj,
                points,
                jac=grad,
                method='L-BFGS-B',
                options={
                    'maxiter': 1000,
                    'maxfun': 5000,
                    'ftol': 1e-12,
                    'gtol': 1e-8,
                    'maxls': 20
                }
            )
            
            new_points = result.x
            new_score = _compute_score(new_points)
            
            if new_score > best_score:
                best_score = new_score
                best_points = new_points.copy()
                points = new_points.copy()
                
        except Exception:
            pass
        
        if time.time() > deadline:
            break
    
    # 3. Local refinement with steepest descent if time permits
    if time.time() < deadline - 10.0:
        # Simple gradient ascent on the actual min function using subgradients
        # This is a heuristic refinement step
        
        current_points = best_points.copy()
        lr = 1e-4
        steps = 1000
        
        for step in range(steps):
            if time.time() > deadline:
                break
                
            p = current_points
            diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=-1))
            np.fill_diagonal(dists, np.inf)
            
            # Find the minimum distance pair and boundary
            min_pair_dist = np.min(dists) / 2.0
            norms = np.linalg.norm(p, axis=1)
            boundary_dists = 1.0 - norms
            min_boundary = np.min(boundary_dists)
            
            current_score = min(min_pair_dist, min_boundary)
            
            # Compute subgradient
            grad = np.zeros_like(p)
            
            if min_pair_dist < min_boundary:
                # The bottleneck is a pair
                # Find the pair with minimum distance
                min_idx = np.unravel_index(np.argmin(dists), dists.shape)
                i, j = min_idx[0], min_idx[1]
                if i > j:
                    i, j = j, i
                if i == j:
                    continue
                    
                vec = p[i] - p[j]
                dist = np.linalg.norm(vec)
                if dist > 1e-12:
                    unit_vec = vec / dist
                    # To increase distance, move i away from j and j away from i
                    grad[i] += 0.5 * unit_vec
                    grad[j] -= 0.5 * unit_vec
            else:
                # The bottleneck is a boundary point
                min_bound_idx = np.argmin(boundary_dists)
                i = min_bound_idx
                norm_i = np.linalg.norm(p[i])
                if norm_i > 1e-12:
                    unit_vec = p[i] / norm_i
                    # To increase distance to boundary, move inward
                    grad[i] -= unit_vec
            
            # Normalize gradient
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 1e-12:
                grad = grad / grad_norm
            
            # Update
            new_points = p + lr * grad
            
            # Check if improved
            new_score = _compute_score(new_points)
            if new_score > current_score:
                current_points = new_points
            else:
                lr *= 0.95
    
    # Final check
    final_score = _compute_score(best_points)
    if final_score > best_score:
        best_score = final_score
    
    return best_points

# EVOLVE-BLOCK-END
