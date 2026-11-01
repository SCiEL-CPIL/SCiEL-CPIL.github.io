"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50

# EVOLVE-BLOCK-START

def _score(points):
    """Compute the packing score: min of inter-circle distances/2 and boundary margins."""
    points = np.asarray(points)
    n = points.shape[0]
    # Boundary constraint: 1 - |c_i|
    dists_from_origin = np.linalg.norm(points, axis=1)
    boundary_margin = np.min(1.0 - dists_from_origin)
    
    # Pairwise distances
    # Compute all pairwise differences
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pairwise = np.min(dists) / 2.0
    
    return min(boundary_margin, min_pairwise)


def _soft_score(points, beta):
    """
    Smooth surrogate for the packing score.
    Uses log-sum-exp to create a smooth approximation of the min operation.
    f(points) = -1/beta * log(sum(exp(-beta * d_ij/2))) - 1/beta * log(sum(exp(-beta * (1-|c_i|))))
    We want to maximize the min, so we minimize -soft_score.
    """
    points = np.asarray(points)
    n = points.shape[0]
    
    # Boundary terms
    dists_from_origin = np.linalg.norm(points, axis=1)
    boundary_terms = 1.0 - dists_from_origin
    # Avoid log(0) or log(negative) by clipping
    boundary_terms = np.maximum(boundary_terms, 1e-12)
    log_boundary = np.log(np.sum(np.exp(-beta * boundary_terms)))
    soft_boundary = -1.0 / beta * log_boundary
    
    # Pairwise terms
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    # We only want unique pairs, but summing all off-diagonals is fine for the log-sum-exp
    # Actually, to be consistent with min, we should look at the smallest.
    # log-sum-exp approximates -1/beta * log(sum(exp(-beta*x_i))) -> min(x_i) as beta->inf
    # But summing all pairs doubles the count for each pair (i,j) and (j,i).
    # This doesn't change the argmin, but affects the gradient magnitude slightly.
    # Let's just use the upper triangle to be cleaner.
    iu = np.triu_indices(n, k=1)
    pair_dists = dists[iu]
    pair_dists = np.maximum(pair_dists, 1e-12)
    log_pairs = np.log(np.sum(np.exp(-beta * pair_dists / 2.0)))
    soft_pairs = -1.0 / beta * log_pairs
    
    # The score is min(boundary_margin, min_pairwise/2)
    # So soft_score = min(soft_boundary, soft_pairs) is not quite right because min is not linear.
    # Actually, the objective is min(A, B).
    # A smooth approximation of min(A,B) is -1/beta * log(exp(-beta*A) + exp(-beta*B)).
    # Here A = boundary_margin, B = min_pairwise/2.
    # But A and B are themselves mins.
    # Let's approximate the whole thing.
    # Total energy E = -beta * min( min_i (1-|c_i|), min_{i<j} |c_i-c_j|/2 )
    # E ≈ -beta * ( -1/beta * log( sum_i exp(-beta(1-|c_i|)) + sum_{i<j} exp(-beta|c_i-c_j|/2) ) )
    # E ≈ log( sum_i exp(-beta(1-|c_i|)) + sum_{i<j} exp(-beta|c_i-c_j|/2) )
    # We want to maximize the score, which is equivalent to minimizing -score.
    # So we minimize: log( sum_i exp(-beta(1-|c_i|)) + sum_{i<j} exp(-beta|c_i-c_j|/2) )
    
    all_terms = np.concatenate([np.exp(-beta * boundary_terms), np.exp(-beta * pair_dists / 2.0)])
    # To avoid overflow in exp, we can shift by max, but beta is large and terms are small/negative.
    # Actually, exp(-beta * x) where x is positive (margin) is small.
    # If x is close to 0, exp(0)=1.
    # Sum of these terms.
    s = np.sum(all_terms)
    if s == 0:
        return 0.0
    return np.log(s)


def _soft_score_grad(points, beta):
    """
    Gradient of the soft score objective: log(sum(exp(-beta * terms)))
    terms = [1-|c_i| for i, |c_i-c_j|/2 for i<j]
    """
    points = np.asarray(points)
    n = points.shape[0]
    grad = np.zeros_like(points)
    
    # Boundary terms
    dists_from_origin = np.linalg.norm(points, axis=1)
    # Avoid division by zero
    safe_dists = np.maximum(dists_from_origin, 1e-12)
    boundary_terms = 1.0 - dists_from_origin
    exp_boundary = np.exp(-beta * boundary_terms)
    
    # Gradient of log(sum(exp(-beta * (1-|c_i|)))) wrt c_i
    # d/dc_i log(S) = (1/S) * dS/dc_i
    # dS/dc_i = sum_k exp(-beta * term_k) * (-beta) * d(term_k)/dc_i
    # For boundary term k=i: term_i = 1-|c_i|, d/dc_i = -c_i/|c_i|
    # So contribution to grad_i = exp(-beta*term_i) * (-beta) * (-c_i/|c_i|) / S
    # = beta * exp(-beta*term_i) * (c_i/|c_i|) / S
    
    S_boundary = np.sum(exp_boundary)
    # We need the total S (boundary + pairs) for the normalization
    # Let's compute S_total later.
    
    # For now, let's compute the numerator for boundary part
    # grad_boundary_i = beta * exp_boundary[i] * (points[i] / safe_dists[i])
    grad_boundary = beta * exp_boundary[:, np.newaxis] * (points / safe_dists[:, np.newaxis])
    
    # Pairwise terms
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    iu = np.triu_indices(n, k=1)
    pair_dists = dists[iu]
    exp_pairs = np.exp(-beta * pair_dists / 2.0)
    
    # Gradient of log(S) wrt c_i from pair (i,j)
    # term_ij = |c_i - c_j| / 2
    # d/dc_i term_ij = (c_i - c_j) / (2 * |c_i - c_j|)
    # Contribution to grad_i from pair (i,j): exp(-beta*term_ij) * (-beta) * d(term_ij)/dc_i / S_total
    # = -beta * exp(-beta*term_ij) * (c_i - c_j) / (2 * dist_ij) / S_total
    # Contribution to grad_j from pair (i,j): exp(-beta*term_ij) * (-beta) * d(term_ij)/dc_j / S_total
    # = -beta * exp(-beta*term_ij) * (c_j - c_i) / (2 * dist_ij) / S_total
    # = beta * exp(-beta*term_ij) * (c_i - c_j) / (2 * dist_ij) / S_total
    
    # Let's accumulate these
    # For each pair (i, j) with weight w_ij = beta * exp(-beta * dist_ij / 2) / (2 * dist_ij)
    # grad_i += w_ij * (c_i - c_j)
    # grad_j -= w_ij * (c_i - c_j)
    
    safe_pair_dists = np.maximum(pair_dists, 1e-12)
    weights = beta * exp_pairs / (2.0 * safe_pair_dists)
    
    # We need to scatter these gradients
    # This is tricky to vectorize without loops for the scatter, but we can use np.add.at
    # Or we can build a matrix.
    # Let's use a loop for the scatter as N=50 is small, or vectorize with advanced indexing.
    # Vectorized scatter:
    # For each pair (i, j), we add to grad[i] and subtract from grad[j].
    # We can create an array of indices and values.
    
    # Actually, let's just compute the full gradient matrix G where G[i,j] is the vector contribution to i from j.
    # This is O(N^2) memory, which is fine for N=50.
    
    # Let's stick to a simpler approach: compute the full pairwise gradient matrix.
    # grad_from_pairs[i] = sum_{j != i} beta * exp(-beta * |c_i-c_j|/2) * (c_i - c_j) / (2 * |c_i-c_j|)
    
    # We can compute this using the full distance matrix.
    # exp_full = exp(-beta * dists / 2) with diagonal 0
    exp_full = np.zeros_like(dists)
    exp_full[iu] = exp_pairs
    exp_full[(iu[1], iu[0])] = exp_pairs  # Symmetric
    
    # weights_full[i,j] = beta * exp_full[i,j] / (2 * dists[i,j])
    safe_dists_full = np.maximum(dists, 1e-12)
    np.fill_diagonal(safe_dists_full, 1.0)  # Avoid div by zero on diagonal
    weights_full = beta * exp_full / (2.0 * safe_dists_full)
    
    # grad_from_pairs[i] = sum_j weights_full[i,j] * (c_i - c_j)
    # = sum_j weights_full[i,j] * c_i - sum_j weights_full[i,j] * c_j
    # = c_i * sum_j weights_full[i,j] - (weights_full @ points)[i]
    
    row_sums = np.sum(weights_full, axis=1)
    grad_from_pairs = points * row_sums[:, np.newaxis] - (weights_full @ points)
    
    # Total S for normalization
    S_total = S_boundary + np.sum(exp_pairs)
    if S_total == 0:
        return 0.0, np.zeros_like(points)
    
    total_grad = (grad_boundary + grad_from_pairs) / S_total
    
    # The objective is log(S_total).
    # d/dc log(S) = (1/S) * dS/dc
    # We computed dS/dc = grad_boundary + grad_from_pairs (numerator part)
    # So the gradient of the objective is total_grad.
    
    # We want to minimize this objective.
    return np.log(S_total), total_grad


def _hex_lattice_init(n):
    """Generate a hexagonal lattice initial guess."""
    spacing = 2.0 / (np.sqrt(n) + 1.0)
    points = []
    k = int(1.0 / spacing) + 2
    for i in range(-k, k + 1):
        for j in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if np.hypot(x, y) <= 1.0 - spacing / 2.0:
                points.append([x, y])
    points = np.array(points)
    if len(points) >= n:
        # Sort by radius and take first n
        radius = np.linalg.norm(points, axis=1)
        indices = np.argsort(radius, kind="stable")[:n]
        return points[indices]
    else:
        # Fallback: random
        return np.random.rand(n, 2) * 2 - 1


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Initial guess
    points = _hex_lattice_init(N)
    
    # Optimization loop with increasing beta
    betas = [10, 50, 100, 500, 1000, 5000, 10000]
    
    current_points = points.copy()
    best_score = _score(current_points)
    best_points = current_points.copy()
    
    for beta in betas:
        if time.time() > deadline:
            break
            
        # Define objective and gradient
        def obj(x):
            val, grad = _soft_score_grad(x.reshape(-1, 2), beta)
            return val, grad
        
        # L-BFGS-B
        res = minimize(obj, current_points.flatten(), jac=True, method='L-BFGS-B',
                       options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
        
        if res.success or res.fun < 1e10:
            new_points = res.x.reshape(-1, 2)
            # Check validity
            if np.all(np.isfinite(new_points)):
                score = _score(new_points)
                if score > best_score:
                    best_score = score
                    best_points = new_points.copy()
                current_points = new_points
    
    # Final polish with high beta
    if time.time() < deadline:
        for beta in [50000, 100000]:
            if time.time() > deadline:
                break
            def obj(x):
                val, grad = _soft_score_grad(x.reshape(-1, 2), beta)
                return val, grad
            res = minimize(obj, current_points.flatten(), jac=True, method='L-BFGS-B',
                           options={'maxiter': 100, 'ftol': 1e-15, 'gtol': 1e-13})
            if res.success or res.fun < 1e10:
                new_points = res.x.reshape(-1, 2)
                if np.all(np.isfinite(new_points)):
                    score = _score(new_points)
                    if score > best_score:
                        best_score = score
                        best_points = new_points.copy()
                    current_points = new_points
    
    return best_points

# EVOLVE-BLOCK-END
