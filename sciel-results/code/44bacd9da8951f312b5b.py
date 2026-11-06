import numpy as np
import time
from scipy.optimize import minimize

N = 100
_DEADLINE = 115.0

def _compute_radius(c):
    """Compute the achievable common radius for centres c (N,3)."""
    n = c.shape[0]
    # Wall distances
    wall = 0.5 - np.max(np.abs(c), axis=1)
    min_wall = np.min(wall)
    # Pairwise distances - vectorized
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_dist = np.min(dist)
    return min(min_dist / 2.0, min_wall)

def _lse_objective_and_grad(flat_c, beta):
    """
    Compute the log-sum-exp objective and gradient for sphere packing.
    flat_c: shape (3*N,)
    beta: temperature parameter
    Returns: objective value, gradient shape (3*N,)
    """
    c = flat_c.reshape(N, 3)
    
    # 1. Pairwise distances
    # diff[i, j, k] = c[i, k] - c[j, k]
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    # Avoid division by zero in gradient calculation
    dist = np.sqrt(dist_sq)
    np.fill_diagonal(dist, 0.0)
    
    # We want to minimize -log(sum(exp(-beta * dist))) ?
    # No, maximizing min dist is equivalent to minimizing log-sum-exp of exp(-beta * dist) ?
    # If dist is large, exp(-beta * dist) is small. Sum is small. Log is negative large.
    # We want LARGE dist. So we want to MINIMIZE log(sum(exp(-beta * dist)))?
    # Let's check limits.
    # If all dist are equal to d, sum = M * exp(-beta * d). Log = log(M) - beta * d.
    # Minimizing this means maximizing beta * d, i.e., maximizing d. Correct.
    
    # Calculate terms
    # Mask diagonal to avoid self-interaction
    mask = ~np.eye(N, dtype=bool)
    
    # For numerical stability in log-sum-exp, subtract max
    # But here exp(-beta * dist) is <= 1. So max is 1 (if dist=0).
    # dist > 0 for i != j.
    
    # We compute the sum of exp(-beta * dist) for i < j
    # To be efficient, we can use the full matrix and divide by 2 later, or just sum upper triangle.
    # Gradient is easier with full matrix.
    
    # Let S = sum_{i != j} exp(-beta * dist_{ij})
    # Note: this counts each pair twice.
    # J_pair = log(S)
    # dJ/dc_i = (1/S) * sum_{j != i} d(exp(-beta * dist_{ij}))/dc_i
    # d(exp(-beta * dist))/dc_i = exp(-beta * dist) * (-beta) * (c_i - c_j) / dist
    
    # Compute exp terms
    # dist is (N, N). Diagonal is 0. exp(0) = 1.
    # We should exclude diagonal from sum.
    exp_terms = np.exp(-beta * dist)
    np.fill_diagonal(exp_terms, 0.0)
    
    S_pair = np.sum(exp_terms)
    
    if S_pair == 0:
        # Fallback if underflow
        return 0.0, np.zeros_like(flat_c)
        
    # Gradient contribution from pairs
    # grad_i = (1/S_pair) * sum_{j} exp(-beta * dist_{ij}) * (-beta) * (c_i - c_j) / dist_{ij}
    # Note: dist_{ij} = dist_{ji}.
    
    # We need to handle dist=0 case in division.
    # If dist=0, the term is 0 in sum (because we zeroed diagonal), so it doesn't contribute.
    # But if two distinct spheres coincide, dist=0.
    # In that case, gradient is undefined. We can add a small epsilon to dist.
    
    dist_safe = np.where(dist > 1e-8, dist, 1.0)
    
    # Coefficient for each pair (i, j) in gradient of c_i
    # coeff_{ij} = exp(-beta * dist_{ij}) * (-beta) / dist_safe_{ij}
    coeffs = exp_terms * (-beta) / dist_safe
    
    # grad_i = (1/S_pair) * sum_j coeffs_{ij} * (c_i - c_j)
    # c_i - c_j is diff[i, j]
    # sum_j coeffs_{ij}[:, :, np.newaxis] * diff[i, j, k]
    
    grad_pair = np.sum(coeffs[:, :, np.newaxis] * diff, axis=1)
    grad_pair /= S_pair
    
    # 2. Wall distances
    # w_{ik} = 0.5 - |c_{ik}|
    # We want to maximize min w.
    # Similar LSE: J_wall = log(sum_{i,k} exp(-beta * w_{ik}))
    # Minimize J_wall to maximize min w.
    
    abs_c = np.abs(c)
    w = 0.5 - abs_c
    # w can be negative if out of bounds, but we assume bounds are respected or penalty is high.
    # With L-BFGS-B bounds, c stays in [-0.5, 0.5]. So w >= 0.
    
    exp_w = np.exp(-beta * w)
    S_wall = np.sum(exp_w)
    
    if S_wall == 0:
        grad_wall = np.zeros_like(c)
    else:
        # d(exp(-beta * w))/dc_{ik} = exp(-beta * w) * (-beta) * d(-|c_{ik}|)/dc_{ik}
        # d(-|c|)/dc = -sign(c)
        # So term = exp(-beta * w) * (-beta) * (-sign(c)) = beta * sign(c) * exp(-beta * w)
        
        sign_c = np.sign(c)
        # Handle c=0: sign is 0.
        
        grad_wall_terms = beta * sign_c * exp_w
        grad_wall = np.sum(grad_wall_terms, axis=0) # Sum over k? No, w is per component.
        # Wait, J_wall = log( sum_{i=1}^N sum_{k=1}^3 exp(-beta * w_{ik}) )
        # dJ/d c_{ik} = (1/S_wall) * beta * sign(c_{ik}) * exp(-beta * w_{ik})
        
        grad_wall = (beta * sign_c * exp_w) / S_wall
        
    # Combine gradients
    # Weighting: Pairwise distances are usually more critical for packing density.
    # We can weight them. Let's start with equal weights or tune.
    # The magnitude of pair gradient depends on N. Wall on 3N.
    # Let's just add them.
    
    grad_total = grad_pair + grad_wall
    
    # Objective value
    obj = np.log(S_pair) + np.log(S_wall)
    
    # Flatten gradient
    grad_flat = grad_total.reshape(-1)
    
    return obj, grad_flat

def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE
    
    # Initial guess: Grid
    # 5x5x4 grid
    nx, ny, nz = 5, 5, 4
    xs = (np.arange(nx) + 0.5) / nx - 0.5
    ys = (np.arange(ny) + 0.5) / ny - 0.5
    zs = (np.arange(nz) + 0.5) / nz - 0.5
    c0 = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    
    best_c = c0.copy()
    best_r = _compute_radius(best_c)
    
    # Annealing schedule
    # Start with low beta, increase to high beta
    betas = [10.0, 50.0, 100.0, 200.0, 500.0, 1000.0]
    
    current_c = c0.copy().flatten()
    
    for beta in betas:
        if time.time() > deadline - 5:
            break
            
        # Run optimization for this beta
        # Use L-BFGS-B
        # Bounds: all coordinates in [-0.5, 0.5]
        bounds = [(-0.5, 0.5)] * (N * 3)
        
        # We might need to run multiple steps or let it converge
        # L-BFGS-B is efficient.
        
        try:
            res = minimize(
                _lse_objective_and_grad,
                current_c,
                args=(beta,),
                jac=True,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            current_c = res.x
        except Exception as e:
            # If optimization fails, keep previous
            pass
        
        c_curr = current_c.reshape(N, 3)
        r_curr = _compute_radius(c_curr)
        
        if r_curr > best_r:
            best_r = r_curr
            best_c = c_curr.copy()
            
    # Final polish: Try to increase radius slightly using local search or simple relaxation
    # The LSE method tends to find good configurations.
    # We can try a few random restarts if time permits?
    # Given the deadline, we should stick to the best found.
    
    # Ensure validity
    best_c = np.clip(best_c, -0.49999, 0.49999)
    
    return best_c
