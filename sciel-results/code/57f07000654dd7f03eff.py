"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_radius(pts):
    """Compute the maximum common radius for given centers."""
    n = len(pts)
    wall_dists = 0.5 - np.abs(pts)
    min_wall = np.min(wall_dists, axis=1)
    wall_radius = np.min(min_wall)
    
    if n < 2:
        return wall_radius
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    sq_dists = np.sum(diff ** 2, axis=2)
    iu = np.triu_indices(n, k=1)
    pair_dists = np.sqrt(sq_dists[iu])
    min_pair = np.min(pair_dists)
    pair_radius = min_pair / 2.0
    
    return min(wall_radius, pair_radius)

def _soft_min_radius_and_grad(pts, beta):
    """
    Compute soft-min radius and its gradient with respect to pts.
    
    The soft-min is: -T * log(sum(exp(-r_i / T))) where T = 1/beta
    This equals: -(1/beta) * logsumexp(-beta * r_i)
    
    r_i are the candidate radii (wall distances and half pairwise distances).
    """
    n = len(pts)
    T = 1.0 / beta
    
    # Wall distances: for each point, min over 3 coords of (0.5 - |x_k|)
    # We need to compute soft-min over ALL candidate radii (walls + pairs)
    # For efficiency, we'll compute the log-sum-exp directly.
    
    # Wall contribution: for point i, wall radius = min(0.5-|x_ix|, 0.5-|x_iy|, 0.5-|x_iz|)
    # But we want soft-min over ALL candidates, so we need to combine.
    
    # Let's compute all candidate radii and their gradients.
    
    # Wall radii: w_{i,k} = 0.5 - |x_{i,k}| for k in {0,1,2}
    # d w_{i,k} / d x_{i,k} = -sign(x_{i,k})
    
    # Pair radii: p_{i,j} = 0.5 * ||x_i - x_j|| for i < j
    # d p_{i,j} / d x_i = 0.5 * (x_i - x_j) / ||x_i - x_j||
    # d p_{i,j} / d x_j = 0.5 * (x_j - x_i) / ||x_i - x_j||
    
    # We'll compute the soft-min using log-sum-exp for numerical stability.
    # Let's build the list of all radii and compute the weighted sum.
    
    # For memory efficiency with n=100, we have 300 wall radii and 4950 pair radii.
    # Total ~5250 radii. This is manageable.
    
    # Compute wall radii
    wall_radii = 0.5 - np.abs(pts)  # shape (n, 3)
    wall_radii_flat = wall_radii.ravel()  # shape (3n,)
    
    # Compute pair radii
    iu = np.triu_indices(n, k=1)
    diff_ij = pts[iu[0]] - pts[iu[1]]  # shape (m, 3)
    pair_dists = np.sqrt(np.sum(diff_ij ** 2, axis=1))  # shape (m,)
    pair_radii = 0.5 * pair_dists  # shape (m,)
    
    # Combine all radii
    all_radii = np.concatenate([wall_radii_flat, pair_radii])  # shape (3n + m,)
    
    # Compute soft-min: -T * log(sum(exp(-all_radii / T)))
    # = -T * logsumexp(-all_radii / T)
    
    # For numerical stability:
    # logsumexp(z) = max(z) + log(sum(exp(z - max(z))))
    z = -all_radii / T
    z_max = np.max(z)
    logsumexp_val = z_max + np.log(np.sum(np.exp(z - z_max)))
    soft_min_val = -T * logsumexp_val
    
    # Compute gradient
    # d(soft_min)/d(x) = -T * d(logsumexp(z))/d(x)
    # = -T * sum_j (exp(z_j - z_max) / sum_k(exp(z_k - z_max))) * dz_j/dx
    # = -T * sum_j w_j * dz_j/dx where w_j = exp(z_j - z_max) / sum(exp(z_k - z_max))
    
    # First compute weights
    exp_vals = np.exp(z - z_max)
    sum_exp = np.sum(exp_vals)
    weights = exp_vals / sum_exp  # shape (3n + m,)
    
    # Now compute gradient contribution from each radius
    grad = np.zeros_like(pts)
    
    # Wall contributions
    # w_{i,k} = 0.5 - |x_{i,k}|, dw/dx_{i,k} = -sign(x_{i,k})
    # dz/dx = -(1/T) * dw/dx = -(1/T) * (-sign(x)) = sign(x)/T
    # d(soft_min)/dx = -T * sum w_j * dz_j/dx = -T * sum w_j * (sign(x_{i,k})/T) = -sum w_j * sign(x_{i,k})
    
    # For each point i and coordinate k, the weight is weights[3*i + k]
    wall_weights = weights[:3*n].reshape(n, 3)
    signs = np.sign(pts)
    # Handle x=0: sign(0)=0, which is fine (subgradient)
    grad_wall = -wall_weights * signs
    grad += grad_wall
    
    # Pair contributions
    # p_{i,j} = 0.5 * ||x_i - x_j||
    # dp/dx_i = 0.5 * (x_i - x_j) / ||x_i - x_j||
    # dz/dx_i = -(1/T) * dp/dx_i
    # d(soft_min)/dx_i = -T * w * dz/dx_i = -T * w * (-(1/T) * 0.5 * (x_i-x_j)/||x_i-x_j||) = w * 0.5 * (x_i-x_j)/||x_i-x_j||
    
    # For each pair (i, j) with i < j, weight is weights[3n + pair_idx]
    pair_weights = weights[3*n:]
    
    # For each pair, add to grad[i] and grad[j]
    for idx in range(len(pair_weights)):
        i, j = iu[0][idx], iu[1][idx]
        d = diff_ij[idx]
        dist = pair_dists[idx]
        if dist < 1e-15:
            continue
        w = pair_weights[idx]
        unit_vec = d / dist  # (x_i - x_j) / ||x_i - x_j||
        # Contribution to x_i: w * 0.5 * unit_vec
        # Contribution to x_j: w * 0.5 * (-unit_vec)
        grad[i] += 0.5 * w * unit_vec
        grad[j] -= 0.5 * w * unit_vec
    
    return soft_min_val, grad

def _objective_and_grad(x_flat, beta):
    """Objective and gradient for L-BFGS-B. x_flat is 1D array of length 3N."""
    pts = x_flat.reshape(N, 3)
    val, grad = _soft_min_radius_and_grad(pts, beta)
    # We want to maximize soft_min, so minimize -soft_min
    return -val, -grad

def _generate_fcc_initial():
    """Generate initial configuration from FCC-inspired grid."""
    m = 5
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    wall_dists = 0.5 - np.abs(grid)
    min_wall = np.min(wall_dists, axis=1)
    sorted_indices = np.argsort(-min_wall)
    selected_indices = sorted_indices[:N]
    fcc_pts = grid[selected_indices].copy()
    
    return fcc_pts

def _high_precision_polish(pts, max_iters=3000, time_limit=15.0):
    """Final polish using exact radius computation."""
    start_time = time.time()
    best_pts = pts.copy()
    best_score = _compute_radius(best_pts)
    step = 0.001
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        improved = False
        for i in range(len(pts)):
            for d in range(3):
                for direction in [1, -1]:
                    new_pts = best_pts.copy()
                    new_pts[i, d] += direction * step
                    new_pts[i, d] = np.clip(new_pts[i, d], -0.5 + 1e-9, 0.5 - 1e-9)
                    
                    new_score = _compute_radius(new_pts)
                    if new_score > best_score + 1e-12:
                        best_pts = new_pts
                        best_score = new_score
                        improved = True
        
        if not improved:
            step *= 0.5
            if step < 1e-8:
                break
    
    return best_pts, best_score

def construct():
    start_time = time.time()
    total_time_limit = 145.0  # Safety margin below 150s
    
    # Initial grid
    initial_pts = _generate_fcc_initial()
    
    best_overall = initial_pts.copy()
    best_score = _compute_radius(best_overall)
    
    np.random.seed(42)
    
    # Progressive sharpening with L-BFGS-B
    # Start with low beta (smooth) and increase to high beta (sharp)
    beta_schedule = [5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0]
    
    current_pts = initial_pts.copy()
    
    for beta in beta_schedule:
        elapsed = time.time() - start_time
        if elapsed > total_time_limit - 30:
            break
        
        # Flatten and set up optimization
        x0 = current_pts.ravel()
        
        # Bounds: each coordinate in [-0.5+eps, 0.5-eps]
        eps = 1e-9
        bounds = [(-0.5 + eps, 0.5 - eps)] * (N * 3)
        
        # Time limit for this beta stage
        time_left = total_time_limit - elapsed - 25
        if time_left < 5:
            break
        
        # Run L-BFGS-B
        result = minimize(
            _objective_and_grad,
            x0,
            args=(beta,),
            method='L-BFGS-B',
            bounds=bounds,
            jac=True,
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 1000}
        )
        
        new_pts = result.x.reshape(N, 3)
        new_pts = np.clip(new_pts, -0.5 + eps, 0.5 - eps)
        
        # Evaluate exact score
        exact_score = _compute_radius(new_pts)
        
        if exact_score > best_score:
            best_score = exact_score
            best_overall = new_pts.copy()
        
        current_pts = new_pts
    
    # Try a few random restarts with gradient optimization
    n_restarts = 3
    for s in range(n_restarts):
        elapsed = time.time() - start_time
        if elapsed > total_time_limit - 25:
            break
        
        # Perturb from best
        pts = best_overall + np.random.standard_normal(best_overall.shape) * 0.02
        pts = np.clip(pts, -0.5 + 1e-9, 0.5 - 1e-9)
        
        # Use moderate beta for restart
        beta = 50.0
        x0 = pts.ravel()
        bounds = [(-0.5 + 1e-9, 0.5 - 1e-9)] * (N * 3)
        
        time_left = total_time_limit - elapsed - 15
        if time_left < 5:
            break
        
        result = minimize(
            _objective_and_grad,
            x0,
            args=(beta,),
            method='L-BFGS-B',
            bounds=bounds,
            jac=True,
            options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        new_pts = result.x.reshape(N, 3)
        new_pts = np.clip(new_pts, -0.5 + 1e-9, 0.5 - 1e-9)
        
        exact_score = _compute_radius(new_pts)
        if exact_score > best_score:
            best_score = exact_score
            best_overall = new_pts.copy()
    
    # Final high-precision polish
    elapsed = time.time() - start_time
    if elapsed < total_time_limit - 10:
        time_left = total_time_limit - elapsed - 5
        if time_left > 3:
            best_overall, best_score = _high_precision_polish(
                best_overall, max_iters=2000, time_limit=min(10.0, time_left)
            )
    
    # Ensure valid output
    best_overall = np.clip(best_overall, -0.5 + 1e-10, 0.5 - 1e-10)
    
    return best_overall
# EVOLVE-BLOCK-END
