"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 110.0 # Safety margin for 120s limit

# EVOLVE-BLOCK-START

def _compute_radius(points):
    """
    Compute the maximum possible radius for the given points.
    r = min( min_{i<j} |p_i - p_j| / 2, min_i min_k (0.5 - |p_ik|) )
    """
    if len(points) == 0:
        return 0.0
    
    # Distance to boundaries
    # Points are in [-0.5, 0.5]^3
    # Distance to wall for point p is min(0.5 - p_k) for k in 0,1,2
    # Actually, distance to wall is 0.5 - |p_k|? No.
    # If p_k is in [-0.5, 0.5], distance to boundary at 0.5 is 0.5 - p_k.
    # Distance to boundary at -0.5 is p_k - (-0.5) = p_k + 0.5.
    # So min distance to any wall for coordinate k is min(0.5 - p_k, p_k + 0.5) = 0.5 - |p_k|.
    # Let's verify: if p_k = 0, dist = 0.5. Correct.
    # if p_k = 0.4, dist = 0.1. Correct.
    
    abs_pts = np.abs(points)
    wall_dists = 0.5 - np.max(abs_pts, axis=1)
    min_wall_dist = np.min(wall_dists)
    
    # Pairwise distances
    # Use broadcasting for efficiency, N=50 is small
    # diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    # dists = np.linalg.norm(diff, axis=2)
    # np.fill_diagonal(dists, np.inf)
    # min_pair_dist = np.min(dists)
    
    # More memory efficient for larger N, but N=50 is fine with broadcasting
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    r = min(min_pair_dist / 2.0, min_wall_dist)
    return r

def _smooth_min_radius(points, beta=100.0):
    """
    Smooth surrogate for the minimum radius using log-sum-exp.
    We want to maximize min(d_ij/2, wall_i).
    Maximizing min is equivalent to minimizing -min.
    -min(x) = - (1/beta) * log(sum(exp(-beta*x)))  for large beta.
    Wait, standard soft-min: min(x) ~ (1/beta) log(sum(exp(-beta x))).
    So we want to minimize f(x) = -min(x) ~ -(1/beta) log(sum(exp(-beta x))).
    
    Let's define the objective to minimize as:
    L(x) = - (1/beta) log( sum_{pairs} exp(-beta * d_ij/2) + sum_{i,k} exp(-beta * wall_ik) )
    
    This is a smooth approximation of -min_radius.
    Minimizing L is maximizing the smooth min radius.
    """
    # Pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # We only need upper triangle to avoid double counting, but sum exp is fine
    # d/2
    d_half = dists / 2.0
    
    # Wall distances
    abs_pts = np.abs(points)
    wall_dists = 0.5 - abs_pts # shape (N, 3)
    
    # Combine all constraints
    # Flatten
    d_flat = d_half.flatten()
    w_flat = wall_dists.flatten()
    
    all_dists = np.concatenate([d_flat, w_flat])
    
    # Handle inf
    all_dists = np.where(np.isinf(all_dists), 1e10, all_dists)
    
    # Log-sum-exp trick for stability
    # min_val = np.min(all_dists)
    # log_sum_exp = min_val + np.log(np.sum(np.exp(-beta * (all_dists - min_val))))
    # result = -log_sum_exp / beta
    
    # Simpler:
    scaled = -beta * all_dists
    max_scaled = np.max(scaled)
    log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    
    return -log_sum_exp / beta

def _gradient_smooth_min_radius(points, beta=100.0):
    """
    Gradient of the smooth min radius objective.
    f(x) = - (1/beta) log( sum exp(-beta d_k) )
    grad f = (1/beta) * sum( exp(-beta d_k) * (-beta grad d_k) ) / sum exp(-beta d_k)
           = - sum( w_k * grad d_k ) / sum w_k
    where w_k = exp(-beta d_k)
    """
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    d_half = dists / 2.0
    
    abs_pts = np.abs(points)
    wall_dists = 0.5 - abs_pts
    
    # Flatten
    d_flat = d_half.flatten()
    w_flat = wall_dists.flatten()
    
    all_dists = np.concatenate([d_flat, w_flat])
    all_dists = np.where(np.isinf(all_dists), 1e10, all_dists)
    
    # Weights
    scaled = -beta * all_dists
    max_scaled = np.max(scaled)
    weights = np.exp(scaled - max_scaled)
    sum_weights = np.sum(weights)
    
    # Normalized weights
    norm_weights = weights / sum_weights
    
    # Gradients
    # Pairwise: grad d_ij/2 = (p_i - p_j) / (2 * |p_i - p_j|) * 1/2 ?
    # d_ij = |p_i - p_j|. d_half = d_ij / 2.
    # grad d_half = (p_i - p_j) / (2 * d_ij)
    # Note: sum over all i,j.
    
    # We need to map flat indices back to (i,j) or (i,k)
    
    n_pairs = N * N
    n_walls = N * 3
    
    grad = np.zeros_like(points)
    
    # Pairwise part
    # Reshape weights to (N, N)
    w_pairs = norm_weights[:n_pairs].reshape(N, N)
    
    # dists is (N, N)
    # diff is (N, N, 3)
    # grad d_half_ij = diff_ij / (2 * dists_ij)
    # But dists_diag is inf, so weight is 0 there (exp(-inf) = 0).
    # Need to avoid division by zero or inf issues.
    
    # Create mask for valid pairs (i != j)
    mask = np.eye(N, dtype=bool)
    
    # Compute gradient contribution
    # term_ij = w_ij * (p_i - p_j) / (2 * dists_ij)
    # grad_i = sum_j term_ij
    # grad_j = sum_i term_ji = sum_i w_ij (p_j - p_i) / (2 * dists_ij) = - sum_i w_ij (p_i - p_j) / (2 * dists_ij)
    
    # Using broadcasting:
    # diff (N, N, 3)
    # dists (N, N)
    # w_pairs (N, N)
    
    # Safe division
    dists_safe = np.where(dists > 1e-8, dists, 1.0)
    # Zero out diagonal in diff to avoid issues, though weight is 0
    diff_safe = np.where(np.expand_dims(mask, -1), diff, 0.0)
    
    grad_pairs = (w_pairs[:, :, np.newaxis] * diff_safe) / (2.0 * dists_safe[:, :, np.newaxis])
    
    # Sum over j for each i
    grad_from_pairs = np.sum(grad_pairs, axis=1)
    
    # Add to grad
    grad += grad_from_pairs
    
    # Wall part
    # w_walls = norm_weights[n_pairs:]
    # wall_dists = 0.5 - |p_ik|
    # grad wall_ik wrt p_ik = -sign(p_ik)
    # So grad p_ik = - sum_k w_ik * (-sign(p_ik)) = sum_k w_ik * sign(p_ik)
    
    w_walls = norm_weights[n_pairs:].reshape(N, 3)
    
    signs = np.sign(points)
    # Handle 0: sign(0)=0, which is fine (subgradient, any value in [-1,1] works, 0 is good)
    
    grad_from_walls = np.sum(w_walls * signs, axis=1)
    
    grad += grad_from_walls
    
    # The objective was -log_sum_exp / beta.
    # The gradient derived above is for f(x) = -min(x) approx.
    # Let's check sign.
    # We want to MINIMIZE f.
    # If points are too close, d is small, exp(-beta d) is large.
    # Gradient should push points apart.
    # If p_i is close to p_j, diff = p_i - p_j.
    # grad_i += w * diff / d. This pushes p_i in direction of p_i - p_j. Correct.
    # If p_i is close to wall (p_i > 0), wall_dist = 0.5 - p_i.
    # grad wall = -1.
    # grad p_i += w * (-1) * (-1) = w. Pushes p_i in positive direction?
    # Wait.
    # f = -log(sum exp(-beta d)).
    # df/dp = - (1/sum) sum exp(-beta d) * (-beta) * dd/dp
    #       = (beta/sum) sum exp(-beta d) * dd/dp
    #       = sum w_norm * dd/dp
    # My previous derivation had a sign error in the manual step but the code logic:
    # grad_pairs = w * diff / (2*d).
    # If p_i > p_j, diff > 0. grad_i > 0. p_i increases. Moves away from p_j. Correct.
    # Wall: dd/dp = -sign(p).
    # grad = sum w * (-sign(p)).
    # If p > 0, sign = 1. grad = -w. p decreases. Moves away from wall at 0.5. Correct.
    
    return grad

def construct():
    start_time = time.time()
    
    # Initializations
    # 1. Random
    # 2. Grid
    # 3. FCC-like (subset of FCC lattice)
    
    best_points = None
    best_radius = 0.0
    
    # Generate initial configurations
    inits = []
    
    # Random
    rng = np.random.default_rng(42)
    for _ in range(5):
        pts = rng.uniform(-0.4, 0.4, size=(N, 3))
        inits.append(pts)
        
    # Grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    inits.append(grid[:N])
    
    # FCC-like
    # FCC basis vectors: (0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)
    # Scale to fit in cube.
    # Simple FCC grid
    a = 0.5 # lattice constant
    fcc_pts = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                fcc_pts.append([i*a, j*a, k*a])
                fcc_pts.append([i*a + a/2, j*a + a/2, k*a])
                fcc_pts.append([i*a + a/2, j*a, k*a + a/2])
                fcc_pts.append([i*a, j*a + a/2, k*a + a/2])
    
    fcc_pts = np.array(fcc_pts)
    # Center it
    fcc_pts -= np.mean(fcc_pts, axis=0)
    # Scale to fit in [-0.5, 0.5]
    max_coord = np.max(np.abs(fcc_pts))
    if max_coord > 0:
        fcc_pts *= 0.45 / max_coord # Leave some margin
    
    # Select N points from FCC
    # Sort by distance from center to get compact cluster? Or just take first N?
    # Taking first N might be biased. Let's randomize selection from the FCC grid points
    # But we need exactly N.
    # Let's just take the first N after shuffling
    idx = rng.permutation(len(fcc_pts))[:N]
    inits.append(fcc_pts[idx])
    
    # Optimization Loop
    # We will run L-BFGS-B with annealing (noise injection) or just multiple restarts with L-BFGS-B.
    # Given time constraints, multiple restarts with L-BFGS-B on the smooth objective is robust.
    # We can also add a "simulated annealing" step by restarting from perturbed best solutions.
    
    # Strategy:
    # 1. For each init, run L-BFGS-B.
    # 2. Keep the best result.
    # 3. If time remains, perturb the best result and run again.
    
    for i, init in enumerate(inits):
        if time.time() - start_time > DEADLINE - 5:
            break
            
        # Run L-BFGS-B
        # Objective: minimize -smooth_min_radius
        # Gradient: gradient of -smooth_min_radius
        
        # We need to wrap the functions for L-BFGS-B
        # L-BFGS-B expects x as a flat array
        
        def objective_flat(x):
            pts = x.reshape(N, 3)
            # Add small penalty if out of bounds to guide back?
            # L-BFGS-B handles bounds.
            val = _smooth_min_radius(pts, beta=50.0)
            return val
            
        def gradient_flat(x):
            pts = x.reshape(N, 3)
            grad = _gradient_smooth_min_radius(pts, beta=50.0)
            return grad.flatten()
            
        # Bounds: [-0.5, 0.5] for all coordinates
        bounds = [(-0.5, 0.5)] * (N * 3)
        
        try:
            res = minimize(
                objective_flat, 
                init.flatten(), 
                jac=gradient_flat, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if res.success or res.fun < objective_flat(init.flatten()):
                curr_pts = res.x.reshape(N, 3)
                curr_rad = _compute_radius(curr_pts)
                
                if curr_rad > best_radius:
                    best_radius = curr_rad
                    best_points = curr_pts.copy()
                    
        except Exception as e:
            pass
            
    # Refinement Phase:
    # If we have time, do a "polish" step on the best points found.
    # Use a higher beta for the smooth objective to get closer to the true min.
    # Or use Nelder-Mead for a local polish without gradients.
    
    if best_points is not None and time.time() - start_time < DEADLINE - 10:
        
        def polish_objective(x):
            pts = x.reshape(N, 3)
            # Use higher beta
            return _smooth_min_radius(pts, beta=200.0)
            
        def polish_gradient(x):
            pts = x.reshape(N, 3)
            return _gradient_smooth_min_radius(pts, beta=200.0).flatten()
            
        try:
            res = minimize(
                polish_objective, 
                best_points.flatten(), 
                jac=polish_gradient, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if res.fun < polish_objective(best_points.flatten()):
                curr_pts = res.x.reshape(N, 3)
                curr_rad = _compute_radius(curr_pts)
                if curr_rad > best_radius:
                    best_radius = curr_rad
                    best_points = curr_pts.copy()
                    
        except Exception as e:
            pass
            
    # Final check and return
    if best_points is None:
        # Fallback to grid
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N]
        
    return best_points

# EVOLVE-BLOCK-END
