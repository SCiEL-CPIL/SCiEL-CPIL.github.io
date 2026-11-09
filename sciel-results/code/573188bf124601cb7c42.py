"""
Strategy: Smooth Surrogate Optimization with Adaptive Annealing
Instead of MD or non-smooth L-BFGS, we optimize a differentiable surrogate for the minimum clearance.
We use the log-sum-exp function to approximate the minimum of the pairwise distances and boundary clearances.
By annealing the temperature parameter beta (low to high), we start with a smooth, global-ish landscape 
that is easy to optimize, and gradually sharpen it to focus on the actual minimum constraint.
This allows L-BFGS-B to navigate the landscape effectively without getting stuck in the non-differentiable 
creases of the raw min function.
"""

import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _radius_score(centers):
    """Compute the maximum common radius for given centers in [-0.5, 0.5]^2."""
    if len(centers) != N:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(centers < -0.5 - 1e-9) or np.any(centers > 0.5 + 1e-9):
        return 0.0
    
    centers = np.clip(centers, -0.5, 0.5)
    
    # Boundary distance
    dist_to_boundary = np.min(np.stack([
        centers[:, 0] + 0.5,
        0.5 - centers[:, 0],
        centers[:, 1] + 0.5,
        0.5 - centers[:, 1]
    ]), axis=0)
    min_boundary = np.min(dist_to_boundary)
    
    # Pairwise distances
    # Efficiently compute pairwise distances
    # For N=50, we can use broadcasting
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dists, np.inf)
    min_pairwise = np.min(dists) / 2
    
    return min(min_boundary, min_pairwise)

def _smooth_min_objective(centers, beta):
    """
    Objective function to minimize: -smooth_min(clearances).
    clearances = [d_ij/2 for all i<j] + [dist_to_boundary for all i]
    We want to maximize the minimum of these.
    """
    centers = np.clip(centers, -0.5, 0.5)
    
    # 1. Boundary clearances
    # c_i must be in [-0.5+r, 0.5-r]
    # Clearance to boundary is min(x+0.5, 0.5-x, y+0.5, 0.5-y)
    # We treat these as 4N terms.
    b_x_pos = centers[:, 0] + 0.5
    b_x_neg = 0.5 - centers[:, 0]
    b_y_pos = centers[:, 1] + 0.5
    b_y_neg = 0.5 - centers[:, 1]
    
    boundary_clearances = np.concatenate([b_x_pos, b_x_neg, b_y_pos, b_y_neg])
    
    # 2. Pairwise clearances
    # d_ij / 2
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    # Mask diagonal
    mask = ~np.eye(N, dtype=bool)
    # We only need upper triangle to avoid duplicates, but for logsumexp duplicates don't hurt much, 
    # just change the effective beta. Let's take unique pairs.
    iu = np.triu_indices(N, k=1)
    pair_clearances = dists[iu] / 2.0
    
    all_clearances = np.concatenate([boundary_clearances, pair_clearances])
    
    # Soft min: min(x) ~= (1/beta) * log(sum(exp(-beta * x)))
    # We want to maximize min, so minimize -soft_min.
    # To avoid overflow, subtract max.
    # Note: if beta is very high, this becomes numerically unstable if values are large, but here they are small (~0.1).
    # scaled = -beta * all_clearances
    # max_scaled = np.max(scaled)
    # log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    # soft_min = log_sum_exp / beta
    
    # Actually, simpler:
    # soft_min = - (1/beta) * log( sum( exp(-beta * x) ) )
    # Let's compute -soft_min = (1/beta) * log( sum( exp(-beta * x) ) )
    
    # Numerical stability trick for logsumexp
    # logsumexp(a) = max(a) + log(sum(exp(a - max(a))))
    # Here a_i = -beta * x_i
    a = -beta * all_clearances
    max_a = np.max(a)
    # If max_a is -inf (shouldn't happen), handle
    if not np.isfinite(max_a):
        return 0.0
        
    log_sum = max_a + np.log(np.sum(np.exp(a - max_a)))
    obj = log_sum / beta
    
    return obj

def _optimize_smooth(centers, time_budget=10.0, start_beta=50.0, end_beta=5000.0):
    """
    Run L-BFGS-B with annealing beta.
    """
    start_time = time.time()
    centers = centers.copy()
    
    # We will run L-BFGS in stages with increasing beta.
    # Stage 1: Low beta (smooth)
    # Stage 2: Medium beta
    # Stage 3: High beta (sharp)
    
    betas = [start_beta, 200.0, 1000.0, 3000.0, end_beta]
    
    current_centers = centers
    
    for beta in betas:
        if time.time() > time_budget:
            break
            
        def _obj(x):
            return _smooth_min_objective(x, beta)
        
        try:
            res = minimize(
                _obj,
                current_centers,
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 2),
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res.x is not None and np.all(np.isfinite(res.x)):
                # Check if it improved the actual score
                r_old = _radius_score(current_centers)
                r_new = _radius_score(res.x)
                if r_new >= r_old - 1e-9:
                    current_centers = res.x
        except Exception:
            pass
            
    return current_centers

def _construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin
    
    rng = np.random.default_rng(42)
    
    best_centers = None
    best_radius = -np.inf
    
    # Initial configurations
    inits = []
    
    # 1. Grid
    m, n = 7, 8
    points = []
    for i in range(n):
        for j in range(m):
            if len(points) >= N:
                break
            x = (j + 0.5) / m - 0.5
            y = (i + 0.5) / n - 0.5
            points.append([x, y])
    inits.append(np.array(points[:N]))
    
    # 2. Hex
    points = []
    row_height = 0.09
    col_width = 0.1
    offset = col_width / 2
    for row in range(8):
        y = -0.5 + 0.05 + row * row_height
        for col in range(7):
            if len(points) >= N:
                break
            if row % 2 == 0:
                x = -0.5 + 0.05 + col * col_width
            else:
                x = -0.5 + 0.05 + offset + col * col_width
            points.append([x, y])
    inits.append(np.array(points[:N]))
    
    # 3. Random
    for _ in range(5):
        inits.append(rng.uniform(-0.4, 0.4, size=(N, 2)))
        
    # 4. Best known packing approximation? 
    # We don't have the exact coordinates, but we can try to use a high-quality random or grid.
    # Let's add a "quasi-random" (Halton-like) or just more randoms.
    for _ in range(5):
        inits.append(rng.uniform(-0.4, 0.4, size=(N, 2)))
        
    # Optimize each init
    for i, init in enumerate(inits):
        if time.time() > deadline - 10.0:
            break
            
        # Allocate time based on remaining budget
        remaining = deadline - time.time()
        if remaining < 1.0:
            break
            
        # Give more time to earlier (likely better) inits? Or equal?
        # Let's do equal but capped.
        time_per_init = min(10.0, remaining / max(1, len(inits) - i))
        
        optimized = _optimize_smooth(init, time_budget=time_per_init)
        
        r = _radius_score(optimized)
        if r > best_radius:
            best_radius = r
            best_centers = optimized.copy()
            
    # Final polish on the best result with very high beta
    if best_centers is not None and time.time() < deadline - 2.0:
        # Try to push it further with a very high beta, maybe starting from best
        # Also try small perturbations
        for _ in range(3):
            if time.time() > deadline - 1.0:
                break
            # Perturb
            perturbed = best_centers + rng.normal(0, 0.005, size=(N, 2))
            perturbed = np.clip(perturbed, -0.49, 0.49)
            
            # High beta optimization
            polished = _optimize_smooth(perturbed, time_budget=min(5.0, deadline - time.time() - 1.0), start_beta=500.0, end_beta=10000.0)
            
            r = _radius_score(polished)
            if r > best_radius:
                best_radius = r
                best_centers = polished.copy()
                
    if best_centers is None:
        # Fallback
        m, n = 7, 8
        points = []
        for i in range(n):
            for j in range(m):
                if len(points) >= N:
                    break
                x = (j + 0.5) / m - 0.5
                y = (i + 0.5) / n - 0.5
                points.append([x, y])
        best_centers = np.array(points[:N])
    else:
        best_centers = np.clip(best_centers, -0.5, 0.5)
        
    return best_centers

def construct():
    return _construct()
