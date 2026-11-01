"""
Packing 200 equal circles in a unit square.
Strategy:
1. Initialize with hexagonal lattice.
2. Global optimization using Log-Sum-Exp surrogate for max-min distance.
3. High-precision local polish focusing on active constraints.
"""

N = 200
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import time

def _compute_radius(c):
    """Compute the largest common radius for given centres."""
    n = c.shape[0]
    if n == 0:
        return 0.0
    dists_x = 0.5 - np.abs(c[:, 0])
    dists_y = 0.5 - np.abs(c[:, 1])
    min_boundary = np.minimum(np.min(dists_x), np.min(dists_y))
    dists = pdist(c, metric='euclidean')
    min_pair_dist = np.min(dists) if len(dists) > 0 else np.inf
    return min(min_pair_dist / 2.0, min_boundary)

def _soft_min_energy(params, beta=200.0):
    """
    Minimize the negative log-sum-exp of squared distances.
    This is a smooth approximation of -log(sum(exp(-beta * d_i^2))).
    Minimizing this maximizes the soft-min distance.
    We also add a wall penalty to keep points inside.
    """
    c = params.reshape(N, 2)
    x = c[:, 0]
    y = c[:, 1]
    
    # Pairwise squared distances
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d2 = dx**2 + dy**2
    np.fill_diagonal(d2, 1.0) # Avoid division by zero or log(0) in gradient if needed, though we only take upper triangle
    
    # Upper triangle indices for pairs
    iu = np.triu_indices(N, k=1)
    d2_pairs = d2[iu]
    
    # Log-Sum-Exp trick for stability: log(sum(exp(-beta * d2)))
    # We want to MINIMIZE -log(sum(exp(-beta * d2))) = log(1/sum(exp(-beta * d2)))
    # Actually, maximizing min d is approximated by minimizing -soft_min.
    # soft_min(d) = -1/beta * log(sum(exp(-beta * d^2)))
    # So we minimize F = -soft_min = 1/beta * log(sum(exp(-beta * d^2)))
    
    # For numerical stability, use max trick:
    # log(sum(exp(-beta * d2))) = log(sum(exp(-beta * (d2 - d2_max)))) + (-beta * d2_max)
    # d2_max is the maximum d2? No, for sum of exp(-x), the dominant term is the smallest x (largest distance? No, smallest d2 -> largest exp).
    # We want to spread points, so we want to penalize small d2.
    # The term exp(-beta * d2) is large when d2 is small.
    # So sum(exp(-beta * d2)) is dominated by the smallest d2.
    # Let d2_min_val = np.min(d2_pairs)
    # log_sum = d2_min_val * (-beta) + log(sum(exp(-beta * (d2 - d2_min_val))))
    
    # Gradient of log(sum(exp(-beta * d2_i))) wrt d2_i is:
    # d/dx log(sum exp(-beta d2)) = sum(exp(-beta d2) * (-beta) * 2 * dx_vec) / sum(exp(-beta d2))
    
    # Let's compute weights w_i = exp(-beta * d2_i)
    # To avoid under/overflow, shift by min d2 (which makes the largest exp 1)
    d2_shifted = d2_pairs - np.min(d2_pairs)
    weights = np.exp(-beta * d2_shifted)
    
    # Soft min value (up to constant)
    soft_min_val = -np.log(np.sum(weights)) / beta # This is negative of log sum... wait.
    # Let E = 1/beta * log(sum(exp(-beta * d2)))
    # grad E wrt d2_i = 1/beta * 1/sum(exp) * exp(-beta d2_i) * (-beta) = -weights_i / sum(weights)
    # So grad wrt d2_i is negative probability.
    
    # We need grad wrt positions.
    # d2_i = |c_i - c_j|^2
    # grad d2_i wrt c_i = 2(c_i - c_j)
    
    # Let p_i = weights_i / sum(weights)
    p = weights / np.sum(weights)
    
    # Energy = 1/beta * log_sum_exp... 
    # Actually, let's just use the gradient of the energy directly.
    # E = 1/beta * log(sum(exp(-beta * d2_k)))
    # dE/dc_i = sum_k p_k * dE/d(d2_k) * d(d2_k)/dc_i
    # dE/d(d2_k) = -1
    # So dE/dc_i = -sum_{k: i in k} p_k * 2(c_i - c_j_k)
    
    grad_x = np.zeros(N)
    grad_y = np.zeros(N)
    
    # We need to map p back to pairs
    # iu contains (i, j) with i < j
    i_idx, j_idx = iu
    vec_x = x[i_idx] - x[j_idx]
    vec_y = y[i_idx] - y[j_idx]
    
    # Contribution to grad for each pair k with weight p_k
    # dE/d(c_i_x) += -p_k * 2 * vec_x_k
    # dE/d(c_j_x) += -p_k * 2 * (-vec_x_k) = +p_k * 2 * vec_x_k
    
    grad_x[i_idx] -= 2.0 * p * vec_x
    grad_x[j_idx] += 2.0 * p * vec_x
    
    grad_y[i_idx] -= 2.0 * p * vec_y
    grad_y[j_idx] += 2.0 * p * vec_y
    
    # Wall Penalty
    # We want to penalize getting close to walls.
    # Smooth penalty: 1/d^2 or exp(-d^2)?
    # Let's use a simple quadratic penalty for boundaries to keep it smooth.
    # P_wall = sum (1/d_i^2) where d_i is dist to nearest wall.
    # This was in the original. It works well.
    eps = 1e-6
    dx_w_left = np.clip(0.5 - x, eps, None)
    dx_w_right = np.clip(0.5 + x, eps, None)
    dy_w_down = np.clip(0.5 - y, eps, None)
    dy_w_up = np.clip(0.5 + y, eps, None)
    
    E_wall = (1.0/dx_w_left**2 + 1.0/dx_w_right**2 + 1.0/dy_w_down**2 + 1.0/dy_w_up**2).sum()
    
    # We need to combine the soft-min energy and wall energy.
    # The soft-min energy scale is roughly 1/beta * log(N). 
    # The wall energy scale can be large if close.
    # Let's normalize or just add them.
    # To make the soft-min dominate for global spread, and wall for local containment.
    
    E_total = E_wall + 1.0 * (np.log(np.sum(weights)) / beta) # Note: log(sum(exp(-beta d2))) is negative large if d2 small?
    # Wait. exp(-beta * d2). If d2 is small, exp is large. Log is large positive.
    # We want to MINIMIZE energy.
    # If points are close, d2 is small, exp(-beta d2) is large, log is large positive.
    # So E_pair = log(sum(exp(-beta d2))) / beta is LARGE when points are close.
    # So minimizing E_pair moves points apart. Correct.
    
    # Gradient of wall part
    grad_x += -2.0 / dx_w_left**3 + 2.0 / dx_w_right**3
    grad_y += -2.0 / dy_w_down**3 + 2.0 / dy_w_up**3
    
    total_grad = np.concatenate([grad_x, grad_y])
    return E_total, total_grad

def _initialize_hexagonal():
    """Initialize with hexagonal pattern."""
    best_c = None
    best_r = 0
    
    # Try different densities
    for d in np.linspace(0.055, 0.085, 20):
        points = []
        row_spacing = d * np.sqrt(3) / 2
        x_offset = d / 2
        
        y = -0.5 + d / 2
        row = 0
        while y <= 0.5 - d / 2:
            x = -0.5 + d / 2
            if row % 2 == 1:
                x += x_offset
            while x <= 0.5 - d / 2:
                points.append([x, y])
                x += d
            y += row_spacing
            row += 1
        
        points = np.array(points)
        if len(points) >= N:
            c = points[:N].copy()
            c = np.clip(c, -0.495, 0.495)
            r = _compute_radius(c)
            if r > best_r:
                best_r = r
                best_c = c.copy()
    
    if best_c is not None:
        return best_c
    
    # Fallback grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()

def _optimize_energy(c0, time_budget=20.0, beta=100.0):
    """Run L-BFGS-B with Log-Sum-Exp energy."""
    start = time.time()
    bounds = [(-0.499, 0.499)] * (2 * N)
    
    try:
        # Wrap to handle beta
        def fun_and_grad(params):
            return _soft_min_energy(params, beta=beta)
            
        res = minimize(
            fun_and_grad,
            c0.flatten(),
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
            options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 1000}
        )
        c_opt = np.clip(res.x.reshape(N, 2), -0.499, 0.499)
        return c_opt
    except Exception:
        return c0

def _local_polish(c, time_limit=30.0):
    """
    Fast local polish using active-constraint gradient ascent on radius.
    Uses a smooth penalty-based approach for speed.
    """
    start_time = time.time()
    c = c.copy()
    current_r = _compute_radius(c)
    
    # Use L-BFGS-B on a smooth surrogate that maximizes the minimum distance
    # Focus on the most constrained points
    
    def _polish_energy(params, beta=200.0):
        """Energy that penalizes violations of the current best radius."""
        c_ = params.reshape(N, 2)
        x = c_[:, 0]
        y = c_[:, 1]
        
        # Pairwise distances
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d2 = dx**2 + dy**2
        np.fill_diagonal(d2, 1.0)
        
        iu = np.triu_indices(N, k=1)
        d2_pairs = d2[iu]
        d_pairs = np.sqrt(d2_pairs)
        
        # Boundary distances
        bdists = np.minimum(
            np.minimum(0.5 - x, 0.5 + x),
            np.minimum(0.5 - y, 0.5 + y)
        )
        
        # We want to maximize min(d_pairs/2, bdists)
        # Use log-sum-exp to get a smooth approximation of the negative min
        # E = -soft_min = (1/beta) * log(sum(exp(-beta * d)))
        
        # For pairs
        d2_shifted = d2_pairs - np.min(d2_pairs)
        w_pairs = np.exp(-beta * d2_shifted)
        E_pairs = np.log(np.sum(w_pairs)) / beta
        
        # For boundaries
        bd_shifted = bdists - np.min(bdists)
        w_bounds = np.exp(-beta * bd_shifted)
        E_bounds = np.log(np.sum(w_bounds)) / beta
        
        E = E_pairs + E_bounds
        
        # Gradient
        p_pairs = w_pairs / np.sum(w_pairs)
        p_bounds = w_bounds / np.sum(w_bounds)
        
        grad_x = np.zeros(N)
        grad_y = np.zeros(N)
        
        i_idx, j_idx = iu
        vec_x = x[i_idx] - x[j_idx]
        vec_y = y[i_idx] - y[j_idx]
        
        # dE/d(d2_k) = -1 (from the log-sum-exp form)
        # d(d2_k)/d(c_i) = 2*(c_i - c_j)
        grad_x[i_idx] -= 2.0 * p_pairs * vec_x
        grad_x[j_idx] += 2.0 * p_pairs * vec_x
        grad_y[i_idx] -= 2.0 * p_pairs * vec_y
        grad_y[j_idx] += 2.0 * p_pairs * vec_y
        
        # Boundary gradient
        # dE/d(bd_i) = -p_bounds_i
        # bd_i = min(0.5-x, 0.5+x, 0.5-y, 0.5+y)
        # Approximate with soft assignment
        d_left = 0.5 - x
        d_right = 0.5 + x
        d_down = 0.5 - y
        d_up = 0.5 + y
        
        # Soft assignment weights (which wall is closest)
        d_all = np.stack([d_left, d_right, d_down, d_up], axis=1)  # (N, 4)
        d_min = np.min(d_all, axis=1, keepdims=True)
        # exp(-beta * (d - d_min)) gives weight to closest walls
        wall_weights = np.exp(-beta * (d_all - d_min))  # (N, 4)
        wall_weights /= wall_weights.sum(axis=1, keepdims=True)
        
        # Gradient contribution from boundary
        # d(bd_i)/d(x_i) for left wall (0.5-x): -1
        # d(bd_i)/d(x_i) for right wall (0.5+x): +1
        # d(bd_i)/d(y_i) for down wall (0.5-y): -1
        # d(bd_i)/d(y_i) for up wall (0.5+y): +1
        grad_x += p_bounds * (-wall_weights[:, 0] + wall_weights[:, 1])
        grad_y += p_bounds * (-wall_weights[:, 2] + wall_weights[:, 3])
        
        total_grad = np.concatenate([grad_x, grad_y])
        return E, total_grad
    
    # Run L-BFGS-B for polishing
    bounds = [(-0.4999, 0.4999)] * (2 * N)
    
    try:
        res = minimize(
            _polish_energy,
            c.flatten(),
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
            options={'maxiter': 50, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 200}
        )
        c_opt = np.clip(res.x.reshape(N, 2), -0.4999, 0.4999)
        r_opt = _compute_radius(c_opt)
        if r_opt > current_r:
            c = c_opt
            current_r = r_opt
    except Exception:
        pass
    
    # Final micro-adjustment: try to nudge active points
    if time.time() - start_time < time_limit:
        for _ in range(50):
            if time.time() - start_time > time_limit - 1:
                break
            
            diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
            dist_matrix = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dist_matrix, np.inf)
            
            bdists = np.minimum(
                np.minimum(0.5 - c[:, 0], 0.5 + c[:, 0]),
                np.minimum(0.5 - c[:, 1], 0.5 + c[:, 1])
            )
            
            min_pair_dist = np.min(dist_matrix)
            min_boundary_dist = np.min(bdists)
            target_r = min(min_pair_dist / 2.0, min_boundary_dist)
            
            if target_r > current_r + 1e-12:
                current_r = target_r
            
            # Find active constraints
            tol = 1e-8
            active_pairs = np.argwhere(dist_matrix < min_pair_dist + tol)
            active_bounds = np.where(bdists < min_boundary_dist + tol)[0]
            
            if len(active_pairs) == 0 and len(active_bounds) == 0:
                break
            
            # Build force vectors
            forces = np.zeros((N, 2))
            
            for i, j in active_pairs:
                if i == j:
                    continue
                vec = c[i] - c[j]
                norm = np.linalg.norm(vec)
                if norm > 1e-10:
                    unit_vec = vec / norm
                    forces[i] += unit_vec
                    forces[j] -= unit_vec
            
            for i in active_bounds:
                d_left = 0.5 - c[i, 0]
                d_right = c[i, 0] + 0.5
                d_down = 0.5 - c[i, 1]
                d_up = c[i, 1] + 0.5
                
                min_d = min(d_left, d_right, d_down, d_up)
                if d_left == min_d:
                    forces[i, 0] += 1.0
                elif d_right == min_d:
                    forces[i, 0] -= 1.0
                elif d_down == min_d:
                    forces[i, 1] += 1.0
                else:
                    forces[i, 1] -= 1.0
            
            # Try moving each active point
            moved = False
            for step in [1e-5, 5e-6, 1e-6, 5e-7, 1e-7]:
                if moved:
                    break
                for i in range(N):
                    if np.linalg.norm(forces[i]) > 1e-10:
                        direction = forces[i] / np.linalg.norm(forces[i])
                        new_pos = c[i] + direction * step
                        if np.all(np.abs(new_pos) < 0.49999):
                            old_pos = c[i].copy()
                            c[i] = new_pos
                            new_r = _compute_radius(c)
                            if new_r > current_r:
                                current_r = new_r
                                moved = True
                            else:
                                c[i] = old_pos
    
    return c, current_r

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    best_c = _initialize_hexagonal()
    best_r = _compute_radius(best_c)
    
    # Run global optimization with multiple beta values
    for beta in [30.0, 50.0, 80.0]:
        if time.time() > deadline - 60:
            break
        c_opt = _optimize_energy(best_c, time_budget=20.0, beta=beta)
        r_opt = _compute_radius(c_opt)
        
        if r_opt > best_r:
            best_r = r_opt
            best_c = c_opt.copy()
    
    # Local polish - use most of the remaining time
    remaining = deadline - time.time() - 3
    if remaining > 10:
        c_polished, r_polished = _local_polish(best_c, time_limit=min(remaining, 80.0))
        if r_polished > best_r:
            best_r = r_polished
            best_c = c_polished.copy()
    
    # Second polish pass if time allows
    remaining = deadline - time.time() - 2
    if remaining > 10:
        c_polished2, r_polished2 = _local_polish(best_c, time_limit=min(remaining, 40.0))
        if r_polished2 > best_r:
            best_r = r_polished2
            best_c = c_polished2.copy()
    
    # Final safety check
    best_c = np.clip(best_c, -0.4999, 0.4999)
    
    return best_c
