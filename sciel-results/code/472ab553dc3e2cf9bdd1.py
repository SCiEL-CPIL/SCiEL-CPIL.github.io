import numpy as np
import time
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

N = 50

def compute_radius(positions):
    """
    Compute the maximum radius for the given positions.
    Returns radius. If invalid, returns a large negative value to act as penalty.
    """
    # positions shape (N, 2)
    
    # Boundary distances
    max_abs = np.max(np.abs(positions), axis=1)
    bound_dists = 0.5 - max_abs
    
    # If any point is outside, return penalty
    if np.any(bound_dists < 0):
        return -1.0 - np.max(-bound_dists)
        
    min_bound = np.min(bound_dists)
    
    # Pairwise distance constraint using cdist (C-optimized)
    dists = cdist(positions, positions)
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    
    r = min(min_bound, min_dist / 2.0)
    
    return r

def construct():
    start_time = time.time()
    deadline = start_time + 115.0 # 5s margin
    
    # 1. Initialization: Hexagonal Grid
    s = 1.0
    points = []
    for j in range(-15, 16):
        for i in range(-15, 16):
            x = i * s + (j % 2) * s / 2.0
            y = j * s * np.sqrt(3.0) / 2.0
            points.append([x, y])
            
    points = np.array(points)
    dists_to_center = np.sum(points**2, axis=1)
    indices = np.argsort(dists_to_center)[:N]
    positions = points[indices].copy()
    
    max_abs = np.max(np.abs(positions))
    if max_abs > 0:
        scale = 0.5 / max_abs
        positions = positions * scale
        
    r_init_val = compute_radius(positions)
    noise_std = max(0.001, r_init_val * 0.01)
    positions += np.random.normal(0, noise_std, size=positions.shape)
    positions = np.clip(positions, -0.4999, 0.4999)
    
    current_radius = compute_radius(positions)
    best_radius = current_radius
    best_positions = positions.copy()
    
    # 2. Simulated Annealing
    T = 0.001
    T_end = 1e-7
    alpha = 0.999
    step_size = 0.01
    min_step_size = 0.0001
    max_step_size = 0.05
    
    accept_count = 0
    total_count = 0
    
    sa_deadline = start_time + 80.0 # Leave more time for polishing
    
    while time.time() < sa_deadline:
        for _ in range(100):
            if time.time() >= sa_deadline:
                break
                
            total_count += 1
            idx = np.random.randint(N)
            
            if total_count % 100 == 0:
                accept_rate = accept_count / max(1, total_count)
                if accept_rate < 0.2:
                    step_size *= 1.2
                elif accept_rate > 0.8:
                    step_size *= 0.8
                step_size = np.clip(step_size, min_step_size, max_step_size)
                accept_count = 0
                total_count = 0
                
            delta = np.random.normal(0, step_size, size=2)
            new_pos = positions[idx].copy() + delta
            
            old_pos = positions[idx].copy()
            positions[idx] = new_pos
            new_radius = compute_radius(positions)
            
            if new_radius > current_radius:
                current_radius = new_radius
                accept_count += 1
                if new_radius > best_radius:
                    best_radius = new_radius
                    best_positions = positions.copy()
            else:
                delta_r = new_radius - current_radius
                if delta_r > -T * 10:
                    prob = np.exp(delta_r / T)
                    if np.random.random() < prob:
                        current_radius = new_radius
                        accept_count += 1
                    else:
                        positions[idx] = old_pos
                else:
                    positions[idx] = old_pos
                    
        T *= alpha
        if T < T_end:
            T = T_end
            
    # 3. Polish Phase: Gradient-based optimization using soft-min surrogate
    def polish(pos_in):
        pos = pos_in.copy()
        current_r = compute_radius(pos)
        
        # Objective function: minimize -soft_min of all constraints
        def objective(p):
            pp = p.reshape(N, 2)
            
            # Boundary distances
            max_abs = np.max(np.abs(pp), axis=1)
            bound_dists = 0.5 - max_abs
            
            # If any point is outside, return large penalty
            if np.any(bound_dists < 0):
                return 1000.0
            
            # Pairwise distances using cdist
            dists = cdist(pp, pp)
            np.fill_diagonal(dists, np.inf)
            half_dists = dists / 2.0
            
            # Combine boundary and pairwise distances
            # Only use upper triangle to avoid double counting
            iu = np.triu_indices(N, k=1)
            pair_dists = half_dists[iu]
            
            # All constraints: boundary dists + pairwise half-dists
            all_constr = np.concatenate([bound_dists, pair_dists])
            
            # Soft-min approximation
            # We want to maximize the minimum constraint.
            # Equivalent to minimizing -min(constraints).
            # Use log-sum-exp for soft-min:
            # min(x) approx -1/k * log(sum(exp(-k*x)))
            
            # Adaptive k: start high (sharp), lower if stuck? 
            # For now, fixed k=50 seems good.
            k = 50.0
            
            neg_vals = -all_constr
            max_neg = np.max(neg_vals)
            exp_vals = np.exp((neg_vals - max_neg) * k)
            sum_exp = np.sum(exp_vals)
            
            if sum_exp == 0:
                return -max_neg
            
            soft_max_neg = np.log(sum_exp) / k + max_neg
            soft_min_val = -soft_max_neg
            
            return -soft_min_val
        
        # Use L-BFGS-B with bounds
        bounds = [(-0.4999, 0.4999)] * (2 * N)
        
        # Multiple restarts with different initial perturbations
        best_polished = pos.copy()
        best_polished_r = current_r
        
        # Strategy: 
        # 1. One pass with moderate k (smooth) to escape local minima
        # 2. One pass with high k (sharp) to refine
        
        ks = [20.0, 50.0, 100.0]
        
        for restart in range(3):
            if time.time() > deadline - 5.0:
                break
            
            # Slightly perturb for diversity on restarts > 0
            if restart > 0:
                p_init = pos + np.random.normal(0, 0.005, size=pos.shape)
                p_init = np.clip(p_init, -0.4999, 0.4999)
            else:
                p_init = pos.copy()
                
            # For the first restart, use a smoother objective (lower k) to potentially escape local minima
            # For subsequent restarts, use sharper objectives
            
            if restart == 0:
                k_val = 20.0
            elif restart == 1:
                k_val = 50.0
            else:
                k_val = 100.0
                
            def obj_wrapper(p, k_val=k_val):
                pp = p.reshape(N, 2)
                
                # Boundary distances
                max_abs = np.max(np.abs(pp), axis=1)
                bound_dists = 0.5 - max_abs
                
                if np.any(bound_dists < 0):
                    return 1000.0
                
                # Pairwise distances using cdist
                dists = cdist(pp, pp)
                np.fill_diagonal(dists, np.inf)
                half_dists = dists / 2.0
                
                iu = np.triu_indices(N, k=1)
                pair_dists = half_dists[iu]
                
                all_constr = np.concatenate([bound_dists, pair_dists])
                
                neg_vals = -all_constr
                max_neg = np.max(neg_vals)
                exp_vals = np.exp((neg_vals - max_neg) * k_val)
                sum_exp = np.sum(exp_vals)
                
                if sum_exp == 0:
                    return -max_neg
                
                soft_max_neg = np.log(sum_exp) / k_val + max_neg
                soft_min_val = -soft_max_neg
                
                return -soft_min_val

            try:
                result = minimize(
                    obj_wrapper, 
                    p_init.flatten(), 
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                p_opt = result.x.reshape(N, 2)
                p_opt = np.clip(p_opt, -0.4999, 0.4999)
                r_opt = compute_radius(p_opt)
                
                if r_opt > best_polished_r:
                    best_polished_r = r_opt
                    best_polished = p_opt.copy()
                    
            except Exception:
                pass
        
        return best_polished, best_polished_r

    if time.time() < deadline - 2.0:
        polished_pos, polished_r = polish(best_positions)
        if polished_r > best_radius:
            best_positions = polished_pos
            best_radius = polished_r
            
    return best_positions
