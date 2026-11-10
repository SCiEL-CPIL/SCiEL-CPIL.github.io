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

def soft_min(values, k=1000.0):
    """
    Soft minimum of an array of values using log-sum-exp trick.
    This provides a smooth surrogate for the min function.
    """
    # min(x) = -max(-x)
    # soft_min(x) = -soft_max(-x)
    # soft_max(x) = log(sum(exp(x - max_x))) + max_x
    neg_vals = -values
    max_neg = np.max(neg_vals)
    exp_vals = np.exp(neg_vals - max_neg)
    sum_exp = np.sum(exp_vals)
    if sum_exp == 0:
        return max_neg
    soft_max_neg = np.log(sum_exp) + max_neg
    return -soft_max_neg

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin before 120s limit
    
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
    
    # 2. Simulated Annealing (shorter, to leave time for gradient polish)
    T = 0.001
    T_end = 1e-7
    alpha = 0.999
    step_size = 0.01
    min_step_size = 0.0001
    max_step_size = 0.05
    
    accept_count = 0
    total_count = 0
    
    sa_deadline = start_time + 75.0
    
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
            
    # 3. Polish Phase: Gradient-based local search using soft-min surrogate with L-BFGS-B
    def objective_and_grad(flat_pos, k=100.0):
        """
        Objective to minimize: -soft_min(bound_dists, dists/2)
        Returns (value, gradient)
        """
        pos = flat_pos.reshape(N, 2)
        
        # Boundary distances
        max_abs = np.max(np.abs(pos), axis=1)
        bound_dists = 0.5 - max_abs
        
        # If any point is outside, return a large penalty
        if np.any(bound_dists < 0):
            return 100.0, np.zeros_like(flat_pos)
        
        # Pairwise distances
        dists = cdist(pos, pos)
        np.fill_diagonal(dists, np.inf)
        dists_half = dists / 2.0
        
        # Combine bound_dists and dists/2 into a single array for soft_min
        # bound_dists has shape (N,), dists_half has shape (N, N)
        # We only need upper triangle of dists_half to avoid double counting, but soft_min works on any array
        # Let's take all off-diagonal elements
        mask = ~np.eye(N, dtype=bool)
        dists_flat = dists_half[mask]
        
        # Combine
        all_vals = np.concatenate([bound_dists, dists_flat])
        
        # Compute soft_min and its gradient
        # soft_min(x) = -logsumexp(-x)
        # d(soft_min)/dx_i = -exp(-x_i) / sum(exp(-x_j))
        
        neg_vals = -all_vals
        max_neg = np.max(neg_vals)
        exp_vals = np.exp(neg_vals - max_neg)
        sum_exp = np.sum(exp_vals)
        
        if sum_exp == 0:
            val = -max_neg
            grad = np.zeros_like(flat_pos)
        else:
            soft_max_neg = np.log(sum_exp) + max_neg
            val = -soft_max_neg
            
            # Gradient of soft_min with respect to all_vals
            # d(soft_min)/dx_i = -exp(-x_i) / sum(exp(-x_j)) = -exp_vals[i] / sum_exp
            grad_all = -exp_vals / sum_exp
            
            # Now map back to flat_pos
            grad = np.zeros_like(flat_pos)
            
            # Boundary part: grad_all[0:N] corresponds to bound_dists
            # bound_dists[i] = 0.5 - max(|x_i|, |y_i|)
            # d(bound_dists[i])/d(x_i) = -sign(x_i) if |x_i| >= |y_i| else 0
            # d(bound_dists[i])/d(y_i) = -sign(y_i) if |y_i| > |x_i| else 0
            
            for i in range(N):
                g = grad_all[i]
                if np.abs(pos[i, 0]) >= np.abs(pos[i, 1]):
                    grad[i, 0] += g * (-np.sign(pos[i, 0]))
                else:
                    grad[i, 1] += g * (-np.sign(pos[i, 1]))
            
            # Distance part: grad_all[N:] corresponds to dists_flat
            # dists_half[i,j] = 0.5 * ||pos_i - pos_j||
            # d(dists_half[i,j])/d(pos_i) = 0.5 * (pos_i - pos_j) / ||pos_i - pos_j||
            # d(dists_half[i,j])/d(pos_j) = 0.5 * (pos_j - pos_i) / ||pos_i - pos_j||
            
            # We need to map from the flattened upper triangle back to (i,j) pairs
            # The mask is ~np.eye(N, dtype=bool), which is row-major
            # So dists_flat[k] corresponds to the k-th True element in mask
            
            # For efficiency, let's iterate over the upper triangle
            # Actually, let's just loop over all i < j
            idx = 0
            for i in range(N):
                for j in range(i+1, N):
                    g = grad_all[N + idx]
                    idx += 1
                    d = dists[i, j]
                    if d < 1e-10:
                        # Degenerate case, add small random perturbation
                        grad[i, 0] += g * 0.5 * 0.001
                        grad[j, 0] += g * (-0.5 * 0.001)
                    else:
                        dir_ij = (pos[i] - pos[j]) / d
                        grad[i] += g * 0.5 * dir_ij
                        grad[j] += g * (-0.5 * dir_ij)
        
        return val, grad
    
    # Use L-BFGS-B to minimize the negative soft-min (i.e., maximize soft-min)
    if time.time() < deadline - 5.0:
        pos_flat = best_positions.flatten()
        
        # Run L-BFGS-B with time limit
        # We'll run multiple iterations with decreasing k (sharper soft-min)
        k_values = [50.0, 100.0, 200.0, 500.0]
        
        for k in k_values:
            if time.time() >= deadline - 2.0:
                break
            
            # Define objective with fixed k
            def obj_func(flat_pos, k=k):
                val, grad = objective_and_grad(flat_pos, k)
                return val, grad
            
            try:
                result = minimize(
                    obj_func,
                    pos_flat,
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
                )
                
                if result.success or result.fun < 0:  # result.fun < 0 means we found something better
                    new_pos = result.x.reshape(N, 2)
                    new_pos = np.clip(new_pos, -0.4999, 0.4999)
                    new_radius = compute_radius(new_pos)
                    
                    if new_radius > best_radius:
                        best_radius = new_radius
                        best_positions = new_pos.copy()
                        pos_flat = best_positions.flatten()
                        
            except Exception as e:
                pass
        
        # Final polish with a few more SA iterations if time remains
        if time.time() < deadline - 2.0:
            T = 1e-5
            step_size = 0.001
            for _ in range(5000):
                if time.time() >= deadline - 1.0:
                    break
                
                idx = np.random.randint(N)
                delta = np.random.normal(0, step_size, size=2)
                old_pos = best_positions[idx].copy()
                new_pos = old_pos + delta
                
                best_positions[idx] = new_pos
                new_radius = compute_radius(best_positions)
                
                if new_radius > best_radius:
                    best_radius = new_radius
                else:
                    best_positions[idx] = old_pos
                
                # Slowly decrease step size
                step_size *= 0.999
    
    return best_positions
