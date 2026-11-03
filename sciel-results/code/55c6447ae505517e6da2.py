"""Circle packing in unit square: 26 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

def _compute_radii(pos, n):
    """
    Compute maximum possible radii for each circle given positions.
    Uses iterative relaxation to ensure strict non-overlap and boundary containment.
    """
    radii = np.minimum(
        np.minimum(pos[:, 0], pos[:, 1]),
        np.minimum(1.0 - pos[:, 0], 1.0 - pos[:, 1])
    )
    radii = np.maximum(radii, 0.0)
    
    for _ in range(50):
        changed = False
        for i in range(n):
            r_bound = min(pos[i, 0], pos[i, 1], 1.0 - pos[i, 0], 1.0 - pos[i, 1])
            if radii[i] > r_bound:
                radii[i] = r_bound
                changed = True
            for j in range(n):
                if i == j:
                    continue
                d = np.linalg.norm(pos[i] - pos[j])
                if d < radii[i] + radii[j]:
                    excess = radii[i] + radii[j] - d
                    radii[i] = max(0.0, radii[i] - excess / 2.0)
                    radii[j] = max(0.0, radii[j] - excess / 2.0)
                    changed = True
        if not changed:
            break
    
    for _ in range(10):
        changed = False
        for i in range(n):
            r_bound = min(pos[i, 0], pos[i, 1], 1.0 - pos[i, 0], 1.0 - pos[i, 1])
            if radii[i] > r_bound:
                radii[i] = r_bound
                changed = True
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(pos[i] - pos[j])
                if d < radii[i] + radii[j]:
                    excess = radii[i] + radii[j] - d
                    radii[i] = max(0.0, radii[i] - excess / 2.0)
                    radii[j] = max(0.0, radii[j] - excess / 2.0)
                    changed = True
        if not changed:
            break
    
    return radii

def _objective_and_grad(params, n, penalty_weight):
    """
    Objective function for gradient-based optimization.
    params: flattened array of shape (3*n,) containing [x1,y1,r1, x2,y2,r2, ...]
    Returns: negative sum of radii minus penalty for violations, and its gradient.
    """
    x = params[:n*2].reshape(n, 2)
    r = params[n*2:]
    
    # Negative sum of radii (we minimize)
    obj = -np.sum(r)
    grad_obj = np.zeros_like(params)
    grad_obj[n*2:] = -1.0
    
    # Penalty for boundary violations
    boundary_violation = 0.0
    for i in range(n):
        # Circle must be inside [0,1]^2
        # x[i] >= r[i], y[i] >= r[i], x[i] <= 1-r[i], y[i] <= 1-r[i]
        viol = 0.0
        if x[i, 0] < r[i]:
            viol += (r[i] - x[i, 0]) ** 2
        if x[i, 0] > 1.0 - r[i]:
            viol += (x[i, 0] - (1.0 - r[i])) ** 2
        if x[i, 1] < r[i]:
            viol += (r[i] - x[i, 1]) ** 2
        if x[i, 1] > 1.0 - r[i]:
            viol += (x[i, 1] - (1.0 - r[i])) ** 2
        boundary_violation += viol
    
    # Penalty for overlap violations
    overlap_violation = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(x[i] - x[j])
            if d < r[i] + r[j]:
                viol = (r[i] + r[j] - d) ** 2
                overlap_violation += viol
    
    penalty = penalty_weight * (boundary_violation + overlap_violation)
    obj += penalty
    
    # Gradient of penalty
    grad_penalty = np.zeros_like(params)
    
    # Boundary penalty gradients
    for i in range(n):
        # d/dx_i of (r_i - x_i)^2 when x_i < r_i
        if x[i, 0] < r[i]:
            grad_penalty[2*i] = penalty_weight * 2 * (r[i] - x[i, 0])
            grad_penalty[n*2 + i] += penalty_weight * 2 * (x[i, 0] - r[i])
        if x[i, 0] > 1.0 - r[i]:
            grad_penalty[2*i] = penalty_weight * 2 * (x[i, 0] - (1.0 - r[i]))
            grad_penalty[n*2 + i] += penalty_weight * 2 * ((1.0 - r[i]) - x[i, 0])
        if x[i, 1] < r[i]:
            grad_penalty[2*i + 1] = penalty_weight * 2 * (r[i] - x[i, 1])
            grad_penalty[n*2 + i] += penalty_weight * 2 * (x[i, 1] - r[i])
        if x[i, 1] > 1.0 - r[i]:
            grad_penalty[2*i + 1] = penalty_weight * 2 * (x[i, 1] - (1.0 - r[i]))
            grad_penalty[n*2 + i] += penalty_weight * 2 * ((1.0 - r[i]) - x[i, 1])
    
    # Overlap penalty gradients
    for i in range(n):
        for j in range(i + 1, n):
            d_vec = x[i] - x[j]
            d = np.linalg.norm(d_vec)
            if d > 1e-10 and d < r[i] + r[j]:
                viol = r[i] + r[j] - d
                # d/dx_i of (r_i + r_j - d)^2 = 2*(r_i+r_j-d) * d/dx_i(-d)
                # d/dx_i(-d) = -d_vec/d
                grad_penalty[2*i:2*i+2] += penalty_weight * 2 * viol * (-d_vec / d)
                grad_penalty[2*j:2*j+2] += penalty_weight * 2 * viol * (d_vec / d)
                # d/dr_i of (r_i + r_j - d)^2 = 2*(r_i+r_j-d)
                grad_penalty[n*2 + i] += penalty_weight * 2 * viol
                grad_penalty[n*2 + j] += penalty_weight * 2 * viol
    
    grad = grad_obj + grad_penalty
    return obj, grad

def _optimize_gradient(pos_init, n, deadline):
    """
    Run gradient-based optimization starting from pos_init.
    """
    radii_init = _compute_radii(pos_init, n)
    
    # Initial params
    params_init = np.zeros(3 * n)
    params_init[:n*2] = pos_init.flatten()
    params_init[n*2:] = radii_init
    
    # Start with moderate penalty, increase if violations persist
    penalty_weights = [10.0, 50.0, 200.0, 1000.0, 5000.0]
    
    best_params = params_init.copy()
    best_score = -np.sum(radii_init)
    
    for pw in penalty_weights:
        if time.time() > deadline:
            break
        
        try:
            result = minimize(
                _objective_and_grad,
                params_init,
                args=(n, pw),
                method='L-BFGS-B',
                jac=True,
                options={
                    'maxiter': 200,
                    'ftol': 1e-10,
                    'gtol': 1e-8,
                    'maxfun': 500
                }
            )
            
            if result.fun < best_score:
                # Check if this is better in terms of actual sum of radii
                # (the objective includes penalties, so we need to verify)
                x = result.x[:n*2].reshape(n, 2)
                r = result.x[n*2:]
                radii_valid = _compute_radii(x, n)
                actual_score = np.sum(radii_valid)
                
                if -actual_score < best_score or np.sum(radii_valid) > -best_score:
                    best_score = -actual_score
                    best_params = result.x.copy()
            
            # Use result as new starting point
            params_init = result.x.copy()
        except Exception:
            break
    
    # Extract final result
    x = best_params[:n*2].reshape(n, 2)
    r = best_params[n*2:]
    
    # Compute valid radii
    radii_final = _compute_radii(x, n)
    
    return x, radii_final, np.sum(radii_final)

def _initial_grid(n):
    """Create initial grid configuration."""
    pos = np.zeros((n, 2))
    rows = 5
    cols = 6
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= n:
                break
            x = (j + 0.5) / cols
            y = (i + 0.5) / rows
            pos[idx] = [x, y]
            idx += 1
    while idx < n:
        pos[idx] = [0.5, 0.5]
        idx += 1
    return pos

def _initial_random(n, seed=42):
    """Create random initial configuration with minimum separation."""
    rng = np.random.RandomState(seed)
    pos = np.zeros((n, 2))
    min_sep = 0.08
    for i in range(n):
        attempts = 0
        while attempts < 100:
            p = rng.uniform(0.05, 0.95, size=2)
            ok = True
            for j in range(i):
                if np.linalg.norm(p - pos[j]) < min_sep:
                    ok = False
                    break
            if ok:
                pos[i] = p
                break
            attempts += 1
        if attempts >= 100:
            pos[i] = rng.uniform(0.05, 0.95, size=2)
    return pos

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_result = None
    best_score = -np.inf
    
    # Generate multiple initial configurations
    configs = []
    
    # Grid
    pos_grid = _initial_grid(N)
    configs.append(pos_grid)
    
    # Random starts with different seeds
    for seed in [0, 1, 2, 3, 4, 5, 6, 7]:
        pos_rand = _initial_random(N, seed=seed)
        configs.append(pos_rand)
    
    # Process each config with gradient optimization
    for idx, pos_init in enumerate(configs):
        if time.time() > deadline - 15:
            break
        
        try:
            pos_opt, radii_opt, score = _optimize_gradient(pos_init, N, deadline - 5)
            
            if score > best_score:
                best_score = score
                result = np.zeros((N, 3))
                result[:, :2] = pos_opt
                result[:, 2] = radii_opt
                best_result = result
        except Exception:
            continue
    
    if best_result is None:
        # Fallback
        result = np.zeros((N, 3))
        idx = 0
        for i in range(5):
            for j in range(5):
                if idx >= N:
                    break
                x = (i + 0.5) / 5.0
                y = (j + 0.5) / 5.0
                r = 0.08
                result[idx, :] = [x, y, r]
                idx += 1
        if idx < N:
            result[idx, :] = [0.5, 0.5, 0.05]
        return result
    
    return best_result

# EVOLVE-BLOCK-END
