"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import minimize

def _check_validity(pos, radii, tol=1e-7):
    """Vectorized validity check."""
    n = len(pos)
    if np.any(pos[:, 0] < radii - tol) or np.any(pos[:, 0] > 1 - radii + tol):
        return False
    if np.any(pos[:, 1] < radii - tol) or np.any(pos[:, 1] > 1 - radii + tol):
        return False
    if np.any(radii < -tol):
        return False
    for i in range(n):
        dx = pos[i, 0] - pos[i+1:, 0]
        dy = pos[i, 1] - pos[i+1:, 1]
        dist = np.sqrt(dx * dx + dy * dy)
        if np.any(dist < radii[i] + radii[i+1:] - tol):
            return False
    return True

def _objective_and_grad(z, n, lam=100.0):
    """
    Compute objective and gradient for circle packing optimization.
    z is flattened array [x0, y0, r0, x1, y1, r1, ...]
    Objective: -sum(r) + lam * (sum(overlap_penalties) + sum(boundary_penalties))
    """
    pos = z[0::3].copy() # This is incorrect slicing for [x0,y0,r0, x1,y1,r1]
    # Correct slicing:
    x = z[0::3]
    y = z[1::3]
    r = z[2::3]
    
    # Ensure r is positive for calculation stability, though bounds handle it
    r_safe = np.maximum(r, 1e-6)
    
    obj = -np.sum(r)
    grad = np.zeros_like(z)
    grad[2::3] -= 1.0 # Gradient of -sum(r) wrt r_i is -1
    
    # Boundary penalties
    # x_i < r_i  => (r_i - x_i)^2
    # x_i > 1 - r_i => (x_i - (1 - r_i))^2
    # Similarly for y
    
    for i in range(n):
        # x lower
        if x[i] < r[i]:
            pen = (r[i] - x[i])**2
            obj += lam * pen
            # d/dx_i: 2(r_i - x_i)(-1)
            grad[3*i + 0] += lam * 2 * (r[i] - x[i]) * (-1)
            # d/dr_i: 2(r_i - x_i)(1)
            grad[3*i + 2] += lam * 2 * (r[i] - x[i]) * (1)
            
        # x upper
        if x[i] > 1 - r[i]:
            pen = (x[i] - (1 - r[i]))**2
            obj += lam * pen
            # d/dx_i: 2(x_i - (1-r_i))(1)
            grad[3*i + 0] += lam * 2 * (x[i] - (1 - r[i])) * (1)
            # d/dr_i: 2(x_i - (1-r_i))(1)  (since d(1-r_i)/dr_i = -1, so d(x - 1 + r)/dr = 1)
            grad[3*i + 2] += lam * 2 * (x[i] - (1 - r[i])) * (1)
            
        # y lower
        if y[i] < r[i]:
            pen = (r[i] - y[i])**2
            obj += lam * pen
            grad[3*i + 1] += lam * 2 * (r[i] - y[i]) * (-1)
            grad[3*i + 2] += lam * 2 * (r[i] - y[i]) * (1)
            
        # y upper
        if y[i] > 1 - r[i]:
            pen = (y[i] - (1 - r[i]))**2
            obj += lam * pen
            grad[3*i + 1] += lam * 2 * (y[i] - (1 - r[i])) * (1)
            grad[3*i + 2] += lam * 2 * (y[i] - (1 - r[i])) * (1)
            
    # Overlap penalties
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist_sq = dx*dx + dy*dy
            dist = np.sqrt(dist_sq)
            min_dist = r[i] + r[j]
            
            if dist < min_dist:
                # Penalty: (min_dist - dist)^2
                diff = min_dist - dist
                pen = diff * diff
                obj += lam * pen
                
                if dist < 1e-9:
                    # Avoid division by zero, random gradient
                    grad[3*i + 0] += lam * diff
                    grad[3*i + 1] += lam * diff
                    grad[3*i + 2] += lam * 2 * diff
                    grad[3*j + 0] += lam * diff
                    grad[3*j + 1] += lam * diff
                    grad[3*j + 2] += lam * 2 * diff
                else:
                    # d/dx_i: 2(diff) * d(min_dist - dist)/dx_i = 2(diff) * (-dx/dist)
                    # d/dx_j: 2(diff) * d(min_dist - dist)/dx_j = 2(diff) * (dx/dist)
                    # d/dr_i: 2(diff) * 1
                    # d/dr_j: 2(diff) * 1
                    
                    grad_xi = 2 * diff * (-dx / dist)
                    grad_xj = 2 * diff * (dx / dist)
                    grad_yi = 2 * diff * (-dy / dist)
                    grad_yj = 2 * diff * (dy / dist)
                    grad_ri = 2 * diff
                    grad_rj = 2 * diff
                    
                    grad[3*i + 0] += lam * grad_xi
                    grad[3*i + 1] += lam * grad_yi
                    grad[3*i + 2] += lam * grad_ri
                    
                    grad[3*j + 0] += lam * grad_xj
                    grad[3*j + 1] += lam * grad_yj
                    grad[3*j + 2] += lam * grad_rj
                    
    return obj, grad

def _optimize_single(pos, radii, time_budget=10.0, start_time=None):
    if start_time is None:
        start_time = time.time()
    
    n = len(pos)
    z0 = np.zeros(3 * n)
    for i in range(n):
        z0[3*i + 0] = pos[i, 0]
        z0[3*i + 1] = pos[i, 1]
        z0[3*i + 2] = radii[i]
        
    bounds = []
    for i in range(n):
        bounds.append((0, 1)) # x
        bounds.append((0, 1)) # y
        bounds.append((0, 0.5)) # r
        
    # L-BFGS-B is efficient for large N with gradients
    # We provide the gradient
    def fun(z):
        return _objective_and_grad(z, n)[0]
        
    def jac(z):
        return _objective_and_grad(z, n)[1]
        
    # Check time
    elapsed = time.time() - start_time
    if elapsed > time_budget:
        return pos, radii
        
    try:
        res = minimize(fun, z0, method='L-BFGS-B', jac=jac, bounds=bounds, 
                       options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50})
        
        if res.success:
            z_opt = res.x
            pos_opt = np.array([[z_opt[3*i+0], z_opt[3*i+1]] for i in range(n)])
            radii_opt = np.array([z_opt[3*i+2] for i in range(n)])
            
            # Validate and clamp
            for i in range(n):
                radii_opt[i] = np.clip(radii_opt[i], 0, 0.5)
                pos_opt[i, 0] = np.clip(pos_opt[i, 0], radii_opt[i], 1 - radii_opt[i])
                pos_opt[i, 1] = np.clip(pos_opt[i, 1], radii_opt[i], 1 - radii_opt[i])
                
            return pos_opt, radii_opt
    except Exception:
        pass
        
    return pos, radii

def _build_grid(n):
    side = int(np.ceil(np.sqrt(n)))
    pos = np.array([[(i + 0.5) / side, (j + 0.5) / side]
                    for j in range(side) for i in range(side)][:n])
    r = 0.42 / side
    radii = np.full(n, r)
    return pos, radii

def _build_random(n, seed=0):
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.1, 0.9, size=(n, 2))
    radii = np.full(n, 0.07)
    return pos, radii

def construct():
    start_time = time.time()
    deadline = 145.0 # Safety margin
    
    best_score = 0.0
    best_pos = None
    best_radii = None
    
    # Generate multiple initial configurations
    configs = []
    pos, radii = _build_grid(N)
    configs.append((pos.copy(), radii.copy()))
    
    for seed in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        pos, radii = _build_random(N, seed=seed)
        configs.append((pos.copy(), radii.copy()))
        
    # Add a few "corner" like configs manually
    pos_corner = np.array([
        [0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9],
        [0.5, 0.1], [0.5, 0.9], [0.1, 0.5], [0.9, 0.5],
        [0.3, 0.3], [0.7, 0.3], [0.3, 0.7], [0.7, 0.7],
        [0.5, 0.5], [0.2, 0.5], [0.8, 0.5], [0.5, 0.2],
        [0.5, 0.8], [0.3, 0.5], [0.7, 0.5], [0.5, 0.3],
        [0.5, 0.7], [0.4, 0.4], [0.6, 0.4], [0.4, 0.6],
        [0.6, 0.6], [0.5, 0.55]
    ])
    radii_corner = np.full(N, 0.08)
    configs.append((pos_corner, radii_corner))
    
    # Optimize each config
    for i, (pos, radii) in enumerate(configs):
        elapsed = time.time() - start_time
        if elapsed > deadline - 5:
            break
            
        # Allocate time per optimization
        time_per_opt = min(10.0, (deadline - elapsed - 2) / max(1, len(configs) - i))
        
        pos_opt, radii_opt = _optimize_single(pos, radii, time_budget=time_per_opt, start_time=start_time)
        
        if _check_validity(pos_opt, radii_opt):
            score = np.sum(radii_opt)
            if score > best_score:
                best_score = score
                best_pos = pos_opt.copy()
                best_radii = radii_opt.copy()
                
    # Final polish on best result
    if best_pos is not None:
        elapsed = time.time() - start_time
        remaining = deadline - elapsed - 2
        if remaining > 2:
            pos_final, radii_final = _optimize_single(best_pos, best_radii, time_budget=remaining, start_time=start_time)
            if _check_validity(pos_final, radii_final):
                score = np.sum(radii_final)
                if score > best_score:
                    best_score = score
                    best_pos = pos_final
                    best_radii = radii_final
                    
    if best_pos is None:
        pos, radii = _build_grid(N)
        best_pos, best_radii = pos, radii
        
    result_arr = np.zeros((N, 3))
    for i in range(N):
        result_arr[i] = [best_pos[i, 0], best_pos[i, 1], best_radii[i]]
        
    return result_arr
# EVOLVE-BLOCK-END
