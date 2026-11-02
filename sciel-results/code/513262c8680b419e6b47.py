"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 21

# EVOLVE-BLOCK-START

def _dist_matrix(pos):
    """pos: (N,2) -> (N,N) pairwise distances"""
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    return np.sqrt(np.sum(diff**2, axis=-1))

def _objective_and_grad(pos, w, h, r, penalty=1e6):
    """
    Objective: -sum(r) + penalty * (max(0, 2*max_r - w)**2 + max(0, 2*max_r - h)**2)
    where max_r = max_i (r_i + d_ij/2) over all pairs and walls.
    Actually, let's define:
    For each circle i, its 'effective radius' in the packing sense is r_i.
    The constraint is that for all i,j: d_ij >= r_i + r_j, and for all i:
    x_i >= r_i, y_i >= r_i, x_i <= w - r_i, y_i <= h - r_i.
    
    To maximize sum(r) subject to these, we can use a penalty method or barrier.
    Let's define a smooth penalty for violations.
    
    Violation for pair (i,j): max(0, r_i + r_j - d_ij)
    Violation for wall: max(0, r_i - x_i), max(0, r_i - y_i), max(0, r_i - (w-x_i)), max(0, r_i - (h-y_i))
    
    Objective = -sum(r) + lambda * sum(violations^2)
    """
    n = len(r)
    d = _dist_matrix(pos)
    np.fill_diagonal(d, np.inf)
    
    # Pairwise violations
    r_sum = r[:, np.newaxis] + r[np.newaxis, :]
    pair_viol = np.maximum(0, r_sum - d)
    
    # Wall violations
    wall_viol = np.concatenate([
        np.maximum(0, r - pos[:, 0]),
        np.maximum(0, r - pos[:, 1]),
        np.maximum(0, r - (w - pos[:, 0])),
        np.maximum(0, r - (h - pos[:, 1]))
    ])
    
    total_viol_sq = np.sum(pair_viol**2) + np.sum(wall_viol**2)
    obj = -np.sum(r) + penalty * total_viol_sq
    
    # Gradient w.r.t. pos and r
    # d(obj)/d(pos)
    # For pair (i,j): d/d(pos_i) [max(0, r_i+r_j-d_ij)^2] = 2*max(0, r_i+r_j-d_ij) * (-d/d(pos_i) d_ij)
    # d/d(pos_i) d_ij = (pos_i - pos_j) / d_ij
    # So d/d(pos_i) [violation^2] = 2*violation * (pos_j - pos_i)/d_ij
    
    grad_pos = np.zeros_like(pos)
    for i in range(n):
        for j in range(i+1, n):
            v = pair_viol[i, j]
            if v > 0:
                diff = pos[i] - pos[j]
                dist = d[i, j]
                if dist > 1e-12:
                    grad_pos[i] += 2 * v * (pos[j] - pos[i]) / dist
                    grad_pos[j] += 2 * v * (pos[i] - pos[j]) / dist
    
    # Wall gradients
    # d/d(x_i) max(0, r_i - x_i)^2 = 2*max(0, r_i-x_i)*(-1) = -2*max(0, r_i-x_i)
    # d/d(x_i) max(0, r_i - (w-x_i))^2 = 2*max(0, r_i-(w-x_i))*(1) = 2*max(0, r_i-(w-x_i))
    grad_pos[:, 0] += -2 * np.maximum(0, r - pos[:, 0]) + 2 * np.maximum(0, r - (w - pos[:, 0]))
    grad_pos[:, 1] += -2 * np.maximum(0, r - pos[:, 1]) + 2 * np.maximum(0, r - (h - pos[:, 1]))
    
    grad_pos *= penalty
    
    # d(obj)/d(r_i)
    # From -sum(r): -1
    # From pair (i,j): 2*max(0, r_i+r_j-d_ij) for each j != i
    # From wall: 2*max(0, r_i-x_i) + 2*max(0, r_i-y_i) + 2*max(0, r_i-(w-x_i)) + 2*max(0, r_i-(h-y_i))
    grad_r = -np.ones(n)
    for i in range(n):
        for j in range(n):
            if i != j:
                grad_r[i] += 2 * pair_viol[i, j]
        grad_r[i] += 2 * (np.maximum(0, r[i] - pos[i, 0]) + 
                          np.maximum(0, r[i] - pos[i, 1]) + 
                          np.maximum(0, r[i] - (w - pos[i, 0])) + 
                          np.maximum(0, r[i] - (h - pos[i, 1])))
    grad_r *= penalty
    
    return obj, grad_pos, grad_r

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initial guess: grid arrangement
    # For 21 circles, a 5x4 grid + 1 extra, or 6x4 grid - 3
    # Let's start with a 5x5 grid and remove 4, or just use 5x4+1
    # Better: use a hexagonal-like initial guess
    
    # Start with equal radii in a grid
    side = int(np.ceil(np.sqrt(N)))
    # Place circles in a grid of side x side, take first N
    x0 = np.array([((i % side) + 0.5) / side for i in range(N)])
    y0 = np.array([((i // side) + 0.5) / side for i in range(N)])
    pos0 = np.column_stack([x0, y0])
    r0 = np.full(N, 0.5 / side)
    
    # Rectangle: perimeter 4, so 2(w+h) = 4 => w+h = 2
    # Start with square: w=h=1
    w = 1.0
    h = 1.0
    
    # Optimization: we optimize pos, r, and also w,h with constraint 2(w+h)=4
    # Actually, let's fix w+h=2 and optimize w (h=2-w)
    # Or optimize all 3 with penalty for perimeter
    
    # Let's use scipy.optimize.minimize with L-BFGS-B
    from scipy.optimize import minimize
    
    def full_objective(x):
        """x = [pos (2N), r (N), w] with h = 2-w"""
        pos = x[:2*N].reshape(N, 2)
        r = x[2*N:3*N]
        w_val = x[3*N]
        h_val = 2.0 - w_val
        
        # Ensure w, h > 0
        if w_val <= 0.01 or h_val <= 0.01:
            return 1e10, None
        
        # Ensure radii positive
        r = np.maximum(r, 1e-6)
        
        obj, grad_pos, grad_r = _objective_and_grad(pos, w_val, h_val, r, penalty=1e4)
        
        # Gradient w.r.t. w: h = 2-w
        # d(obj)/dw = d(obj)/dw + d(obj)/dh * (-1)
        # This is complex; let's just use finite differences for w or include it in penalty
        
        # For simplicity, let's fix w=1, h=1 initially and just optimize pos and r
        # Then try different w,h ratios
        
        grad = np.concatenate([grad_pos.flatten(), grad_r, [0.0]])
        return obj, grad
    
    # Start with w=1, h=1
    x_init = np.concatenate([pos0.flatten(), r0, [1.0]])
    
    best_obj = np.inf
    best_x = x_init.copy()
    
    # Try multiple initial conditions and w,h ratios
    for w_ratio in [0.8, 0.9, 1.0, 1.1, 1.25]:
        w_val = 2 * w_ratio / (1 + w_ratio)  # w/(w+h) = w_ratio, w+h=2
        h_val = 2 - w_val
        
        # Scale initial positions
        pos_scaled = pos0 * np.array([w_val, h_val])
        r_scaled = r0 * min(w_val, h_val)  # rough scaling
        
        x_init = np.concatenate([pos_scaled.flatten(), r_scaled, [w_val]])
        
        try:
            res = minimize(full_objective, x_init, method='L-BFGS-B', 
                          jac=True, options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12})
            if res.fun < best_obj:
                best_obj = res.fun
                best_x = res.x
        except:
            pass
        
        if time.time() > deadline:
            break
    
    # Extract best solution
    pos = best_x[:2*N].reshape(N, 2)
    r = np.maximum(best_x[2*N:3*N], 1e-8)
    w_val = best_x[3*N]
    h_val = 2.0 - w_val
    
    # Validate and clean up
    # Ensure all circles are within bounds
    for i in range(N):
        pos[i, 0] = np.clip(pos[i, 0], r[i], w_val - r[i])
        pos[i, 1] = np.clip(pos[i, 1], r[i], h_val - r[i])
    
    # Check for overlaps and fix
    d = _dist_matrix(pos)
    np.fill_diagonal(d, np.inf)
    for i in range(N):
        for j in range(i+1, N):
            if d[i, j] < r[i] + r[j]:
                # Push apart
                diff = pos[i] - pos[j]
                dist = d[i, j]
                if dist > 1e-12:
                    overlap = r[i] + r[j] - dist
                    pos[i] += diff / dist * overlap / 2
                    pos[j] -= diff / dist * overlap / 2
    
    # Final validation
    # Ensure perimeter <= 4
    perim = 2 * (w_val + h_val)
    if perim > 4.0:
        scale = 4.0 / perim
        pos *= scale
        r *= scale
        w_val *= scale
        h_val *= scale
    
    # Build output: (x, y, r)
    result = np.column_stack([pos, r])
    
    return result

# EVOLVE-BLOCK-END
