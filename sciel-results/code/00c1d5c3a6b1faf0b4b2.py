import numpy as np
from scipy.optimize import minimize
import time

N = 50

def compute_violation_and_grad(c, r):
    """
    Computes the squared violation of constraints and its gradient.
    c: shape (N, 2)
    r: scalar
    Returns:
        violation: scalar
        grad: shape (N, 2)
    """
    N = c.shape[0]
    # Pairwise distances
    # d[i, j] is distance between c[i] and c[j]
    # We only need i < j
    c_flat = c.reshape(-1, 1)
    # This is O(N^2) memory and compute, N=50 is small (2500 pairs)
    
    # Efficient distance calculation
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    
    # Mask for i < j
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    dist_pairs = dist[mask]
    # Indices of pairs
    i_idx, j_idx = np.where(mask)
    
    # Violations for pairs
    # Constraint: dist >= 2r
    # Violation: max(0, 2r - dist)
    viol_pairs = np.maximum(0, 2 * r - dist_pairs)
    
    # Boundary violations
    # Constraint: |c_ik| <= 0.5 - r
    # Violation: max(0, |c_ik| - (0.5 - r))
    abs_c = np.abs(c)
    limit = 0.5 - r
    viol_bound = np.maximum(0, abs_c - limit)
    
    # Total violation (sum of squares)
    violation = np.sum(viol_pairs**2) + np.sum(viol_bound**2)
    
    # Gradient
    grad = np.zeros_like(c)
    
    # Pair gradients
    # d/dx_i (max(0, 2r - d_ij)^2)
    # If viol > 0:
    #   d/dx_i ( (2r - d)^2 ) = 2(2r - d) * (-1) * d/dx_i (d)
    #   d/dx_i (d) = (x_i - x_j) / d
    #   So term is -2(2r - d) * (x_i - x_j) / d
    #   For x_j, it is -2(2r - d) * (x_j - x_i) / d = 2(2r - d) * (x_i - x_j) / d
    
    active_pairs = viol_pairs > 0
    if np.any(active_pairs):
        i_act = i_idx[active_pairs]
        j_act = j_idx[active_pairs]
        d_act = dist_pairs[active_pairs]
        v_act = viol_pairs[active_pairs]
        
        # Vectorized update
        # coeff = -2 * v_act / d_act
        coeff = -2 * v_act / d_act
        diff_act = c[i_act] - c[j_act] # shape (num_active, 2)
        
        # grad_i += coeff * diff
        # grad_j -= coeff * diff (since d/dx_j d = (x_j - x_i)/d = -diff/d)
        # Wait, d/dx_j ( (2r-d)^2 ) = 2(2r-d) * (-1) * (x_j-x_i)/d = -2(2r-d) * (x_j-x_i)/d
        # Since x_j - x_i = - (x_i - x_j), this is 2(2r-d) * (x_i - x_j) / d
        # So grad_j += 2 * v_act / d_act * diff_act
        
        # Accumulate using np.add.at for repeated indices
        np.add.at(grad, i_act, coeff * diff_act)
        np.add.at(grad, j_act, -coeff * diff_act)
        
    # Boundary gradients
    active_bound = viol_bound > 0
    if np.any(active_bound):
        v_b = viol_bound[active_bound]
        # d/dx (max(0, |x| - L)^2) = 2(|x|-L) * sign(x)
        # Indices
        rows, cols = np.where(active_bound)
        signs = np.sign(c[rows, cols])
        grad[rows, cols] += 2 * v_b * signs
        
    return violation, grad

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Initial configuration: Grid
    # 50 circles. 7x7 is 49. Let's use 8x7 grid and take 50.
    # Or 5x10.
    # Let's try a 7x7 grid + 1 in the center? No, center is occupied.
    # Let's use a 8x7 grid, spacing 1/8, 1/7.
    m_x, m_y = 8, 7
    x_ticks = (np.arange(m_x) + 0.5) / m_x - 0.5
    y_ticks = (np.arange(m_y) + 0.5) / m_y - 0.5
    grid = np.array([(x, y) for y in y_ticks for x in x_ticks])
    c0 = grid[:N].copy()
    
    # Random jitter to escape local minima
    np.random.seed(42)
    c0 += np.random.normal(0, 0.01, c0.shape)
    
    best_r = 0.0
    best_c = c0.copy()
    
    # Binary search for r
    # Lower bound: 0.05
    # Upper bound: 0.072 (known best is ~0.07137)
    r_low = 0.05
    r_high = 0.072
    
    # We will perform a limited number of binary search steps
    # Each step involves an optimization run.
    # Optimization might fail to find a feasible solution if r is too high.
    # We accept r if min violation < 1e-6.
    
    max_iters = 20  # Binary search iterations
    for _ in range(max_iters):
        if time.time() > deadline:
            break
            
        r_mid = (r_low + r_high) / 2.0
        
        # Run optimization
        # Use L-BFGS-B
        # We need to flatten c for scipy
        c_flat = c0.flatten()
        
        def obj_func(x):
            c = x.reshape(N, 2)
            v, g = compute_violation_and_grad(c, r_mid)
            return v, g.flatten()
        
        # Constraints are handled by the objective function (penalty)
        # We don't use scipy constraints as they are hard constraints, 
        # but our objective is a penalty.
        
        res = minimize(
            obj_func, 
            c_flat, 
            method='L-BFGS-B', 
            jac=True,
            options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6}
        )
        
        c_opt = res.x.reshape(N, 2)
        v_opt, _ = compute_violation_and_grad(c_opt, r_mid)
        
        if v_opt < 1e-6:
            # Feasible
            best_r = r_mid
            best_c = c_opt.copy()
            r_low = r_mid
        else:
            r_high = r_mid
            
    # Final polish if time permits
    if time.time() < deadline - 10 and best_r > 0:
        # Try to slightly increase r and optimize
        r_try = best_r * 1.001
        c_flat = best_c.flatten()
        
        def obj_func_polish(x):
            c = x.reshape(N, 2)
            v, g = compute_violation_and_grad(c, r_try)
            return v, g.flatten()
            
        res = minimize(
            obj_func_polish, 
            c_flat, 
            method='L-BFGS-B', 
            jac=True,
            options={'maxiter': 100, 'ftol': 1e-9, 'gtol': 1e-7}
        )
        c_polished = res.x.reshape(N, 2)
        v_polished, _ = compute_violation_and_grad(c_polished, r_try)
        
        if v_polished < 1e-6:
            best_r = r_try
            best_c = c_polished
        else:
            # If it failed, stick with best_c from previous step
            # But ensure best_c is valid for best_r
            v_check, _ = compute_violation_and_grad(best_c, best_r)
            if v_check > 1e-5:
                # If previous best became invalid (unlikely), revert to grid
                best_c = c0
                # Recalculate r for grid? Grid r is min dist/2.
                # For 8x7 grid, min dist is min(1/8, 1/7) = 0.125 -> r=0.0625
                # But we might have moved them.
                # Just return best_c and let evaluator calculate score.
                # The evaluator calculates the score based on the returned centers.
                # So we just return the centers.
                pass

    return best_c

