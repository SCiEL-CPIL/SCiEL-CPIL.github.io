"""
Strategy: Binary Search on Radius with Direct Feasibility Optimization.

The current approach uses a soft-min surrogate to approximate the maximum radius.
This approach instead:
1. Binary searches for the maximum feasible radius R.
2. For a given candidate radius R, it defines a feasibility objective:
   J(x) = sum( max(0, 2R - ||x_i - x_j||)^2 + sum_k max(0, R - (0.5 - |x_ik|))^2 )
3. It uses SLSQP (or L-BFGS-B with penalty) to minimize J(x).
4. If J(x) < epsilon, the configuration is feasible for R.
5. This avoids the "gradient noise" of the soft-min near the minimum and allows
   the optimizer to focus entirely on resolving the specific conflicts that
   prevent radius R from being achieved.
"""

import numpy as np
import time
from scipy.optimize import minimize

N = 100
DEADLINE = 145.0  # Safety margin for 150s limit

def _score(c):
    """Compute the largest common radius for centres c (shape (N,3))."""
    if c.shape != (N, 3):
        return -1.0
    if not np.all(np.isfinite(c)):
        return -1.0
    # distance to walls
    # min_i min_k (0.5 - |c_ik|)
    wall_dists = 0.5 - np.abs(c)
    min_wall = np.min(wall_dists)
    
    # pairwise distances
    # min_{i<j} ||c_i - c_j|| / 2
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist) / 2.0
    
    return min(min_wall, min_pair)

def _init_grid():
    """
    Initialize with a 4x5x5 grid. 
    4*5*5 = 100 points exactly.
    This provides a structured starting point that is far from random chaos.
    """
    nx, ny, nz = 4, 5, 5
    xs = (np.arange(nx) + 0.5) / nx - 0.5
    ys = (np.arange(ny) + 0.5) / ny - 0.5
    zs = (np.arange(nz) + 0.5) / nz - 0.5
    
    grid = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    return grid[:N].copy()

def _feasibility_obj_grad(x, R):
    """
    Objective: Sum of squared violations for radius R.
    x: flattened (N*3,)
    R: target radius
    
    Violations:
    1. Pairwise: v_ij = max(0, 2R - ||x_i - x_j||)
    2. Wall: v_ik = max(0, R - (0.5 - |x_ik|))
    
    We minimize sum(v^2).
    """
    x = x.reshape(N, 3)
    
    # --- Pairwise Violations ---
    # diff (N, N, 3)
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Only i < j
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu]
    pair_vecs = diff[iu] # shape (M, 3) where M = N*(N-1)/2
    
    # Target dist = 2R
    # Violation v = max(0, 2R - d)
    target = 2.0 * R
    # We only care about pairs where d < 2R
    active_mask = pair_dists < target
    active_dists = pair_dists[active_mask]
    active_vecs = pair_vecs[active_mask]
    
    if len(active_dists) > 0:
        violations = target - active_dists
        obj_val = np.sum(violations**2)
        
        # Gradient calculation
        # d(obj)/d(x_i) = sum_{j in active pairs with i} 2 * v_ij * (- (x_i - x_j) / d_ij)
        # Note: v = 2R - d. dv/dx_i = - (x_i - x_j)/d.
        # d(v^2)/dx_i = 2v * dv/dx_i = 2(2R-d) * (-(x_i-x_j)/d)
        
        grad = np.zeros((N, 3))
        
        # We need to map active pairs back to indices i, j
        # iu is static. active_mask is boolean over iu.
        active_i, active_j = iu[0][active_mask], iu[1][active_mask]
        
        # For each active pair (i, j, v, d, vec):
        # grad_i += 2 * v * (-vec / d)
        # grad_j += 2 * v * (vec / d)  (since vec = x_i - x_j, grad for x_j is opposite)
        
        # Vectorized accumulation using np.add.at
        # factor = 2 * v / d
        factors = 2.0 * violations / active_dists
        
        # grad_i contribution: -factors * active_vecs
        # grad_j contribution: factors * active_vecs
        
        # We can use np.add.at for unbuffered accumulation
        grad_i_contrib = -factors[:, np.newaxis] * active_vecs
        grad_j_contrib = factors[:, np.newaxis] * active_vecs
        
        np.add.at(grad, active_i, grad_i_contrib)
        np.add.at(grad, active_j, grad_j_contrib)
        
    else:
        obj_val = 0.0
        grad = np.zeros((N, 3))

    # --- Wall Violations ---
    # v_ik = max(0, R - (0.5 - |x_ik|))
    # Let w_ik = 0.5 - |x_ik|
    # v = max(0, R - w)
    
    abs_x = np.abs(x)
    wall_dists = 0.5 - abs_x
    
    # Active walls: w < R
    wall_active_mask = wall_dists < R
    active_wall_dists = wall_dists[wall_active_mask]
    
    if len(active_wall_dists) > 0:
        wall_violations = R - active_wall_dists
        obj_val += np.sum(wall_violations**2)
        
        # Gradient for wall part
        # d(v^2)/dx = 2v * dv/dx
        # v = R - (0.5 - |x|) = R - 0.5 + |x|
        # dv/dx = sign(x)
        # d(v^2)/dx = 2 * (R - 0.5 + |x|) * sign(x)
        
        # We need to apply this to the specific coordinates.
        # active_wall_mask is (N, 3).
        # active_wall_dists is 1D.
        
        # Let's construct the gradient array for walls
        # We need sign(x) for the active positions.
        # Since we flattened, let's work with indices.
        
        # Easier: iterate or use broadcasting carefully.
        # Let's create a full (N,3) gradient array for walls and add to grad.
        wall_grad = np.zeros((N, 3))
        
        # For active positions:
        # val = 2 * violation * sign(x)
        # sign(x) = x / |x| (if x!=0)
        # If x==0, sign is undefined, but violation is R-0.5. 
        # In practice, if x is close to 0, we can pick a direction or assume 1.
        # However, if x is exactly 0, the gradient of |x| is 0 in subgradient sense? 
        # Actually, for max(0, R - 0.5 + |x|), if x=0, it's a kink. 
        # Let's just use np.sign(x). If x=0, np.sign(0)=0. This might stall.
        # Better: if x is very close to 0, push it slightly? 
        # For now, standard gradient.
        
        signs = np.sign(x)
        # Calculate gradient values for active spots
        # grad_val = 2 * violation * sign(x)
        
        # We need to map active_wall_dists back to (i, k)
        active_i_idx, active_k_idx = np.where(wall_active_mask)
        
        # violation values
        v_vals = R - wall_dists[active_i_idx, active_k_idx]
        
        # sign values
        s_vals = signs[active_i_idx, active_k_idx]
        
        # If s_vals is 0, we have a problem. Let's replace 0 with 1 (arbitrary push).
        s_vals[s_vals == 0] = 1.0
        
        g_vals = 2.0 * v_vals * s_vals
        
        # Assign to wall_grad
        wall_grad[active_i_idx, active_k_idx] = g_vals
        
        grad += wall_grad

    return obj_val, grad.flatten()

def _optimize_for_radius(c_init, R, time_limit):
    """
    Attempt to find a configuration feasible for radius R starting from c_init.
    Returns (success, best_c).
    """
    start_time = time.time()
    bounds = [(-0.5, 0.5)] * (N * 3)
    x0 = c_init.flatten()
    
    best_x = x0.copy()
    best_obj = _feasibility_obj_grad(x0, R)[0]
    
    # We use SLSQP because it handles the non-smoothness (kinks in max(0, ...))
    # better via line search, or we can use L-BFGS-B which is faster but might
    # struggle with kinks. Given the short time per step, SLSQP might be too slow.
    # Let's stick to L-BFGS-B as it is robust and fast. The squared penalty
    # makes it smooth enough for L-BFGS-B to work well, except at the kinks.
    
    # To handle kinks, we can add a tiny amount of noise if stuck, but let's try
    # standard L-BFGS-B first.
    
    try:
        res = minimize(
            _feasibility_obj_grad,
            x0,
            args=(R,),
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
        )
        
        if res.success or res.fun < best_obj:
            best_x = res.x
            best_obj = res.fun
            
    except Exception:
        pass
        
    # Check if feasible
    # Tolerance: we want sum of squared violations to be very small.
    # If R is ~0.11, 2R ~ 0.22.
    # A violation of 1e-4 leads to sq 1e-8.
    # Let's say feasible if best_obj < 1e-6.
    
    feasible = best_obj < 1e-6
    return feasible, best_x.reshape(N, 3)

def construct():
    start_time = time.time()
    
    # 1. Initial Grid
    c = _init_grid()
    
    # Calculate initial radius
    r_current = _score(c)
    
    # 2. Binary Search for Maximum Radius
    # Lower bound: current grid score
    # Upper bound: theoretical max? 
    # For N=100 in unit cube, volume argument: 100 * (4/3)pi r^3 <= 1 => r <= (3/400pi)^(1/3) ~ 0.128
    # But packing efficiency is ~0.74, so r is likely lower.
    # Specht's best: 0.11138
    # Let's set high bound to 0.125 (safe upper bound)
    
    r_low = r_current
    r_high = 0.125
    
    # We will perform binary search steps.
    # Each step involves an optimization run.
    # We need to manage time carefully.
    
    # Time per optimization step: 
    # L-BFGS-B on 300 vars, 100 iterations. 
    # Each func/grad eval is O(N^2) = 10k ops. 100 * 10k = 1M ops. Very fast (<0.1s).
    # So we can do many steps.
    
    # However, as R increases, feasibility becomes harder.
    # We might need more iterations or restarts.
    
    # Strategy: 
    # - Start with r_low.
    # - Try to increase R.
    # - If success, update r_low.
    # - If fail, update r_high.
    # - Stop when r_high - r_low < 1e-5 or time runs out.
    
    # To improve robustness, if we fail for a new R, we might try to "relax" 
    # by using the best config from the failed run as the new start for a slightly lower R?
    # Or just binary search.
    
    # Let's do a hybrid: 
    # 1. Coarse binary search (10 steps)
    # 2. Fine binary search (10 steps)
    
    total_steps = 20
    step_budget = (DEADLINE - start_time - 5.0) / total_steps
    
    best_c = c.copy()
    best_r = r_current
    
    for step in range(total_steps):
        if time.time() > DEADLINE - 10.0:
            break
            
        r_mid = (r_low + r_high) / 2.0
        
        # Allocate time for this specific optimization
        t_start = time.time()
        t_limit = t_start + step_budget
        
        # Run optimization
        feasible, c_res = _optimize_for_radius(best_c, r_mid, t_limit)
        
        if feasible:
            r_low = r_mid
            best_c = c_res
            best_r = r_mid
        else:
            # If not feasible, we can still use the result c_res as a better starting
            # point for the next lower R, because it likely moved the spheres apart
            # as much as possible.
            # However, strictly speaking, we should keep best_c from the last successful
            # feasible config to ensure we don't drift into a worse local minimum for
            # the next lower R?
            # Actually, c_res is the minimizer of violations for R_mid. 
            # If we lower R, c_res is likely already feasible or close.
            # So using c_res as the next start is smart.
            r_high = r_mid
            # Update start point to the "best effort" config
            # But be careful: if c_res is bad, we might lose good structure.
            # Let's check the score of c_res.
            r_res = _score(c_res)
            if r_res > _score(best_c) * 0.9: # Heuristic: if it's not much worse, use it
                best_c = c_res
            # Else, keep best_c from last success
            
        # Check convergence
        if r_high - r_low < 1e-5:
            break
            
    # Final Polish: 
    # We have a candidate best_c with radius best_r.
    # Try to optimize for best_r + small_delta to squeeze out more.
    if time.time() < DEADLINE - 5.0:
        # Try a few small increments
        for _ in range(5):
            if time.time() > DEADLINE - 5.0:
                break
            r_try = best_r + 0.0001
            feasible, c_res = _optimize_for_radius(best_c, r_try, 5.0)
            if feasible:
                best_c = c_res
                best_r = r_try
            else:
                break
                
    return best_c

# Ensure the function is exposed correctly
# The task requires construct() -> array of shape (N, 3)

if __name__ == "__main__":
    # Quick test
    start = time.time()
    res = construct()
    end = time.time()
    print(f"Time: {end - start:.2f}s")
    print(f"Radius: {_score(res):.10f}")
    print(f"Shape: {res.shape}")
