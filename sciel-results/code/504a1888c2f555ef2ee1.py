"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_radius(p):
    """Compute the exact common radius for a set of centres."""
    # Boundary constraint
    b = 0.5 - np.max(np.abs(p), axis=1)
    min_b = np.min(b)
    
    # Pairwise distance constraint
    diffs = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_d = np.min(dists) / 2.0
    
    return min(min_b, min_d)

def _smooth_min_radius(p, beta):
    """
    Compute a smooth approximation of the negative radius.
    We want to maximize r, so we minimize -r.
    r = min( min_i b_i, min_{i<j} d_ij/2 )
    
    Smooth min of x is approx by -1/beta * log(sum(exp(-beta*x)))
    Here we define the objective to minimize.
    Let f(p) = -1/beta * log( sum(exp(-beta * b_i)) + sum(exp(-beta * d_ij/2)) )
    As beta -> infinity, this approaches -min(b_i, d_ij/2).
    Minimizing this function maximizes the min radius.
    """
    b = 0.5 - np.max(np.abs(p), axis=1)
    
    diffs = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    np.fill_diagonal(dists, np.inf)
    
    # We only need upper triangle for pairs to avoid double counting, 
    # but for smooth min, summing all is fine if we adjust beta or just accept 
    # that the "weight" of pairs is higher. To be rigorous, let's take unique pairs.
    # For N=50, 1225 pairs. 
    # Actually, summing over all i!=j doubles the count of each distance.
    # min(x, x) = x. The smooth min of a set with duplicates is still x.
    # So summing over all i!=j is valid for the min operation.
    
    # Combine boundary and pairwise distances
    # b is shape (N,)
    # dists is shape (N, N) with inf on diagonal.
    # We want min of all b_i and all d_ij/2.
    
    # To avoid overflow in exp, we subtract the max (or min) first.
    # We are computing -1/beta * log( sum( exp(-beta * val) ) )
    # Let v = -beta * val. We want log-sum-exp(v).
    # log-sum-exp(v) = max(v) + log(sum(exp(v - max(v))))
    
    # Values to minimize: b_i and d_ij/2.
    # Let's create a vector of all constraints.
    # But doing this explicitly might be slow if done naively in python loops.
    # Vectorized:
    
    # Term 1: Boundaries
    v_b = -beta * b
    max_v_b = np.max(v_b)
    sum_exp_b = np.sum(np.exp(v_b - max_v_b))
    
    # Term 2: Pairwise
    v_d = -beta * dists / 2.0
    # Mask diagonal
    v_d = np.where(np.eye(N, dtype=bool), -np.inf, v_d)
    max_v_d = np.max(v_d)
    sum_exp_d = np.sum(np.exp(v_d - max_v_d))
    
    # Combine
    # log( A + B ) where A = sum_exp_b * exp(max_v_b) and B = sum_exp_d * exp(max_v_d)
    # log( A + B ) = log( exp(max_v_b) * (sum_exp_b + exp(max_v_d - max_v_b) * sum_exp_d) )
    # = max_v_b + log( sum_exp_b + exp(max_v_d - max_v_b) * sum_exp_d )
    
    # To be safe against overflow/underflow in the combined log:
    m = np.max([max_v_b, max_v_d])
    sum_total = np.exp(v_b - m) + np.exp(v_d - m)
    log_sum = m + np.log(np.sum(sum_total))
    
    # Objective is -1/beta * log_sum
    return -1.0 / beta * log_sum

def _grad_smooth_min_radius(p, beta):
    """Gradient of the smooth min radius objective."""
    b = 0.5 - np.max(np.abs(p), axis=1)
    
    diffs = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    np.fill_diagonal(dists, np.inf)
    
    v_b = -beta * b
    max_v_b = np.max(v_b)
    exp_b = np.exp(v_b - max_v_b)
    sum_exp_b = np.sum(exp_b)
    
    v_d = -beta * dists / 2.0
    v_d = np.where(np.eye(N, dtype=bool), -np.inf, v_d)
    max_v_d = np.max(v_d)
    exp_d = np.exp(v_d - max_v_d)
    sum_exp_d = np.sum(exp_d)
    
    m = np.max([max_v_b, max_v_d])
    # Total sum of exps relative to m
    total_exp_sum = np.sum(np.exp(v_b - m) + np.exp(v_d - m))
    
    # Derivative of log_sum w.r.t p
    # d/dp log_sum = (1/total_exp_sum) * d/dp (sum exp(v) )
    # v_b_i = -beta * (0.5 - max(|p_i|))
    # dv_b_i/dp_ik = -beta * d/dp_ik (0.5 - max(|p_i|))
    # Let M_i = max(|p_i|). dM_i/dp_ik = sign(p_ik) if |p_ik| == M_i else 0.
    # So dv_b_i/dp_ik = -beta * (-sign(p_ik) * indicator) = beta * sign(p_ik) * indicator
    
    # v_d_ij = -beta/2 * dist_ij
    # dv_d_ij/dp_ik = -beta/2 * d(dist_ij)/dp_ik
    # d(dist_ij)/dp_ik = (p_ik - p_jk) / dist_ij
    
    # Let's compute gradients numerically or analytically. Analytical is better.
    
    grad = np.zeros_like(p)
    
    # Boundary part contribution
    # weight_b_i = exp(v_b_i - m) / total_exp_sum
    weight_b = np.exp(v_b - m) / total_exp_sum
    
    # d v_b_i / d p_ik
    # For each i, find the axis k that maximizes |p_ik|.
    abs_p = np.abs(p)
    max_abs = np.max(abs_p, axis=1)
    # Indicator matrix
    # If multiple axes have the same max, we split the gradient (subgradient).
    # For smoothness, we can just pick one or average. Let's average over maxes.
    # Actually, standard subgradient picks one. Let's use the one with largest value, or all.
    # Let's compute dM_i/dp_ik.
    dM = np.zeros_like(p)
    for i in range(N):
        max_val = max_abs[i]
        # Find indices where abs(p[i,k]) == max_val
        # Due to floating point, use tolerance
        mask = np.isclose(abs_p[i], max_val)
        if np.any(mask):
            # Average the sign over the maxing axes
            for k in range(3):
                if mask[k]:
                    dM[i, k] = np.sign(p[i, k]) / np.sum(mask)
    
    # grad_b_part = sum_i weight_b_i * d v_b_i / d p_i
    # d v_b_i / d p_i = -beta * d b_i / d p_i = -beta * (-dM_i) = beta * dM_i
    grad_b = beta * dM * weight_b[:, np.newaxis]
    
    # Pairwise part contribution
    # weight_d_ij = exp(v_d_ij - m) / total_exp_sum
    weight_d = np.exp(v_d - m) / total_exp_sum
    
    # d v_d_ij / d p_ik = -beta/2 * (p_ik - p_jk) / dist_ij
    # Sum over j.
    # Note: v_d is symmetric. The sum over all j != i will handle both i,j and j,i.
    # Let's sum over all j.
    # grad_d_i = sum_j weight_d_ij * (-beta/2) * (p_i - p_j) / dist_ij
    
    # Vectorized:
    # diffs is (N, N, 3)
    # safe_dists is (N, N)
    safe_dists = np.where(dists > 1e-9, dists, 1.0)
    unit_vecs = diffs / safe_dists[:, :, np.newaxis]
    
    # weight_d is (N, N). We want to sum weight_d[i, j] * unit_vecs[i, j] over j.
    # This gives the vector sum for each i.
    pair_grads = np.sum(weight_d[:, :, np.newaxis] * unit_vecs, axis=1)
    
    grad_d = -beta / 2.0 * pair_grads
    
    total_grad = grad_b + grad_d
    
    # The objective is -1/beta * log_sum.
    # d/dp (-1/beta * log_sum) = -1/beta * (1/log_sum_val) * d(log_sum)/dp
    # Wait, log_sum_val is the value of log(sum exp).
    # d(log_sum)/dp = total_grad (which is d(sum exp)/dp * 1/sum exp ? No.)
    
    # Let S = sum exp(v).
    # d(log S)/dp = (1/S) * dS/dp = (1/S) * sum exp(v) * dv/dp.
    # My weight_b and weight_d were defined as exp(v-m)/S_total.
    # So sum(exp(v-m) * dv/dp) / S_total * S_total?
    
    # Let's re-verify.
    # f = -1/beta * (m + log(S_rel)) where S_rel = sum exp(v-m).
    # df/dp = -1/beta * (1/S_rel) * d(S_rel)/dp
    # d(S_rel)/dp = sum exp(v_i-m) * dv_i/dp
    # So df/dp = -1/beta * (sum exp(v_i-m) * dv_i/dp) / S_rel
    
    # My weight_b = exp(v_b-m)/S_total.
    # My weight_d = exp(v_d-m)/S_total.
    # S_rel = S_total (since we shifted by m).
    
    # So df/dp = -1/beta * ( sum_i weight_b_i * S_total * dv_b_i/dp + sum_ij weight_d_ij * S_total * dv_d_ij/dp )
    # Wait, weight_b_i * S_total = exp(v_b_i - m).
    # So the term is -1/beta * S_total * (sum ... dv/dp ... ) ?
    # No.
    # df/dp = -1/beta * [ sum_i exp(v_b_i-m) dv_b_i/dp + sum_ij exp(v_d_ij-m) dv_d_ij/dp ] / S_total
    # = -1/beta * [ sum_i (exp(v_b_i-m)/S_total) dv_b_i/dp + ... ]
    # = -1/beta * [ sum_i weight_b_i dv_b_i/dp + sum_ij weight_d_ij dv_d_ij/dp ]
    
    # dv_b_i/dp_i = beta * dM_i
    # dv_d_ij/dp_i = -beta/2 * unit_vec_ij
    
    # So df/dp = -1/beta * [ sum_i weight_b_i * beta * dM_i + sum_ij weight_d_ij * (-beta/2 * unit_vec_ij) ]
    # = -1/beta * beta * [ sum_i weight_b_i * dM_i - 1/2 sum_ij weight_d_ij * unit_vec_ij ]
    # = - [ sum_i weight_b_i * dM_i - 1/2 sum_ij weight_d_ij * unit_vec_ij ]
    # = - sum_i weight_b_i * dM_i + 1/2 sum_ij weight_d_ij * unit_vec_ij
    
    # My previous grad_b was beta * dM * weight_b.
    # My previous grad_d was -beta/2 * pair_grads (where pair_grads is sum weight * unit).
    
    # Let's stick to the formula:
    # grad_obj = -1/beta * (grad_log_sum)
    # grad_log_sum = sum_i weight_b_i * dv_b_i/dp + sum_ij weight_d_ij * dv_d_ij/dp
    # dv_b_i/dp = beta * dM_i
    # dv_d_ij/dp = -beta/2 * unit_vec_ij
    
    # grad_log_sum = sum_i weight_b_i * beta * dM_i + sum_ij weight_d_ij * (-beta/2 * unit_vec_ij)
    # = beta * (sum_i weight_b_i * dM_i - 1/2 sum_ij weight_d_ij * unit_vec_ij)
    
    # grad_obj = -1/beta * beta * ( ... ) = - ( ... )
    
    term_b = np.sum(weight_b[:, np.newaxis] * dM, axis=0) # Wait, dM is (N,3), weight_b is (N,).
    # Actually, dM_i is a vector. weight_b_i is scalar.
    # sum_i weight_b_i * dM_i is a vector sum.
    term_b_vec = np.sum(weight_b[:, np.newaxis] * dM, axis=0)
    
    term_d_vec = 0.5 * np.sum(weight_d[:, :, np.newaxis] * unit_vecs, axis=1)
    # Sum over i? No, grad is for each point i.
    # The gradient of the objective with respect to p_i is:
    # - ( weight_b_i * dM_i - 1/2 * sum_j weight_d_ij * unit_vec_ij )
    
    # So:
    grad = -(weight_b[:, np.newaxis] * dM - 0.5 * np.sum(weight_d[:, :, np.newaxis] * unit_vecs, axis=1))
    
    return grad

def _generate_initial_positions():
    """Generate a jittered grid initial configuration."""
    # Approximate cubic root of 50 is ~3.68.
    # Use a 4x4x4 grid? 64 points. Remove 14 random ones?
    # Or 3x4x5 = 60.
    # Let's use a 3x4x5 grid.
    nx, ny, nz = 3, 4, 5
    points = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                points.append([i, j, k])
    
    points = np.array(points)
    
    # Scale to fit in [-0.5, 0.5]
    # Max coord is 2, 3, 4.
    # Scale factor s such that 4*s = 0.5 => s = 0.125
    # But we want to center it.
    # Center of grid: (1.5, 1.5, 2.0)
    points -= np.array([1.5, 1.5, 2.0])
    
    # Scale
    # Max abs coord is 2.0 (for z).
    # We want max abs coord to be ~0.4 to leave room.
    scale = 0.4 / 2.0
    points *= scale
    
    # Add jitter
    rng = np.random.default_rng(42)
    points += rng.uniform(-0.02, 0.02, size=points.shape)
    
    # We have 60 points. We need 50.
    # Remove 10 random points.
    indices = np.arange(len(points))
    rng.shuffle(indices)
    points = points[indices[:N]]
    
    return points

def construct():
    start_time = time.time()
    time_limit = 135.0  # Safety margin before 150s kill
    
    # 1. Initialize
    p = _generate_initial_positions()
    best_r = _compute_radius(p)
    best_p = p.copy()
    
    # 2. Multi-scale soft-min optimization
    # Start with small beta (smooth) and increase to large beta (sharp)
    betas = [10, 50, 100, 200, 500, 1000, 2000]
    
    for beta in betas:
        if time.time() - start_time > time_limit:
            break
            
        # Define objective and gradient
        def obj(x):
            x = x.reshape(N, 3)
            return _smooth_min_radius(x, beta)
            
        def grad(x):
            x = x.reshape(N, 3)
            return _grad_smooth_min_radius(x, beta).flatten()
        
        # Use L-BFGS-B
        # Bounds: [-0.5, 0.5]
        bounds = [(-0.5, 0.5)] * (N * 3)
        
        try:
            res = minimize(obj, p.flatten(), jac=grad, bounds=bounds, method='L-BFGS-B',
                           options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8})
            if res.success or res.status == 0:
                p_new = res.x.reshape(N, 3)
                r_new = _compute_radius(p_new)
                if r_new > best_r:
                    best_r = r_new
                    best_p = p_new.copy()
                    p = p_new.copy()
        except Exception:
            pass
            
    # 3. Local Search Polish
    # Use a simple gradient ascent on the exact radius?
    # Or just a few steps of repulsion.
    
    # Since _compute_radius is not differentiable, we do a random perturbation search.
    current_p = best_p.copy()
    current_r = best_r
    step_size = 0.01
    
    while time.time() - start_time < time_limit:
        improved = False
        # Try random perturbations
        for _ in range(100):
            p_try = current_p + np.random.normal(0, step_size, size=current_p.shape)
            # Clamp
            p_try = np.clip(p_try, -0.5, 0.5)
            r_try = _compute_radius(p_try)
            if r_try > current_r:
                current_p = p_try
                current_r = r_try
                improved = True
        
        if not improved:
            step_size *= 0.8
            if step_size < 1e-5:
                break
        else:
            # If improved, maybe increase step slightly? No, keep stable.
            pass
            
    if current_r > best_r:
        best_r = current_r
        best_p = current_p.copy()
        
    return best_p
# EVOLVE-BLOCK-END
