"""
Packing 200 equal circles in a unit square.
Strategy:
1. Initialize with optimized hexagonal lattice.
2. Global optimization using L-BFGS-B with a smooth soft-min (log-sum-exp) surrogate 
   for the radius objective, providing analytical gradients.
3. Local polish using active-set formulation with high-precision L-BFGS-B.
"""

import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

N = 200
EPS = 1e-12

def _compute_radius(c):
    """Compute the largest common radius for given centres."""
    n = c.shape[0]
    # Boundary distances
    boundary_dists = 0.5 - np.abs(c)
    min_boundary = np.min(boundary_dists)
    
    if n < 2:
        return min_boundary
    
    # Pairwise distances
    dists = pdist(c, metric='euclidean')
    min_pair_dist = np.min(dists)
    
    return min(min_pair_dist / 2.0, min_boundary)

def _soft_min_objective(params, beta):
    """
    Objective function: -soft_min(constraints)
    Constraints are pairwise distances / 2 and boundary distances.
    We want to maximize the minimum constraint value.
    soft_min(x) = -1/beta * log(sum(exp(-beta * x_i)))
    """
    c = np.clip(params.reshape(-1, 2), -0.5 + EPS, 0.5 - EPS)
    n = c.shape[0]
    
    # Calculate all constraint values
    # 1. Pairwise
    dists = pdist(c, metric='euclidean')
    pair_vals = dists / 2.0
    
    # 2. Boundary
    boundary_dists = 0.5 - np.abs(c)
    
    # Combine
    all_vals = np.concatenate([pair_vals, boundary_dists.flatten()])
    
    # Soft min calculation (numerically stable)
    # min_val approx = -1/beta * log(sum(exp(-beta * v)))
    # To avoid overflow, subtract max of -beta*v (which is -beta*min(v))
    # Let u = -beta * v. Then min(v) = -max(u)/beta.
    # soft_min = -1/beta * log(sum(exp(u)))
    # u = -beta * all_vals
    # max_u = -beta * min(all_vals)
    # sum_exp = sum(exp(u - max_u))
    # log_sum_exp = max_u + log(sum_exp)
    # soft_min = -1/beta * log_sum_exp
    
    min_val = np.min(all_vals)
    # If min_val is very small, beta * min_val might be small, but beta * (all_vals - min_val) is >= 0
    # We want sum(exp(-beta * v_i)).
    # Let's use the standard log-sum-exp trick on -beta * v_i.
    # Actually, it's easier to just shift by the minimum value to keep exponents <= 0.
    # exp(-beta * v_i) = exp(-beta * (v_i - min_val)) * exp(-beta * min_val)
    # sum(exp(-beta * v_i)) = exp(-beta * min_val) * sum(exp(-beta * (v_i - min_val)))
    # soft_min = -1/beta * ( -beta * min_val + log(sum(exp(-beta * (v_i - min_val)))) )
    #          = min_val - 1/beta * log(sum(exp(-beta * (v_i - min_val))))
    
    shifted = -beta * (all_vals - min_val)
    # shifted is <= 0, so exp is safe
    sum_exp = np.sum(np.exp(shifted))
    log_sum_exp = np.log(sum_exp)
    
    soft_min_val = min_val - log_sum_exp / beta
    
    return -soft_min_val

def _soft_min_gradient(params, beta):
    """
    Gradient of -soft_min(constraints) w.r.t params.
    d/dp (-soft_min) = sum_i (p_i) * d(constraint_i)/dp
    where p_i = exp(-beta * v_i) / sum_j exp(-beta * v_j)
    Note: soft_min = min_val - 1/beta * log(sum(exp(-beta*(v-min_val))))
          d(soft_min)/dp = sum_i (exp(-beta*(v_i-min_val))/sum) * d(v_i)/dp
                         = sum_i p_i * d(v_i)/dp
          We minimize -soft_min, so grad = - sum_i p_i * d(v_i)/dp
    """
    c = np.clip(params.reshape(-1, 2), -0.5 + EPS, 0.5 - EPS)
    n = c.shape[0]
    
    dists = pdist(c, metric='euclidean')
    pair_vals = dists / 2.0
    boundary_dists = 0.5 - np.abs(c)
    
    all_vals = np.concatenate([pair_vals, boundary_dists.flatten()])
    
    min_val = np.min(all_vals)
    shifted = -beta * (all_vals - min_val)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    probs = exp_shifted / sum_exp
    
    grad_params = np.zeros_like(params)
    
    # 1. Pairwise gradients
    # v_ij = 0.5 * ||c_i - c_j||
    # d(v_ij)/dc_i = 0.5 * (c_i - c_j) / ||c_i - c_j||
    # d(v_ij)/dc_j = 0.5 * (c_j - c_i) / ||c_i - c_j||
    
    # Unpack pairs
    # pdist order: (0,1), (0,2), ..., (0,n-1), (1,2), ...
    n_pairs = len(pair_vals)
    if n_pairs > 0:
        # Create indices for pairs
        # This is O(N^2) memory if we do it naively, but N=200 is small (19900 pairs)
        # We can iterate or vectorize. Vectorization is faster.
        
        # Generate indices for pdist
        # i < j
        # For k in 0..N-2:
        #   for j in k+1..N-1:
        
        # Let's use a helper to map linear index to (i, j)
        # Or just loop, as N=200 is small enough for a python loop in gradient? 
        # 19900 iterations * 200 calls? No, gradient is called ~100 times. 2M ops is fine.
        # But vectorized is better.
        
        # Let's try to vectorize the accumulation.
        # We need to add p_k * 0.5 * (c_i - c_j) / dist_k to grad[i]
        # and p_k * 0.5 * (c_j - c_i) / dist_k to grad[j]
        
        # Construct i, j arrays
        # This is a bit tricky to vectorize without creating huge matrices.
        # Given N=200, a simple loop is probably faster than complex vectorization overhead in Python for the index mapping.
        
        idx = 0
        for i in range(n - 1):
            for j in range(i + 1, n):
                p_k = probs[idx]
                if p_k < 1e-12:
                    idx += 1
                    continue
                diff = c[i] - c[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-10:
                    idx += 1
                    continue
                factor = -0.5 * p_k / dist # Negative because we minimize -soft_min
                grad_params[i] += factor * diff
                grad_params[j] -= factor * diff
                idx += 1

    # 2. Boundary gradients
    # v_b = 0.5 - |c_k| (for each coordinate)
    # d(v_b)/dc = -sign(c_k)
    # grad = - sum p_b * (-sign(c_k)) = sum p_b * sign(c_k)
    
    n_bounds = len(boundary_dists)
    if n_bounds > 0:
        # boundary_dists is flattened: [x0, y0, x1, y1, ...]
        # probs is aligned with all_vals: [pairs..., bounds...]
        bound_probs = probs[n_pairs:]
        
        # Reshape probs to (N, 2)
        bound_probs_2d = bound_probs.reshape(n, 2)
        
        # Gradient contribution: p * sign(c)
        # Note: d(-soft_min)/dc = - sum p * dv/dc = - sum p * (-sign(c)) = sum p * sign(c)
        signs = np.sign(c)
        grad_params += bound_probs_2d * signs
        
    return grad_params

def _active_set_polish(c0, max_time=10.0):
    """
    Polish using only active constraints.
    """
    start_time = time.time()
    c = c0.copy()
    r = _compute_radius(c)
    n = c.shape[0]
    
    def _get_active_indices(r_current, margin=1e-4):
        dists = pdist(c, metric='euclidean')
        pair_vals = dists / 2.0
        boundary_dists = 0.5 - np.abs(c)
        
        active_pairs = np.where(pair_vals < r_current + margin)[0]
        active_bounds = np.where(boundary_dists.flatten() < r_current + margin)[0]
        
        # Convert pair indices to (i, j)
        pairs = []
        for idx in active_pairs:
            i = 0
            while i < n - 1:
                count = n - i - 1
                if idx < i * count + count:
                    break
                idx -= count
                i += 1
            j = i + 1 + idx
            pairs.append((i, j))
            
        return pairs, active_bounds

    def _active_obj_grad(params, pairs, active_bounds, beta=1000.0):
        c_local = np.clip(params.reshape(-1, 2), -0.5 + EPS, 0.5 - EPS)
        
        vals = []
        grads = np.zeros_like(params)
        
        # Pair vals
        if len(pairs) > 0:
            diffs = []
            for i, j in pairs:
                diffs.append(c_local[i] - c_local[j])
            diffs = np.array(diffs)
            dists = np.linalg.norm(diffs, axis=1)
            pair_vals_local = dists / 2.0
            vals.append(pair_vals_local)
            
        # Bound vals
        if len(active_bounds) > 0:
            bi = active_bounds // 2
            bj = active_bounds % 2
            bd = 0.5 - np.abs(c_local[bi, bj])
            vals.append(bd)
            
        if not vals:
            return 0.0, grads
            
        all_vals = np.concatenate(vals)
        
        # Soft min on active set
        min_val = np.min(all_vals)
        shifted = -beta * (all_vals - min_val)
        exp_shifted = np.exp(shifted)
        sum_exp = np.sum(exp_shifted)
        probs = exp_shifted / sum_exp
        
        obj = -(min_val - np.log(sum_exp) / beta)
        
        # Grads
        # Pairs
        if len(pairs) > 0:
            pair_probs = probs[:len(pairs)]
            for k, (i, j) in enumerate(pairs):
                p_k = pair_probs[k]
                if p_k < 1e-12: continue
                diff = diffs[k]
                dist = dists[k]
                if dist < 1e-10: continue
                factor = -0.5 * p_k / dist
                grads[i] += factor * diff
                grads[j] -= factor * diff
                
        # Bounds
        if len(active_bounds) > 0:
            bound_probs = probs[len(pairs):]
            bi = active_bounds // 2
            bj = active_bounds % 2
            signs = np.sign(c_local[bi, bj])
            grads[bi, bj] += bound_probs * signs
            
        return obj, grads

    # Iterative polish
    for _ in range(3):
        if time.time() > start_time + max_time:
            break
            
        pairs, active_bounds = _get_active_indices(r, margin=1e-5)
        
        if not pairs and not active_bounds:
            break
            
        try:
            result = minimize(
                lambda p: _active_obj_grad(p, pairs, active_bounds, beta=1000.0)[0],
                c.flatten(),
                jac=lambda p: _active_obj_grad(p, pairs, active_bounds, beta=1000.0)[1],
                method='L-BFGS-B',
                bounds=[(-0.5 + EPS, 0.5 - EPS)] * (2 * N),
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-14, 'maxls': 100}
            )
            c_new = np.clip(result.x.reshape(-1, 2), -0.5 + EPS, 0.5 - EPS)
            r_new = _compute_radius(c_new)
            if r_new > r:
                c = c_new
                r = r_new
            else:
                break
        except Exception:
            break
            
    return c, r

def _initialize_hexagonal():
    """Initialize with a hexagonal pattern optimized for the square."""
    best_c = None
    best_r = -1
    
    # Search for best spacing
    # Approx radius for hex is ~ 0.0366. Spacing d ~ 2*r ~ 0.073
    # Let's search d from 0.070 to 0.076
    for d in np.linspace(0.070, 0.076, 30):
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
                if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                    points.append([x, y])
                x += d
            y += row_spacing
            row += 1
        
        points = np.array(points)
        if len(points) >= N:
            c = points[:N].copy()
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

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    best_c = _initialize_hexagonal()
    best_r = _compute_radius(best_c)
    
    # Main optimization loop with annealing beta
    # Start with low beta (smooth) and increase
    betas = [10.0, 20.0, 50.0, 100.0, 200.0]
    
    current_c = best_c.copy()
    
    for beta in betas:
        if time.time() > deadline - 20.0:
            break
            
        def obj(p):
            return _soft_min_objective(p, beta)
        
        def grad(p):
            return _soft_min_gradient(p, beta)
            
        try:
            result = minimize(
                obj,
                current_c.flatten(),
                jac=grad,
                method='L-BFGS-B',
                bounds=[(-0.5 + EPS, 0.5 - EPS)] * (2 * N),
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
            )
            
            c_new = np.clip(result.x.reshape(-1, 2), -0.5 + EPS, 0.5 - EPS)
            r_new = _compute_radius(c_new)
            
            if r_new > best_r:
                best_r = r_new
                best_c = c_new.copy()
                current_c = c_new.copy()
            else:
                # If it didn't improve, maybe try a smaller step or just continue
                # L-BFGS might have stuck.
                pass
        except Exception:
            pass

    # Final polish
    if time.time() < deadline - 5.0:
        remaining_time = deadline - time.time() - 3.0
        if remaining_time > 2.0:
            c_polish, r_polish = _active_set_polish(best_c, max_time=remaining_time)
            if r_polish > best_r:
                best_r = r_polish
                best_c = c_polish.copy()
                
    # Ensure validity
    best_c = np.clip(best_c, -0.5, 0.5)
    
    return best_c
