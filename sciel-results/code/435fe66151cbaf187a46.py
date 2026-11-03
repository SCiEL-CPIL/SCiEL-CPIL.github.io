import numpy as np
import time
from scipy.optimize import minimize

N = 100

def _compute_radius(centers):
    """
    Computes the maximum common radius for a given set of centers.
    Centers: (N, 3) array.
    Cube: [-0.5, 0.5]^3.
    """
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    
    # Wall constraints: distance from center to nearest wall
    # d_i = min(0.5 - x, x - (-0.5), 0.5 - y, ...) = 0.5 - max(|x|, |y|, |z|)
    max_abs = np.max(np.abs(centers), axis=1)
    wall_dists = 0.5 - max_abs
    min_wall_dist = np.min(wall_dists)
    
    if min_wall_dist < 0:
        return 0.0

    # Pairwise constraints
    # Calculate all pairwise distances efficiently
    # Using broadcasting: (N, 1, 3) - (1, N, 3) -> (N, N, 3)
    # This creates a large array (100*100*3 = 30000 floats), which is fine.
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Get upper triangle indices (excluding diagonal)
    i, j = np.triu_indices(N, 1)
    min_pair_dist = np.min(dists[i, j])
    
    # The radius is limited by half the minimum pair distance
    # and the minimum wall distance
    return min(min_pair_dist / 2.0, min_wall_dist)

def _energy_and_gradient(centers, r_target, k=50.0):
    """
    Computes a smooth energy function and its gradient.
    Energy is the sum of soft violations of the constraints.
    We want to maximize r, so we minimize energy for a fixed r_target.
    
    Constraints:
    1. Pairwise distance >= 2*r_target
    2. Wall distance >= r_target
    
    Soft violation for pair (i,j): max(0, 2*r - d_ij)^2
    Soft violation for wall (i): max(0, r - w_i)^2
    
    To make it differentiable, we use a smooth approximation:
    soft_max(0, x) approx x + log(1 + exp(-x/k)) * k ... wait, standard softplus is log(1+exp(x)).
    Let's use: f(x) = x if x>0 else 0. Smooth version: (x + sqrt(x^2 + eps))/2? No.
    Let's use the standard softplus for the penalty term itself.
    Penalty = (softplus(2*r - d) / k)^2 ?
    
    Actually, simpler:
    We minimize: sum_{i<j} (max(0, 2*r - d_ij))^2 + sum_i (max(0, r - w_i))^2
    
    Gradient of max(0, x)^2 is 2*max(0, x) * sign(x) ? No, 2*max(0, x) if x>0 else 0.
    This is not differentiable at 0.
    
    Let's use a smooth approximation:
    h(x) = (x + sqrt(x^2 + eps))/2 is not quite right for max(0,x).
    max(0, x) approx (x + sqrt(x^2 + eps))/2 is actually approx |x|/2 + x/2.
    If x>0, approx x. If x<0, approx 0.
    So max(0, x)^2 approx ((x + sqrt(x^2 + eps))/2)^2.
    
    Let's stick to the non-smooth gradient but use a small epsilon in the sqrt to avoid division by zero if we used analytic gradient of the smooth version.
    However, L-BFGS-B handles non-smoothness okay-ish, but smooth is better.
    
    Let's define:
    E = sum_{i<j} (max(0, 2r - d_ij))^2 + sum_i (max(0, r - w_i))^2
    
    Gradient wrt center k:
    For pair (k, j):
    term = max(0, 2r - d_kj)
    grad_term = -2 * term * (c_k - c_j) / d_kj   (if term > 0)
    
    For wall k:
    term = max(0, r - w_k)
    w_k = 0.5 - max(|x_k|, |y_k|, |z_k|)
    This is tricky because of the max.
    w_k is the distance to the closest face.
    If x_k is the dominant coordinate (|x_k| > |y_k|, |z_k|), then w_k = 0.5 - |x_k|.
    grad w_k / grad x_k = -sign(x_k).
    So grad term = -2 * term * (-sign(x_k)) = 2 * term * sign(x_k).
    
    Let's implement this carefully.
    """
    centers = np.array(centers)
    energy = 0.0
    grad = np.zeros_like(centers)
    
    # Wall constraints
    max_abs = np.max(np.abs(centers), axis=1)
    wall_dists = 0.5 - max_abs
    violations = np.maximum(0, r_target - wall_dists)
    
    # Sum of squared violations
    energy += np.sum(violations**2)
    
    # Gradient of wall violations
    # We need to know which coordinate was the max.
    # If there are ties, gradient is undefined, but we can pick one or average.
    # For now, let's just use the gradient of the active constraint.
    # If |x| is max, grad_x = -sign(x), grad_y=0, grad_z=0.
    for i in range(N):
        if violations[i] > 0:
            # Find which coordinate had the max absolute value
            abs_c = np.abs(centers[i])
            max_idx = np.argmax(abs_c)
            sign = np.sign(centers[i, max_idx])
            if sign == 0: sign = 1.0 # Edge case, center at 0
            
            # d(wall_dist)/d(centers[i, max_idx]) = -sign
            # d(violation)/d(centers[i, max_idx]) = -d(wall_dist)/d(centers[i, max_idx]) = sign
            # d(energy)/d(centers[i, max_idx]) = 2 * violation * sign
            grad[i, max_idx] += 2.0 * violations[i] * sign
            
    # Pairwise constraints
    # To be efficient, we can loop over pairs, but N=100 means 4950 pairs.
    # 4950 * 100 iterations is 500k operations, which is fast in Python if vectorized per pair? No, loop is slow.
    # Let's vectorize the distance calculation, then loop for gradient accumulation?
    # Or just loop over all pairs. 5000 iterations is acceptable.
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Mask for upper triangle
    i_idx, j_idx = np.triu_indices(N, 1)
    pair_dists = dists[i_idx, j_idx]
    pair_violations = np.maximum(0, 2.0 * r_target - pair_dists)
    
    energy += np.sum(pair_violations**2)
    
    # Gradient accumulation
    # For each violating pair (i, j), add gradient to i and j.
    # grad_i += -2 * v_ij * (c_i - c_j) / d_ij
    # grad_j +=  2 * v_ij * (c_i - c_j) / d_ij
    
    # We need to accumulate into the grad array.
    # Vectorized accumulation is tricky with variable indices.
    # Let's loop over the violating pairs only.
    viol_mask = pair_violations > 0
    if np.any(viol_mask):
        viol_i = i_idx[viol_mask]
        viol_j = j_idx[viol_mask]
        viol_v = pair_violations[viol_mask]
        viol_d = pair_dists[viol_mask]
        viol_diff = diff[viol_i, viol_j] # Shape (num_viol, 3)
        
        # Factor: 2 * v / d
        factors = 2.0 * viol_v / viol_d
        
        # Update grad for i: -factor * diff
        # Update grad for j: +factor * diff
        
        # Use np.add.at for unbuffered addition to handle duplicate indices (though here i!=j always)
        np.add.at(grad, viol_i, -factors * viol_diff)
        np.add.at(grad, viol_j, factors * viol_diff)
        
    return energy, grad

def _optimize_for_radius(centers, r_target, max_iter=200, time_limit=10.0):
    """
    Try to find a configuration where radius >= r_target.
    Returns (success, new_centers, final_energy)
    """
    start_time = time.time()
    
    # Flatten centers for scipy
    x0 = centers.flatten()
    
    def fun(x):
        c = x.reshape(N, 3)
        e, _ = _energy_and_gradient(c, r_target)
        return e
    
    def jac(x):
        c = x.reshape(N, 3)
        _, g = _energy_and_gradient(c, r_target)
        return g.flatten()
    
    # Bounds: centers must be in [-0.5, 0.5]
    # But actually, if they go outside, wall violation increases.
    # It's safer to constrain them to [-0.5+r_target, 0.5-r_target]?
    # No, we want to allow them to move anywhere, the penalty will push them in.
    # However, L-BFGS-B works best with bounds.
    # Let's use bounds [-0.5, 0.5].
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    try:
        res = minimize(
            fun, x0, jac=jac, method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        # Check time
        if time.time() - start_time > time_limit:
            return False, centers, fun(x0)
            
        final_centers = res.x.reshape(N, 3)
        final_energy = res.fun
        
        # Check if energy is small enough
        # If energy < 1e-6, we consider it a success
        if final_energy < 1e-6:
            return True, final_centers, final_energy
        else:
            return False, final_centers, final_energy
            
    except Exception:
        return False, centers, 1e10

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Initial positions: Random uniform
    # A better init might be a grid, but random is fine if we optimize.
    np.random.seed(42) # For reproducibility in testing, remove for final? No, keep it for consistent behavior.
    pos = np.random.uniform(-0.4, 0.4, (N, 3))
    
    best_pos = pos.copy()
    best_r = _compute_radius(best_pos)
    
    # Binary search for the maximum radius
    # Lower bound: 0
    # Upper bound: 0.5 (max possible)
    # But 0.5 is impossible for N=100.
    # Let's start with a reasonable guess.
    # Specht's radius is ~0.111.
    # Let's search between 0.05 and 0.15.
    
    lo = 0.05
    hi = 0.15
    
    # First, try to improve the initial radius if it's below lo
    # If best_r < lo, set lo = best_r
    
    # Iterative refinement
    # We will do several passes of binary search.
    # Each pass tries to find a feasible solution for a given r.
    
    # To speed up, we can use the result of a previous success as the start for the next.
    
    # Let's do a coarse binary search first.
    # 10 iterations of binary search: range 0.1 / 2^10 ~ 1e-4 precision.
    
    for iteration in range(12):
        if time.time() > deadline - 5.0:
            break
            
        mid = (lo + hi) / 2.0
        
        # Try to optimize for radius mid
        success, new_pos, energy = _optimize_for_radius(best_pos, mid, max_iter=100, time_limit=5.0)
        
        if success:
            # Update best
            best_pos = new_pos
            # Recalculate actual radius to be safe
            actual_r = _compute_radius(best_pos)
            if actual_r > best_r:
                best_r = actual_r
            lo = mid
        else:
            hi = mid
            
    # After binary search, do a local polish with a slightly lower radius to ensure stability?
    # Or just return best_pos.
    
    # One final check
    final_r = _compute_radius(best_pos)
    if final_r > best_r:
        best_r = final_r
        
    return best_pos
