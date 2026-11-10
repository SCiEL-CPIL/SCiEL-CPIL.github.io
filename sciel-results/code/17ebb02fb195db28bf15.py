"""
Improved solver for packing 100 spheres in a unit cube.
Uses a smooth inverse-power surrogate objective to maximize the minimum radius.
Employs multi-start L-BFGS-B optimization with annealing of the power parameter.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 100
CUBE_HALF = 0.5

def _min_radius(centers):
    """Compute the maximum common radius for given centers inside [-0.5, 0.5]^3."""
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Check inside cube with a small tolerance
    if np.any(centers < -CUBE_HALF - 1e-12) or np.any(centers > CUBE_HALF + 1e-12):
        return 0.0
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    # Distance to walls
    wall_dists = CUBE_HALF - np.abs(centers)
    min_wall = np.min(wall_dists)
    
    r = min(min_pair / 2.0, min_wall)
    return max(r, 0.0)

def _smooth_objective(x, p):
    """
    Objective function: Sum of inverse powers of constraint values.
    Minimizing this approximates maximizing the minimum constraint value.
    Constraint values: d_ij/2 for pairs, w_i for walls.
    """
    c = x.reshape(N, 3)
    
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0) # Avoid division by zero for self, will be masked
    dists = np.sqrt(dist_sq)
    
    # Constraint values for pairs: d_ij / 2
    pair_vals = dists / 2.0
    np.fill_diagonal(pair_vals, 1.0) # Placeholder, will be ignored
    
    # Constraint values for walls: 0.5 - |c_i_k|
    # We take the minimum over axes for each sphere, but for the sum we need to be careful.
    # Actually, the constraint is min_k (0.5 - |c_ik|). 
    # To smooth this, we can use the soft-min over axes for each sphere, or just use the raw wall distances 
    # and rely on the fact that the minimum will dominate. 
    # A better approach: use the minimum wall distance per sphere as the value.
    # But min is not differentiable. 
    # Alternative: use a soft-min over the 3 wall distances for each sphere.
    # soft_min(v1, v2, v3) = -(1/p) * log(sum(exp(-p*v_k)))
    # Or simpler: just use the raw wall distances in the sum? 
    # If we use sum(w_i^{-p}), it will push all walls away, not just the closest one.
    # This is actually fine and often leads to a more uniform distribution, which is good.
    # Let's stick to summing over all wall distances (3 per sphere) and all pair distances.
    
    wall_dists = CUBE_HALF - np.abs(c) # Shape (N, 3)
    
    # Calculate sum of inverse powers
    # Pair term
    mask = np.ones((N, N), dtype=bool)
    np.fill_diagonal(mask, False)
    pair_vals_masked = np.where(mask, pair_vals, 1.0)
    term_pairs = np.sum(pair_vals_masked ** (-p))
    
    # Wall term
    # Ensure wall_dists are positive (they should be if inside cube)
    # Clip to a small positive number to avoid inf
    wall_dists_clipped = np.maximum(wall_dists, 1e-6)
    term_walls = np.sum(wall_dists_clipped ** (-p))
    
    return term_pairs + term_walls

def _smooth_gradient(x, p):
    """
    Gradient of the smooth objective.
    d/dx sum(v^{-p}) = sum(-p * v^{-p-1} * dv/dx)
    """
    c = x.reshape(N, 3)
    
    # Pairwise
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    dists = np.sqrt(dist_sq)
    
    pair_vals = dists / 2.0
    np.fill_diagonal(pair_vals, 1.0)
    
    # v = d/2, dv/dc_i = 0.5 * (c_i - c_j) / d
    # term = v^{-p} = (d/2)^{-p}
    # d(term)/d(c_i) = -p * (d/2)^{-p-1} * 0.5 * (c_i - c_j)/d
    #                = -p * 0.5 * (d/2)^{-p-1} * (c_i - c_j)/d
    #                = -p * 0.5 * (d/2)^{-p-1} * (c_i - c_j) * (2/d)
    #                = -p * (d/2)^{-p-1} * (c_i - c_j)
    
    # Let's compute weights
    # w_ij = p * (d_ij/2)^{-p-1}
    # grad_pair[i] = - sum_j w_ij * (c_i - c_j)
    
    # Avoid division by zero
    inv_pair_vals = np.zeros_like(pair_vals)
    mask = np.ones((N, N), dtype=bool)
    np.fill_diagonal(mask, False)
    inv_pair_vals[mask] = pair_vals[mask] ** (-p - 1)
    
    weights = p * inv_pair_vals
    np.fill_diagonal(weights, 0.0)
    
    # grad_pair[i] = - sum_j weights[i,j] * (c_i - c_j)
    #              = - (c_i * sum_j weights[i,j] - sum_j weights[i,j]*c_j)
    
    W_sum = np.sum(weights, axis=1) # (N,)
    Wc = weights.dot(c) # (N, 3)
    
    grad_pairs = - (c * W_sum[:, np.newaxis] - Wc)
    
    # Walls
    # term = w^{-p}, w = 0.5 - |c_k|
    # d(term)/d(c_k) = -p * w^{-p-1} * dw/dc_k
    # dw/dc_k = -sign(c_k)
    # So d(term)/d(c_k) = p * w^{-p-1} * sign(c_k)
    
    wall_dists = CUBE_HALF - np.abs(c)
    wall_dists_clipped = np.maximum(wall_dists, 1e-6)
    
    # inv_wall = wall_dists^{-p-1}
    inv_walls = wall_dists_clipped ** (-p - 1)
    
    signs = np.sign(c)
    # If c_k is 0, sign is 0. Gradient is undefined, but we can push away from center? 
    # Actually, if c_k=0, w=0.5. dw/dc_k is undefined. 
    # But the function is smooth if we use |c|. 
    # Let's just use the sign. If 0, set to 1 arbitrarily (push to positive).
    signs[signs == 0] = 1.0
    
    grad_walls = p * inv_walls * signs
    
    total_grad = grad_pairs + grad_walls
    
    return total_grad.reshape(N * 3)

def construct():
    start_time = time.time()
    deadline = 145.0 # 5s safety margin
    
    best_centers = None
    best_r = 0.0
    
    # Multi-start
    num_starts = 5
    rng = np.random.default_rng(42)
    
    for start_idx in range(num_starts):
        if time.time() > deadline - 20:
            break
            
        # Generate initial configuration
        # Use a jittered grid or random
        if start_idx == 0:
            # Jittered grid 5x5x4 = 100
            m_x, m_y, m_z = 5, 5, 4
            xs = (np.arange(m_x) + 0.5) / m_x - 0.5
            ys = (np.arange(m_y) + 0.5) / m_y - 0.5
            zs = (np.arange(m_z) + 0.5) / m_z - 0.5
            grid = np.array([(x, y, z) for z in zs for y in ys for x in xs])
            centers = grid[:N].astype(np.float64)
            centers += rng.normal(0, 0.01, size=centers.shape)
        else:
            # Random points, clustered slightly to help optimization? 
            # Or just uniform random
            centers = rng.uniform(-0.45, 0.45, size=(N, 3))
        
        centers = np.clip(centers, -0.49, 0.49)
        x = centers.flatten()
        
        # Annealing of p
        ps = [4, 8, 16, 32, 64]
        
        for p in ps:
            if time.time() > deadline - 10:
                break
                
            try:
                def obj(x_curr, p_val=p):
                    return _smooth_objective(x_curr, p_val)
                
                def grad(x_curr, p_val=p):
                    return _smooth_gradient(x_curr, p_val)
                
                # L-BFGS-B
                res = minimize(obj, x, jac=grad, method='L-BFGS-B',
                               bounds=[(-0.4999, 0.4999)] * (N * 3),
                               options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-6, 'maxls': 50})
                
                if res.fun < 1e10:
                    x_new = np.clip(res.x, -0.4999, 0.4999)
                    c_new = x_new.reshape(N, 3)
                    r_new = _min_radius(c_new)
                    
                    if r_new > best_r:
                        best_r = r_new
                        best_centers = c_new.copy()
                    
                    x = x_new
            except Exception as e:
                pass
        
        # Final polish on the best candidate from this start
        if best_centers is not None and time.time() < deadline - 5:
            x = best_centers.flatten()
            # Use a very high p for polishing
            p_final = 128
            try:
                def obj_final(x_curr):
                    return _smooth_objective(x_curr, p_final)
                def grad_final(x_curr):
                    return _smooth_gradient(x_curr, p_final)
                
                res = minimize(obj_final, x, jac=grad_final, method='L-BFGS-B',
                               bounds=[(-0.4999, 0.4999)] * (N * 3),
                               options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 100})
                
                if res.fun < 1e10:
                    x_new = np.clip(res.x, -0.4999, 0.4999)
                    c_new = x_new.reshape(N, 3)
                    r_new = _min_radius(c_new)
                    
                    if r_new > best_r:
                        best_r = r_new
                        best_centers = c_new.copy()
            except Exception:
                pass

    # If no valid result, return a simple grid
    if best_centers is None:
        m = 5
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_centers = grid[:N].astype(np.float64)
        best_centers = np.clip(best_centers, -0.49, 0.49)

    return best_centers
