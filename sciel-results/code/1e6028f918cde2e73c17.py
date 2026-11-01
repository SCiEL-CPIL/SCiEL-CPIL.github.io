import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _compute_soft_min(points, beta):
    """
    Computes the soft-min of pairwise distances/2 and wall distances.
    Returns (value, gradient).
    """
    x = points[:, 0]
    y = points[:, 1]
    
    # Pairwise distances
    # diff[i, j] = x_i - x_j
    dx = x[:, np.newaxis] - x[np.newaxis, :]
    dy = y[:, np.newaxis] - y[np.newaxis, :]
    dists = np.sqrt(dx**2 + dy**2)
    
    # Mask diagonal
    np.fill_diagonal(dists, 0.0)
    
    # We only want i < j
    i_idx, j_idx = np.triu_indices(N, k=1)
    pair_dists = dists[i_idx, j_idx]
    
    # Wall distances
    wall_x = 0.5 - np.abs(x)
    wall_y = 0.5 - np.abs(y)
    
    # Combine all values into a single vector
    # Values: pair_dists/2, wall_x, wall_y
    vals = np.concatenate([pair_dists / 2.0, wall_x, wall_y])
    
    # Soft min calculation
    # f = -1/beta * log(sum(exp(-beta * vals)))
    # To avoid overflow/underflow, subtract max
    max_val = np.max(vals)
    exp_terms = np.exp(-beta * (vals - max_val))
    sum_exp = np.sum(exp_terms)
    log_sum = np.log(sum_exp)
    
    value = -1.0 / beta * (log_sum - beta * max_val)
    # Wait, the formula is:
    # softmin = -1/beta * log( sum exp(-beta v) )
    # = -1/beta * ( log(sum exp(-beta(v - v_max))) - beta v_max )
    # = -1/beta * log(sum exp(-beta(v - v_max))) + v_max
    value = v_max - (1.0 / beta) * np.log(sum_exp)
    
    # Probabilities p_k = exp(-beta v_k) / sum_exp
    # Note: we computed exp(-beta(v - v_max)), so p_k = exp_terms / sum_exp
    p = exp_terms / sum_exp
    
    # Gradient calculation
    grad = np.zeros_like(points)
    
    # Pairwise contributions
    # For each pair (i, j), val = dist/2
    # p_pair = p[0 : len(pair_dists)]
    p_pairs = p[:len(pair_dists)]
    
    # grad_x_i += p * (x_j - x_i) / (2 * dist)
    # grad_x_j += p * (x_i - x_j) / (2 * dist)
    
    # We need to accumulate gradients
    # dists[i,j] is the distance.
    # Avoid division by zero
    safe_dists = np.where(dists > 1e-12, dists, 1.0)
    
    # Factor for gradient of dist/2 wrt x_i is (x_i - x_j) / (2 * dist)
    # So grad contribution is -p * (x_i - x_j) / (2 * dist) for f?
    # Recall: df/dv = -p.
    # dv/dx_i = (x_i - x_j) / (2 dist).
    # df/dx_i = -p * (x_i - x_j) / (2 dist).
    
    # We want grad of g = -f. So grad_g = -grad_f = p * (x_i - x_j) / (2 dist).
    
    # Let's compute the vector contribution for all pairs.
    # This is a bit tricky to vectorize fully without loops or advanced indexing.
    # Given N=50, a loop over pairs (1225) is fast enough.
    
    for k in range(len(pair_dists)):
        i = i_idx[k]
        j = j_idx[k]
        p_k = p_pairs[k]
        d = dists[i, j]
        if d < 1e-12:
            continue
        factor = p_k / (2.0 * d)
        # grad_x_i += factor * (x_i - x_j)
        grad[i, 0] += factor * (x[i] - x[j])
        grad[j, 0] += factor * (x[j] - x[i])
        # grad_y_i += factor * (y_i - y_j)
        grad[i, 1] += factor * (y[i] - y[j])
        grad[j, 1] += factor * (y[j] - y[i])
        
    # Wall contributions
    # p_walls = p[len(pair_dists):]
    # First N are wall_x, next N are wall_y
    p_wall_x = p[len(pair_dists):len(pair_dists)+N]
    p_wall_y = p[len(pair_dists)+N:]
    
    # w_x = 0.5 - |x|. dw_x/dx = -sign(x).
    # df/dx = -p * (-sign(x)) = p * sign(x).
    # grad_g_x = -df/dx = -p * sign(x).
    
    signs_x = np.sign(x)
    # Handle x=0, sign is 0, which is fine (subgradient 0)
    grad[:, 0] -= p_wall_x * signs_x
    
    signs_y = np.sign(y)
    grad[:, 1] -= p_wall_y * signs_y
    
    return value, grad

def _objective_and_grad(flat_points, beta):
    points = flat_points.reshape(N, 2)
    val, grad = _compute_soft_min(points, beta)
    # We are minimizing -val
    return -val, -grad

def _compute_exact_radius(points):
    if not np.all(np.isfinite(points)):
        return 0.0
    if np.any(points < -0.5) or np.any(points > 0.5):
        return 0.0
    
    x = points[:, 0]
    y = points[:, 1]
    
    dx = x[:, np.newaxis] - x[np.newaxis, :]
    dy = y[:, np.newaxis] - y[np.newaxis, :]
    dists = np.sqrt(dx**2 + dy**2)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    wall_dists = np.min(np.stack([0.5 - np.abs(x), 0.5 - np.abs(y)], axis=1), axis=1)
    min_wall_dist = np.min(wall_dists)
    
    return min(min_pair_dist / 2.0, min_wall_dist)

def construct():
    start_time = time.time()
    deadline = 140.0  # 10s safety margin
    
    # Initial configuration: Jittered grid
    m = 7
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    
    points = grid.copy()
    # Add 50th point at center (0,0) is a common spot, but grid already has near center?
    # 7x7 grid centers: -3/7, -2/7, -1/7, 0, 1/7, 2/7, 3/7.
    # So (0,0) is already in the grid.
    # We need 50 points. Grid has 49.
    # Let's add a point at (0, 0.5) or somewhere else?
    # Actually, just adding a random point and letting optimizer fix it is okay.
    # Or we can use a 8x8 grid and remove 14 points? No, 7x7 is better aspect ratio.
    # Let's place the 50th point at (0,0) and shift the existing (0,0) point slightly?
    # Or just place it randomly.
    p50 = np.random.uniform(-0.5, 0.5, 2)
    points = np.vstack([points, p50])
    
    # Jitter
    points += np.random.normal(0, 0.005, points.shape)
    points = np.clip(points, -0.5, 0.5)
    
    best_points = points.copy()
    best_score = _compute_exact_radius(best_points)
    
    # Optimization loop with increasing beta
    betas = [5.0, 10.0, 20.0, 50.0, 100.0, 200.0]
    
    current_points = best_points.copy()
    
    for beta in betas:
        if time.time() - start_time > deadline:
            break
            
        # Flatten points for scipy
        flat_points = current_points.flatten()
        
        try:
            res = minimize(
                _objective_and_grad,
                flat_points,
                args=(beta,),
                jac=True,
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (2 * N),
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            new_points = res.x.reshape(N, 2)
            new_score = _compute_exact_radius(new_points)
            
            if new_score > best_score:
                best_score = new_score
                best_points = new_points.copy()
                current_points = new_points.copy()
                
        except Exception as e:
            # If optimization fails, keep current best
            pass
            
    # Final polish: Try to improve with a very high beta if time remains
    if time.time() - start_time < deadline - 5:
        beta = 500.0
        flat_points = current_points.flatten()
        try:
            res = minimize(
                _objective_and_grad,
                flat_points,
                args=(beta,),
                jac=True,
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (2 * N),
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            new_points = res.x.reshape(N, 2)
            new_score = _compute_exact_radius(new_points)
            if new_score > best_score:
                best_score = new_score
                best_points = new_points.copy()
        except Exception:
            pass

    return best_points
