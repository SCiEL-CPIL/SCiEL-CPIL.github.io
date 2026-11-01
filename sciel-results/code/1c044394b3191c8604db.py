"""
Strategy: Differentiable optimization using L-BFGS-B on a log-sum-exp soft-min objective,
followed by a Newton-Raphson polish on the exact minimum distance.
"""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import cKDTree

def _score(c):
    """Compute the largest common radius for centres c (shape (N,3))."""
    if c.shape != (N, 3):
        return -1.0
    if not np.all(np.isfinite(c)):
        return -1.0
    # distance to walls
    wall = np.min(np.min(0.5 - np.abs(c), axis=1), axis=0)
    # pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_dist = np.min(dist)
    return min(wall, min_dist / 2.0)

def _init_grid():
    """Simple cubic grid as starting point."""
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].copy()

def _softmin_val(x, beta):
    """
    Compute the soft-minimum of pairwise distances and wall distances.
    x: flattened array of shape (N*3,)
    beta: temperature parameter for log-sum-exp (higher = smoother)
    Returns: -soft_min_distance (we minimize this, so we maximize soft_min)
    """
    c = x.reshape(N, 3)
    
    # Wall distances
    # d_wall_i = min_k (0.5 - |c_ik|)
    # We want to maximize the minimum wall distance.
    # Soft max of wall distances? Or soft min of (0.5 - wall_dist)?
    # Let's use a soft-min on the actual distances.
    
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    
    # We need a smooth approximation of min(dist).
    # min(dist) = -max(-dist).
    # Let's define a potential that is high when distances are small.
    # Objective: minimize F(x) = - (beta * log(sum(exp(-dist^2 / beta)))) ?
    # Actually, standard soft-min for vector v is:
    # m_beta(v) = -1/beta * log( sum(exp(-v_i / beta)) )
    # This approximates min(v_i) as beta -> 0.
    
    # Let's work with squared distances for efficiency in gradient?
    # No, let's stick to distances for clarity, but compute efficiently.
    
    # 1. Wall terms
    # w_i = 0.5 - |c_i| (component-wise)
    # wall_dist_i = min_k w_ik
    # We want to maximize min(wall_dist_i, min_pair_dist_ij/2).
    # Let's maximize the soft-min of all constraints.
    
    # Constraints:
    # 1. For each i, k: 0.5 - |c_ik| >= r
    # 2. For each i<j: dist_ij >= 2r
    
    # We want to maximize r.
    # Equivalent to minimizing -r.
    # Let's define a scalar objective that penalizes violations of a target radius?
    # No, let's just maximize the soft-minimum of all "available space" metrics.
    
    # Metric 1: Wall clearance for sphere i: w_i = min_k (0.5 - |c_ik|)
    # Metric 2: Half distance to nearest neighbor for sphere i: h_i = 0.5 * min_{j!=i} dist_ij
    
    # We want to maximize min( min_i w_i, min_i h_i ).
    # Let V be the vector of all 4N wall clearances (N spheres * 3 axes * 2 sides? No, min over axes handles it) 
    # Actually, for sphere i, the distance to the boundary is min(0.5-c_ix, 0.5+c_ix, ...).
    # Let's compute the vector of all 6N wall distances (each axis, both sides) and all N(N-1)/2 pairwise half-distances.
    
    # To make it efficient, we can just use the N wall minimums and N nearest neighbor half-distances.
    # But soft-min needs all terms.
    
    # Let's construct the vector of "clearances".
    # Wall clearances:
    # For each i, dist to +x wall: 0.5 - c_i[0]
    # dist to -x wall: 0.5 + c_i[0]
    # etc.
    
    walls = np.empty((N, 6))
    walls[:, 0] = 0.5 - c[:, 0]
    walls[:, 1] = 0.5 + c[:, 0]
    walls[:, 2] = 0.5 - c[:, 1]
    walls[:, 3] = 0.5 + c[:, 1]
    walls[:, 4] = 0.5 - c[:, 2]
    walls[:, 5] = 0.5 + c[:, 2]
    
    # Pairwise distances (upper triangle)
    # This is O(N^2) memory and compute. For N=100, 100*99/2 = 4950 terms. Manageable.
    i_idx, j_idx = np.triu_indices(N, k=1)
    d_ij = c[i_idx] - c[j_idx]
    dist_ij = np.linalg.norm(d_ij, axis=1)
    half_dists = 0.5 * dist_ij
    
    # Combine all clearances into one vector
    clearances = np.concatenate([walls.flatten(), half_dists])
    
    # Soft-min:
    # m_beta = -1/beta * log( sum(exp(-clearances / beta)) )
    # We want to MAXIMIZE m_beta.
    # So we MINIMIZE -m_beta = 1/beta * log( sum(exp(-clearances / beta)) )
    
    # Numerical stability: subtract max clearance?
    # exp(-c/beta). If c is large, exp is small. If c is small (close to 0), exp is large.
    # To avoid overflow, we can shift.
    # Let c_max = max(clearances).
    # sum(exp(-(c_i - c_max)/beta)) * exp(-c_max/beta)
    # log(sum) = log(sum_exp_shifted) - c_max/beta
    # m_beta = -1/beta * (log(sum_exp_shifted) - c_max/beta) = c_max/beta - 1/beta*log(sum_exp_shifted)
    # Objective to minimize: -m_beta = 1/beta*log(sum_exp_shifted) - c_max/beta
    
    c_max = np.max(clearances)
    shifted = -clearances / beta + c_max / beta
    # sum(exp(shifted))
    # If beta is small, shifted terms for min(clearances) will be large (close to 0), others negative.
    # Actually, -c_i/beta + c_max/beta = (c_max - c_i)/beta.
    # For i where c_i is min, c_max - c_i is max, so exp is large.
    # This is correct.
    
    # To prevent overflow in exp, if beta is small, (c_max - c_min)/beta can be large.
    # But exp(large) is fine in float64 up to ~700.
    # If beta is 0.01, and diff is 0.1, 0.1/0.01 = 10. exp(10) ~ 22000. Fine.
    
    sum_exp = np.sum(np.exp(shifted))
    log_sum = np.log(sum_exp)
    
    val = (log_sum - c_max / beta) / beta
    # Wait, let's re-derive.
    # m_beta = -1/beta * log( sum(exp(-c_i/beta)) )
    # Let A = sum(exp(-c_i/beta)).
    # log A = log( exp(-c_max/beta) * sum(exp(-(c_i-c_max)/beta)) )
    # log A = -c_max/beta + log( sum(exp((c_max-c_i)/beta)) )
    # m_beta = -1/beta * ( -c_max/beta + log(sum_shifted) )
    # m_beta = c_max/beta^2 - 1/beta * log(sum_shifted)  <-- This dimension is wrong.
    
    # Let's stick to the standard formula:
    # m_beta(v) = -1/beta * log( sum_i exp(-v_i / beta) )
    # Objective to minimize: f(x) = -m_beta(v(x)) = 1/beta * log( sum_i exp(-v_i / beta) )
    
    # Numerical stability:
    # sum_i exp(-v_i/beta) = exp(-v_max/beta) * sum_i exp(-(v_i-v_max)/beta)
    # log(sum) = -v_max/beta + log( sum_i exp((v_max-v_i)/beta) )
    # f(x) = 1/beta * ( -v_max/beta + log( sum_i exp((v_max-v_i)/beta) ) )
    # f(x) = -v_max/beta^2 + 1/beta * log( sum_i exp((v_max-v_i)/beta) )
    
    # This looks complicated. Let's just use the stable log-sum-exp function directly.
    # logsumexp(-clearances / beta)
    # f(x) = 1/beta * logsumexp(-clearances / beta)
    
    # scipy.special.logsumexp is available? Yes.
    from scipy.special import logsumexp
    
    val = (1.0 / beta) * logsumexp(-clearances / beta)
    
    return val

def _softmin_grad(x, beta):
    """
    Gradient of the soft-min objective.
    f(x) = 1/beta * log( sum(exp(-v_i(x)/beta)) )
    df/dx = 1/beta * 1/sum(exp) * sum( exp(-v_i/beta) * (-1/beta) * dv_i/dx )
    df/dx = -1/beta^2 * sum( p_i * dv_i/dx )
    where p_i = exp(-v_i/beta) / sum(exp(-v_j/beta))
    """
    c = x.reshape(N, 3)
    
    # Recompute clearances and weights
    walls = np.empty((N, 6))
    walls[:, 0] = 0.5 - c[:, 0]
    walls[:, 1] = 0.5 + c[:, 0]
    walls[:, 2] = 0.5 - c[:, 1]
    walls[:, 3] = 0.5 + c[:, 1]
    walls[:, 4] = 0.5 - c[:, 2]
    walls[:, 5] = 0.5 + c[:, 2]
    
    i_idx, j_idx = np.triu_indices(N, k=1)
    d_ij = c[i_idx] - c[j_idx]
    dist_ij = np.linalg.norm(d_ij, axis=1)
    half_dists = 0.5 * dist_ij
    
    clearances = np.concatenate([walls.flatten(), half_dists])
    
    # Weights p_i
    # Use log-sum-exp for stability
    from scipy.special import logsumexp
    
    log_weights = -clearances / beta
    log_sum = logsumexp(log_weights)
    weights = np.exp(log_weights - log_sum)
    
    # Gradient of clearances with respect to x
    # grad_x is shape (N, 3)
    grad = np.zeros((N, 3))
    
    # 1. Wall gradients
    # v_wall_ik = 0.5 - c_i[k] or 0.5 + c_i[k]
    # dv/dc = -1 or +1
    # weights are in clearances[0:6N]
    w_weights = weights[:6*N].reshape(N, 6)
    
    # For each sphere i:
    # d(0.5 - c_i[0])/dc_i[0] = -1
    # d(0.5 + c_i[0])/dc_i[0] = +1
    # d(0.5 - c_i[1])/dc_i[1] = -1
    # d(0.5 + c_i[1])/dc_i[1] = +1
    # d(0.5 - c_i[2])/dc_i[2] = -1
    # d(0.5 + c_i[2])/dc_i[2] = +1
    
    # Sum over the 6 wall terms for each sphere
    # grad_i[0] += w_weights[i,0]*(-1) + w_weights[i,1]*(1)
    # grad_i[1] += w_weights[i,2]*(-1) + w_weights[i,3]*(1)
    # grad_i[2] += w_weights[i,4]*(-1) + w_weights[i,5]*(1)
    
    grad[:, 0] = -w_weights[:, 0] + w_weights[:, 1]
    grad[:, 1] = -w_weights[:, 2] + w_weights[:, 3]
    grad[:, 2] = -w_weights[:, 4] + w_weights[:, 5]
    
    # 2. Pairwise gradients
    # v_pair_ij = 0.5 * ||c_i - c_j||
    # dv_pair_ij / dc_i = 0.5 * (c_i - c_j) / ||c_i - c_j||
    # dv_pair_ij / dc_j = 0.5 * (c_j - c_i) / ||c_i - c_j||
    
    p_weights = weights[6*N:]
    
    # d_ij is shape (M, 3) where M = N(N-1)/2
    # dist_ij is shape (M,)
    # Avoid division by zero
    safe_dist = np.where(dist_ij < 1e-8, 1e-8, dist_ij)
    unit_vecs = d_ij / safe_dist[:, np.newaxis]
    
    # grad_i += 0.5 * unit_vec * weight
    # grad_j -= 0.5 * unit_vec * weight
    
    # We need to accumulate these into the grad array
    # np.add.at is useful for this
    np.add.at(grad, i_idx, 0.5 * p_weights[:, np.newaxis] * unit_vecs)
    np.add.at(grad, j_idx, -0.5 * p_weights[:, np.newaxis] * unit_vecs)
    
    # Total gradient of f(x) = -1/beta^2 * sum(p_i * dv_i/dx)
    # Note: The derivative of logsumexp(-v/beta) w.r.t x is:
    # sum( p_i * (-1/beta) * dv_i/dx )
    # So df/dx = 1/beta * sum( p_i * (-1/beta) * dv_i/dx ) = -1/beta^2 * sum( p_i * dv_i/dx )
    
    grad = -1.0 / (beta * beta) * grad
    
    return grad.flatten()

def _lfgs(c0, time_limit):
    """Run L-BFGS-B optimization."""
    x0 = c0.flatten()
    start_time = time.time()
    
    # Beta schedule: start high (smooth), decrease (sharp)
    betas = [0.1, 0.05, 0.02, 0.01, 0.005]
    
    current_x = x0.copy()
    
    for beta in betas:
        if time.time() > time_limit - 5.0:
            break
            
        res = minimize(
            _softmin_val, 
            current_x, 
            method='L-BFGS-B', 
            jac=_softmin_grad, 
            bounds=[(-0.5, 0.5)] * (N * 3),
            args=(beta,),
            options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8, 'disp': False}
        )
        
        current_x = res.x
        # Ensure we stay within bounds (L-BFGS-B should, but just in case)
        current_x = np.clip(current_x, -0.5, 0.5)
        
    return current_x.reshape(N, 3)

def _polish(c, time_limit):
    """
    Newton-Raphson style polishing.
    Identify the most constrained pairs and walls, and move them apart.
    """
    c = c.copy()
    start_time = time.time()
    
    for _ in range(2000):
        if time.time() > time_limit:
            break
            
        # Find current min radius
        r = _score(c)
        
        # Find the limiting constraint
        # 1. Wall check
        wall_dists = np.min(0.5 - np.abs(c), axis=1)
        min_wall_idx = np.argmin(wall_dists)
        min_wall_dist = wall_dists[min_wall_idx]
        
        # 2. Pairwise check
        tree = cKDTree(c)
        d, idx = tree.query(c, k=2)
        nn_dist = d[:, 1]
        min_nn_idx = np.argmin(nn_dist)
        min_nn_dist = nn_dist[min_nn_idx]
        min_nn_rad = min_nn_dist / 2.0
        
        # Determine which is limiting
        if min_wall_rad < min_nn_rad:
            # Limiting is wall
            # Move the sphere away from the wall
            i = min_wall_idx
            # Find which axis is closest
            c_i = c[i]
            dists_to_walls = np.abs(c_i) # distance to center, so 0.5 - dist is clearance
            # Actually, we need the specific wall.
            # Let's just push it towards the center of the cube.
            # Or push it away from the specific wall it's closest to.
            # Closest wall axis:
            ax = np.argmin(np.abs(c_i))
            sign = np.sign(c_i[ax])
            if sign == 0: sign = 1
            
            # Move away from wall
            step = 0.001
            new_c = c.copy()
            new_c[i, ax] = c[i, ax] - sign * step
            new_c[i, ax] = np.clip(new_c[i, ax], -0.5, 0.5)
            
            if _score(new_c) > r:
                c = new_c
            else:
                # Try smaller step
                step = 0.0001
                new_c = c.copy()
                new_c[i, ax] = c[i, ax] - sign * step
                new_c[i, ax] = np.clip(new_c[i, ax], -0.5, 0.5)
                if _score(new_c) > r:
                    c = new_c
                else:
                    break # Stuck?
        else:
            # Limiting is pair
            i = min_nn_idx
            j = idx[i, 1]
            
            diff = c[i] - c[j]
            dist = np.linalg.norm(diff)
            if dist < 1e-8:
                # Random nudge
                c[i] += np.random.normal(0, 0.001, 3)
                c[i] = np.clip(c[i], -0.5, 0.5)
                continue
                
            direction = diff / dist
            
            # Move i and j apart
            step = 0.001
            new_c = c.copy()
            new_c[i] = c[i] + step * direction
            new_c[j] = c[j] - step * direction
            new_c[i] = np.clip(new_c[i], -0.5, 0.5)
            new_c[j] = np.clip(new_c[j], -0.5, 0.5)
            
            if _score(new_c) > r:
                c = new_c
            else:
                # Try moving just one
                for idx_move in [i, j]:
                    for sign_move in [1, -1]:
                        if idx_move == i:
                            d_move = direction
                        else:
                            d_move = -direction
                            
                        new_c = c.copy()
                        new_c[idx_move] = c[idx_move] + step * sign_move * d_move
                        new_c[idx_move] = np.clip(new_c[idx_move], -0.5, 0.5)
                        
                        if _score(new_c) > r:
                            c = new_c
                            break
                    else:
                        continue
                    break
                else:
                    break # Stuck
    
    return c

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin
    
    # 1. Initialize
    c0 = _init_grid().copy()
    
    # 2. L-BFGS-B Optimization
    c_opt = _lfgs(c0, deadline - 10.0)
    
    # 3. Polish
    c_final = _polish(c_opt, deadline - 2.0)
    
    # 4. Final check and return
    r = _score(c_final)
    if r <= 0:
        return c0 # Fallback
    
    return c_final
# EVOLVE-BLOCK-END
