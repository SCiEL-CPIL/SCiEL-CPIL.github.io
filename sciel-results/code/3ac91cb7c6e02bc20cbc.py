"""Rewritten program for packing 200 equal circles in a unit square."""
import time
import numpy as np
from scipy.optimize import minimize

N = 200

def _score(coords):
    """Compute the radius for a given set of coordinates."""
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    min_pair = np.min(dist[~np.eye(N, dtype=bool)])
    min_bound = np.min(np.minimum(np.minimum(0.5 - coords[:, 0], coords[:, 0] + 0.5),
                                   np.minimum(0.5 - coords[:, 1], coords[:, 1] + 0.5)))
    return min(min_pair / 2.0, min_bound)

def _soft_min_surrogate(coords, t, beta):
    """
    Soft-min surrogate for the packing radius.
    We want to maximize t subject to:
      - dist(i,j) >= 2*t for all i<j
      - boundary distances >= t for all i,k
    
    Using log-sum-exp for smooth min:
      soft_min(a, b, c, ...) = -log(exp(-beta*a) + exp(-beta*b) + ...) / beta
    
    We minimize: -t + penalty for violations
    """
    # Pairwise distances
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    
    # Get upper triangle indices
    iu = np.triu_indices(N, k=1)
    pair_dists = dist[iu]
    
    # Boundary distances
    d_left = coords[:, 0] + 0.5
    d_right = 0.5 - coords[:, 0]
    d_bottom = coords[:, 1] + 0.5
    d_top = 0.5 - coords[:, 1]
    bound_dists = np.concatenate([d_left, d_right, d_bottom, d_top])
    
    # All constraints: pair_dists should be >= 2*t, bound_dists should be >= t
    # Violations: 2*t - pair_dists (positive means violation)
    #             t - bound_dists (positive means violation)
    
    # Soft-min of the "slack" values:
    # slack_pair = pair_dists / 2 - t  (should be >= 0)
    # slack_bound = bound_dists - t  (should be >= 0)
    
    # We want to maximize the minimum slack
    # soft_min via log-sum-exp:
    # soft_min(s_1, ..., s_n) = -log(sum(exp(-beta*s_i))) / beta
    
    slacks_pair = pair_dists / 2.0 - t
    slacks_bound = bound_dists - t
    
    # Combine all slacks
    all_slacks = np.concatenate([slacks_pair, slacks_bound])
    
    # Numerically stable log-sum-exp
    min_s = np.min(all_slacks)
    if min_s < -1000:
        return -t + beta * (-min_s)  # Approximate
    exp_terms = np.exp(-beta * (all_slacks - min_s))
    log_sum = np.log(np.sum(exp_terms)) + (-beta * min_s)
    soft_min_val = -log_sum / beta
    
    # Objective: -soft_min_val (we want to maximize soft_min_val)
    # Plus a small penalty to encourage t to be close to the actual min
    return -soft_min_val

def _soft_min_grad(coords, t, beta):
    """Gradient of the soft-min surrogate objective."""
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    
    iu = np.triu_indices(N, k=1)
    pair_dists = dist[iu]
    
    d_left = coords[:, 0] + 0.5
    d_right = 0.5 - coords[:, 0]
    d_bottom = coords[:, 1] + 0.5
    d_top = 0.5 - coords[:, 1]
    bound_dists = np.concatenate([d_left, d_right, d_bottom, d_top])
    
    slacks_pair = pair_dists / 2.0 - t
    slacks_bound = bound_dists - t
    all_slacks = np.concatenate([slacks_pair, slacks_bound])
    
    # Soft-min gradient: 
    # d/dx_i [-log(sum(exp(-beta*s_j)))/beta] = sum_j [exp(-beta*s_j) * (-beta * ds_j/dx_i)] / (beta * sum(exp(-beta*s_j)))
    # = -sum_j [w_j * ds_j/dx_i] where w_j = exp(-beta*s_j) / sum(exp(-beta*s_k))
    
    min_s = np.min(all_slacks)
    exp_terms = np.exp(-beta * (all_slacks - min_s))
    weights = exp_terms / np.sum(exp_terms)
    
    w_pair = weights[:len(pair_dists)]
    w_bound = weights[len(pair_dists):]
    
    grad = np.zeros_like(coords)
    
    # Pairwise contributions
    # ds_pair/d(coords[i]) for pair (i,j): 
    # s = dist/2 - t, d(dist)/d(coords[i]) = (coords[i]-coords[j])/dist
    # d(s)/d(coords[i]) = (coords[i]-coords[j])/(2*dist)
    # d(s)/d(coords[j]) = (coords[j]-coords[i])/(2*dist)
    
    for idx in range(len(pair_dists)):
        i, j = iu[0][idx], iu[1][idx]
        d_ij = coords[i] - coords[j]
        dist_ij = pair_dists[idx]
        if dist_ij < 1e-12:
            continue
        # d(soft_min)/d(coords[i]) includes w_pair[idx] * d(s_pair[idx])/d(coords[i])
        # objective = -soft_min, so grad = -d(soft_min)/d(coords)
        # d(soft_min)/d(coords[i]) = -sum_j w_j * d(s_j)/d(coords[i])
        # So d(objective)/d(coords[i]) = sum_j w_j * d(s_j)/d(coords[i])
        
        ds_di = d_ij / (2.0 * dist_ij)
        ds_dj = -d_ij / (2.0 * dist_ij)
        
        grad[i] += w_pair[idx] * ds_di
        grad[j] += w_pair[idx] * ds_dj
    
    # Boundary contributions
    # d_left: d/d(coords[i,0]) = 1
    # d_right: d/d(coords[i,0]) = -1
    # d_bottom: d/d(coords[i,1]) = 1
    # d_top: d/d(coords[i,1]) = -1
    
    # w_bound[0:N] corresponds to d_left
    # w_bound[N:2N] corresponds to d_right
    # w_bound[2N:3N] corresponds to d_bottom
    # w_bound[3N:4N] corresponds to d_top
    
    for i in range(N):
        grad[i, 0] += w_bound[i] * 1.0        # d_left
        grad[i, 0] += w_bound[N + i] * (-1.0)  # d_right
        grad[i, 1] += w_bound[2*N + i] * 1.0   # d_bottom
        grad[i, 1] += w_bound[3*N + i] * (-1.0) # d_top
    
    # Gradient w.r.t. t:
    # d(soft_min)/dt = -sum_j w_j * d(s_j)/dt
    # d(s_pair)/dt = -1, d(s_bound)/dt = -1
    # d(soft_min)/dt = -sum_j w_j * (-1) = sum_j w_j = 1
    # d(objective)/dt = -1
    
    # But we're minimizing -soft_min, so the gradient w.r.t. the full state [coords, t]
    # We'll handle t separately in the optimizer
    
    return grad

def _overlap_energy(coords, t):
    """
    Smooth overlap energy for global exploration.
    Encourages circles to not overlap by penalizing when dist < 2*t.
    """
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    iu = np.triu_indices(N, k=1)
    pair_dists = dist[iu]
    
    # Penalty for overlaps: sum of max(0, 2*t - dist)^2
    violations = np.maximum(0, 2*t - pair_dists)
    energy = np.sum(violations ** 2)
    
    # Boundary violations
    d_left = coords[:, 0] + 0.5
    d_right = 0.5 - coords[:, 0]
    d_bottom = coords[:, 1] + 0.5
    d_top = 0.5 - coords[:, 1]
    bound_dists = np.concatenate([d_left, d_right, d_bottom, d_top])
    bound_violations = np.maximum(0, t - bound_dists)
    energy += np.sum(bound_violations ** 2)
    
    return energy

def _overlap_energy_grad(coords, t):
    """Gradient of overlap energy."""
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    iu = np.triu_indices(N, k=1)
    pair_dists = dist[iu]
    
    grad = np.zeros_like(coords)
    
    # Pairwise: d/d(coords[i]) sum max(0, 2t - dist)^2
    for idx in range(len(pair_dists)):
        i, j = iu[0][idx], iu[1][idx]
        d_ij = coords[i] - coords[j]
        dist_ij = pair_dists[idx]
        if dist_ij < 1e-12:
            continue
        violation = 2*t - dist_ij
        if violation > 0:
            # d(violation^2)/d(coords[i]) = 2*violation * d(violation)/d(coords[i])
            # d(violation)/d(coords[i]) = -d(dist)/d(coords[i]) = -(coords[i]-coords[j])/dist
            factor = 2 * violation * (-1.0) / (2.0 * dist_ij)
            grad[i] += factor * d_ij
            grad[j] -= factor * d_ij
    
    # Boundary
    d_left = coords[:, 0] + 0.5
    d_right = 0.5 - coords[:, 0]
    d_bottom = coords[:, 1] + 0.5
    d_top = 0.5 - coords[:, 1]
    
    for i in range(N):
        v = t - d_left[i]
        if v > 0:
            grad[i, 0] += 2 * v * (-1.0)
        v = t - d_right[i]
        if v > 0:
            grad[i, 0] += 2 * v * (1.0)
        v = t - d_bottom[i]
        if v > 0:
            grad[i, 1] += 2 * v * (-1.0)
        v = t - d_top[i]
        if v > 0:
            grad[i, 1] += 2 * v * (1.0)
    
    return grad

def _hex_initial(n_points, seed=42):
    """Generate initial hexagonal lattice points."""
    np.random.seed(seed)
    s_est = np.sqrt(2.0 / (n_points * np.sqrt(3.0)))
    points = []
    max_radius = 0.6
    for y_row in range(-50, 51):
        y = y_row * s_est * np.sqrt(3.0) / 2.0
        if abs(y) > max_radius:
            continue
        x_offset = (s_est / 2.0) if (y_row % 2 != 0) else 0.0
        for x_col in range(-50, 51):
            x = x_col * s_est + x_offset
            if abs(x) > max_radius or abs(y) > max_radius:
                continue
            points.append([x, y])
    
    points = np.array(points)
    if len(points) >= n_points:
        idx = np.random.choice(len(points), n_points, replace=False)
        return points[idx]
    else:
        # Pad with random points
        extra = np.random.uniform(-0.4, 0.4, (n_points - len(points), 2))
        return np.vstack([points, extra])

def _rect_initial(n_points, seed=42):
    """Generate initial rectangular grid points."""
    np.random.seed(seed)
    m = int(np.ceil(np.sqrt(n_points)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    if len(grid) >= n_points:
        idx = np.random.choice(len(grid), n_points, replace=False)
        return grid[idx]
    else:
        extra = np.random.uniform(-0.4, 0.4, (n_points - len(grid), 2))
        return np.vstack([grid, extra])

def _polish_soft_min(coords, t, beta, max_iter, deadline):
    """Polish using soft-min surrogate with increasing beta."""
    best_score = _score(coords)
    best_coords = coords.copy()
    best_t = t
    
    for beta_val in [beta, beta * 5, beta * 25, beta * 125]:
        if time.time() > deadline:
            break
        
        def objective(state):
            c = state[:2*N].reshape(N, 2)
            tt = state[2*N]
            return _soft_min_surrogate(c, tt, beta_val)
        
        def obj_grad(state):
            c = state[:2*N].reshape(N, 2)
            tt = state[2*N]
            gc = _soft_min_grad(c, tt, beta_val)
            # Gradient w.r.t. t: d(-soft_min)/dt = -1 (as derived)
            gt = -1.0
            return np.concatenate([gc.flatten(), [gt]])
        
        state = np.concatenate([coords.flatten(), [t]])
        
        try:
            result = minimize(objective, state, method='L-BFGS-B',
                             jac=obj_grad,
                             options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10,
                                     'maxfun': max_iter * 10})
            
            if result.fun < objective(state):
                new_coords = result.x[:2*N].reshape(N, 2)
                new_t = result.x[2*N]
                # Clamp coordinates
                new_coords = np.clip(new_coords, -0.4999, 0.4999)
                new_score = _score(new_coords)
                if new_score > best_score:
                    best_score = new_score
                    best_coords = new_coords
                    best_t = max(new_t, 0.01)
        except Exception:
            pass
        
        coords = best_coords
        t = best_t
    
    return best_coords, best_t, best_score

def _polish_overlap(coords, t, deadline):
    """Polish using overlap energy to find a good feasible region."""
    best_score = _score(coords)
    best_coords = coords.copy()
    
    # Decrease t gradually and minimize overlap energy
    for factor in [1.0, 0.95, 0.9, 0.85, 0.8]:
        if time.time() > deadline:
            break
        t_cur = t * factor
        
        def objective(c):
            return _overlap_energy(c, t_cur)
        
        def obj_grad(c):
            return _overlap_energy_grad(c, t_cur)
        
        try:
            result = minimize(objective, coords, method='L-BFGS-B',
                             jac=obj_grad,
                             options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8,
                                     'maxfun': 2000})
            
            new_coords = np.clip(result.x, -0.4999, 0.4999)
            new_score = _score(new_coords)
            if new_score > best_score:
                best_score = new_score
                best_coords = new_coords
        except Exception:
            pass
    
    return best_coords, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin
    
    best_coords = None
    best_score = -np.inf
    best_t = 0.03
    
    # Try multiple initial configurations
    configs = [
        ('hex', 42),
        ('hex', 123),
        ('rect', 42),
        ('rect', 123),
        ('hex', 999),
    ]
    
    for cfg_type, seed in configs:
        if time.time() > deadline - 30.0:
            break
        
        if cfg_type == 'hex':
            coords = _hex_initial(N, seed)
        else:
            coords = _rect_initial(N, seed)
        
        # Estimate initial t
        t = _score(coords)
        
        # Phase 1: Overlap energy polish
        coords, score = _polish_overlap(coords, t, deadline - 20.0)
        if score > best_score:
            best_score = score
            best_coords = coords.copy()
            best_t = t
        
        # Phase 2: Soft-min polish
        if time.time() > deadline - 15.0:
            break
        coords, t, score = _polish_soft_min(coords, t, beta=50.0, max_iter=200, deadline=deadline - 10.0)
        if score > best_score:
            best_score = score
            best_coords = coords.copy()
            best_t = t
    
    # Final high-precision polish
    if best_coords is not None and time.time() < deadline - 5.0:
        coords, t, score = _polish_soft_min(best_coords, best_t, beta=200.0, max_iter=500, deadline=deadline - 2.0)
        if score > best_score:
            best_score = score
            best_coords = coords.copy()
    
    # Fallback
    if best_coords is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_coords = grid[:N]
    else:
        best_coords = np.clip(best_coords, -0.4999, 0.4999)
    
    return best_coords
