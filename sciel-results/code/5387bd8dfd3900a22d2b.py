"""Rewritten program for packing 200 equal circles in a unit square."""
import time
import numpy as np
from scipy.optimize import minimize

N = 200


def _score(coords):
    """Compute the exact radius for a given set of coordinates."""
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    min_pair = np.min(dist[~np.eye(N, dtype=bool)])
    min_bound = np.min(np.minimum(np.minimum(0.5 - coords[:, 0], coords[:, 0] + 0.5),
                                   np.minimum(0.5 - coords[:, 1], coords[:, 1] + 0.5)))
    return min(min_pair / 2.0, min_bound)


def _soft_min_obj(coords, beta):
    """
    Smooth surrogate objective: minimize -soft_min(min_pair/2, min_bound)
    using log-sum-exp for smoothness.
    Returns (objective_value, gradient).
    """
    # Pairwise distances
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    # Set diagonal to large value
    np.fill_diagonal(dist_sq, np.inf)
    # Use squared distances for numerical stability; min_pair/2 = 0.5 * sqrt(min dist_sq)
    # soft_min over pairwise: we want soft min of (dist_ij / 2)
    # log-sum-exp trick: soft_min(x_i) = -logsumexp(-beta * x_i) / beta
    # Here x_i = dist_ij / 2 for all pairs i<j
    # Use dist_sq to avoid sqrt in the exponent: dist/2 = sqrt(dist_sq)/2
    # -beta * dist/2 = -beta/2 * sqrt(dist_sq)
    neg_beta_half = -beta / 2.0
    exp_vals = np.exp(neg_beta_half * np.sqrt(dist_sq))
    # Sum over all pairs (off-diagonal)
    pair_sum = np.sum(exp_vals)
    soft_min_pair = -np.log(pair_sum) / beta

    # Boundary distances
    d_left = coords[:, 0] + 0.5
    d_right = 0.5 - coords[:, 0]
    d_bottom = coords[:, 1] + 0.5
    d_top = 0.5 - coords[:, 1]
    # Combine into a single array of boundary distances
    d_bound = np.concatenate([d_left, d_right, d_bottom, d_top])
    exp_bound = np.exp(-beta * d_bound)
    bound_sum = np.sum(exp_bound)
    soft_min_bound = -np.log(bound_sum) / beta

    # Now soft-min of soft_min_pair and soft_min_bound
    # Use log-sum-exp again: soft_min(a, b) = -log(exp(-beta*a) + exp(-beta*b)) / beta
    total_exp = np.exp(-beta * soft_min_pair) + np.exp(-beta * soft_min_bound)
    soft_min_total = -np.log(total_exp) / beta

    objective = -soft_min_total

    # Gradient computation
    # d(soft_min_total)/d(coords) via chain rule
    # soft_min_total = -log(exp(-beta*sm_pair) + exp(-beta*sm_bound)) / beta
    # Let E = exp(-beta*sm_pair) + exp(-beta*sm_bound)
    # d(soft_min_total)/d(sm_pair) = (1/beta) * (1/E) * (-beta) * exp(-beta*sm_pair) = -exp(-beta*sm_pair)/E
    # d(soft_min_total)/d(sm_bound) = -exp(-beta*sm_bound)/E
    # d(objective)/d(coords) = -d(soft_min_total)/d(coords)

    w_pair = np.exp(-beta * soft_min_pair) / total_exp
    w_bound = np.exp(-beta * soft_min_bound) / total_exp

    # Gradient of soft_min_pair w.r.t. coords
    # sm_pair = -log(sum_ij exp(-beta/2 * sqrt(dist_sq_ij))) / beta
    # d(sm_pair)/d(dist_sq_ij) = (1/beta) * (1/S) * (-beta/2) * exp(-beta/2*sqrt(dist_sq)) * (1/(2*sqrt(dist_sq)))
    # where S = sum of exp terms
    # = -exp(-beta/2*sqrt(dist_sq)) / (2 * S * sqrt(dist_sq))
    sqrt_dist_sq = np.sqrt(dist_sq)
    exp_pair = np.exp(-beta / 2.0 * sqrt_dist_sq)
    S_pair = np.sum(exp_pair)
    # d(sm_pair)/d(dist_sq_ij) = -exp_pair_ij / (2 * S_pair * sqrt_dist_sq_ij)
    # d(dist_sq_ij)/d(coords_i) = 2*(coords_i - coords_j)
    # d(dist_sq_ij)/d(coords_j) = 2*(coords_j - coords_i)
    # So d(sm_pair)/d(coords_i) = sum_j d(sm_pair)/d(dist_sq_ij) * 2*(coords_i - coords_j)

    grad_sm_pair = np.zeros_like(coords)
    # Vectorized: for each i, sum over j of [-exp_pair[i,j] / (2*S*sqrt_dist_sq[i,j])] * 2*(coords[i]-coords[j])
    # = sum_j [-exp_pair[i,j] / (S*sqrt_dist_sq[i,j])] * (coords[i]-coords[j])
    # Set diagonal to 0
    exp_pair = exp_pair.copy()
    np.fill_diagonal(exp_pair, 0.0)
    sqrt_dist_sq = sqrt_dist_sq.copy()
    np.fill_diagonal(sqrt_dist_sq, 1e-12)  # avoid div by zero
    factor = exp_pair / (S_pair * sqrt_dist_sq)
    # factor[i,j] * (coords[i] - coords[j]), sum over j
    # grad_sm_pair[i] = -sum_j factor[i,j] * (coords[i] - coords[j])
    # = -sum_j factor[i,j]*coords[i] + sum_j factor[i,j]*coords[j]
    # = -coords[i]*sum_j factor[i,j] + sum_j factor[i,j]*coords[j]
    row_sums = np.sum(factor, axis=1)
    weighted_coords = factor @ coords
    grad_sm_pair = -row_sums[:, np.newaxis] * coords + weighted_coords

    # Gradient of soft_min_bound w.r.t. coords
    # sm_bound = -log(sum exp(-beta*d_k)) / beta
    # d(sm_bound)/d(d_k) = exp(-beta*d_k) / S_bound
    # where S_bound = sum exp(-beta*d_k)
    # d_k are: d_left[i]=x_i+0.5, d_right[i]=0.5-x_i, d_bottom[i]=y_i+0.5, d_top[i]=0.5-y_i
    # d(d_left[i])/d(coords[i,0]) = 1
    # d(d_right[i])/d(coords[i,0]) = -1
    # d(d_bottom[i])/d(coords[i,1]) = 1
    # d(d_top[i])/d(coords[i,1]) = -1
    exp_bound = np.exp(-beta * d_bound)
    S_bound = np.sum(exp_bound)
    weights_bound = exp_bound / S_bound  # shape (4N,)
    # Split into d_left, d_right, d_bottom, d_top weights
    w_left = weights_bound[:N]
    w_right = weights_bound[N:2*N]
    w_bottom = weights_bound[2*N:3*N]
    w_top = weights_bound[3*N:4*N]

    grad_sm_bound = np.zeros_like(coords)
    # d(sm_bound)/d(coords[i,0]) = w_left[i]*1 + w_right[i]*(-1) = w_left[i] - w_right[i]
    # d(sm_bound)/d(coords[i,1]) = w_bottom[i]*1 + w_top[i]*(-1) = w_bottom[i] - w_top[i]
    grad_sm_bound[:, 0] = w_left - w_right
    grad_sm_bound[:, 1] = w_bottom - w_top

    # Total gradient of objective = -[w_pair * grad_sm_pair + w_bound * grad_sm_bound]
    grad = -(w_pair * grad_sm_pair + w_bound * grad_sm_bound)

    return objective, grad


def _generate_hex_initial(n_points, seed=42):
    """Generate initial hexagonal lattice points, centered in unit square."""
    rng = np.random.RandomState(seed)
    # Estimate spacing
    s_est = np.sqrt(2.0 / (n_points * np.sqrt(3.0)))
    points = []
    max_radius = 0.55
    for y_row in range(-30, 31):
        y = y_row * s_est * np.sqrt(3.0) / 2.0
        if abs(y) > max_radius:
            continue
        x_offset = (s_est / 2.0) if (y_row % 2 != 0) else 0.0
        for x_col in range(-30, 31):
            x = x_col * s_est + x_offset
            if abs(x) > max_radius or abs(y) > max_radius:
                continue
            points.append([x, y])
    points = np.array(points)
    if len(points) < n_points:
        # Fallback: grid
        m = int(np.ceil(np.sqrt(n_points)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        points = grid
    idx = rng.choice(len(points), n_points, replace=False)
    return points[idx]


def _polish_exact(coords, deadline):
    """
    Polish using exact objective with local search:
    iterate: find most-violated constraint, move to fix it.
    """
    coords = coords.copy()
    best_score = _score(coords)
    
    while time.time() < deadline:
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dist, np.inf)
        
        min_pair_idx = np.unravel_index(np.argmin(dist), dist.shape)
        min_pair_val = dist[min_pair_idx]
        
        d_left = coords[:, 0] + 0.5
        d_right = 0.5 - coords[:, 0]
        d_bottom = coords[:, 1] + 0.5
        d_top = 0.5 - coords[:, 1]
        d_bound = np.minimum(np.minimum(d_left, d_right), np.minimum(d_bottom, d_top))
        min_bound_idx = np.argmin(d_bound)
        min_bound_val = d_bound[min_bound_idx]
        
        current_score = min(min_pair_val / 2.0, min_bound_val)
        if current_score > best_score:
            best_score = current_score
            # Store best
            if best_score > _score(coords):
                pass  # coords already updated
        
        # Find the most violated constraint
        if min_pair_val / 2.0 < min_bound_val:
            i, j = min_pair_idx
            if i == j:
                break
            diff_ij = coords[i] - coords[j]
            dist_ij = np.linalg.norm(diff_ij)
            if dist_ij < 1e-14:
                # Random nudge
                coords[i] += np.random.normal(0, 0.001, 2)
                coords[j] += np.random.normal(0, 0.001, 2)
                continue
            # Move i and j apart
            direction = diff_ij / dist_ij
            step = 0.5 * (min_pair_val / 2.0 + 1e-6 - min_pair_val / 2.0)  # small step
            # Actually, we want to increase the distance
            # Move i by +direction * small_step, j by -direction * small_step
            small_step = 0.001
            coords[i] += direction * small_step
            coords[j] -= direction * small_step
        else:
            idx = min_bound_idx
            # Determine which boundary
            if d_bound[idx] == d_left[idx]:
                coords[idx, 0] += 0.001
            elif d_bound[idx] == d_right[idx]:
                coords[idx, 0] -= 0.001
            elif d_bound[idx] == d_bottom[idx]:
                coords[idx, 1] += 0.001
            else:
                coords[idx, 1] -= 0.001
        
        # Clamp
        coords = np.clip(coords, -0.4999, 0.4999)
        
        # Check if improved
        new_score = _score(coords)
        if new_score > best_score:
            best_score = new_score
        else:
            # Revert if no improvement
            coords = np.clip(coords, -0.4999, 0.4999)
    
    return coords


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s safety margin
    
    best_coords = None
    best_score = -np.inf
    
    # Try multiple initial configurations with soft-min optimization
    n_restarts = 8
    for restart in range(n_restarts):
        if time.time() > deadline - 30.0:
            break
        
        seed = 42 + restart * 17
        if restart == 0:
            coords = _generate_hex_initial(N, seed=seed)
        elif restart == 1:
            coords = _generate_hex_initial(N, seed=seed) + np.random.RandomState(seed).normal(0, 0.005, (N, 2))
        elif restart == 2:
            coords = _generate_hex_initial(N, seed=seed) + np.random.RandomState(seed).normal(0, 0.01, (N, 2))
        else:
            # Random perturbation of hex
            coords = _generate_hex_initial(N, seed=seed)
            coords += np.random.RandomState(seed).normal(0, 0.02, (N, 2))
        
        # Optimize using soft-min objective with L-BFGS-B
        try:
            # Start with moderate beta, increase for sharper approximation
            betas = [20.0, 50.0, 100.0, 200.0, 500.0]
            for beta in betas:
                if time.time() > deadline - 25.0:
                    break
                
                def obj_grad(x, b=beta):
                    return _soft_min_obj(x, b)
                
                result = minimize(lambda x: _soft_min_obj(x, beta)[0], coords,
                                 method='L-BFGS-B',
                                 jac=lambda x: _soft_min_obj(x, beta)[1],
                                 options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10,
                                          'maxfun': 5000})
                
                if result.fun < _soft_min_obj(coords, beta)[0]:
                    coords = result.x
            coords = np.clip(coords, -0.4999, 0.4999)
            
            score = _score(coords)
            if score > best_score:
                best_score = score
                best_coords = coords.copy()
                
        except Exception:
            pass
    
    # Final polish with exact local search
    if best_coords is not None and time.time() < deadline - 10.0:
        polished = _polish_exact(best_coords, deadline - 5.0)
        polished_score = _score(polished)
        if polished_score > best_score:
            best_coords = polished
            best_score = polished_score
    
    # Ensure coordinates are within bounds
    if best_coords is None:
        # Fallback to grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_coords = grid[:N]
    else:
        # Clamp to valid range
        best_coords = np.clip(best_coords, -0.4999, 0.4999)
    
    return best_coords
