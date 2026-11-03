import numpy as np
import time
from scipy.optimize import minimize

N = 100

def _score(points):
    """
    Calculate the exact score (maximum common radius) for a given configuration.
    points: (N, 3) array of centers.
    Returns: max radius r such that spheres of radius r are disjoint and inside [-0.5, 0.5]^3.
    """
    if not np.all(np.isfinite(points)):
        return 0.0
    if np.any(points < -0.5) or np.any(points > 0.5):
        return 0.0
        
    clearance = (0.5 - np.abs(points)).min()
    
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    iu = np.triu_indices(N, k=1)
    min_pair_dist = dists[iu].min()
    
    return min(min_pair_dist * 0.5, clearance)

def _soft_min_obj_and_grad(x, temp):
    """
    Smooth surrogate for maximizing the minimum of:
      - pairwise distances / 2
      - wall clearances (0.5 - |c_ik|)
    
    Uses log-sum-exp (soft-min) with temperature 'temp'.
    As temp -> 0, this approaches the exact min.
    
    We minimize the negative of this soft-min to maximize the radius.
    
    x: flat array of length 3N
    temp: temperature parameter (smaller = sharper min approximation)
    
    Returns: (objective, gradient)
    """
    c = x.reshape(N, 3)
    
    # Pairwise distances (upper triangle only to avoid double counting)
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    
    # We work with squared distances for numerical stability in log-sum-exp
    # But we need the actual distances for the gradient. Let's compute them.
    dists = np.sqrt(np.maximum(dists_sq, 1e-20))
    
    # Get upper triangle indices
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu]  # shape (M,)
    
    # Wall clearances: 0.5 - |c_ik| for each coordinate
    wall_clearances = 0.5 - np.abs(c)  # shape (N, 3)
    wall_vals = wall_clearances.flatten()  # shape (3N,)
    
    # Combine all "radius candidates"
    # For pairs: radius candidate = dist / 2
    # For walls: radius candidate = clearance
    pair_candidates = pair_dists * 0.5
    all_candidates = np.concatenate([pair_candidates, wall_vals])
    
    # Soft-min using log-sum-exp:
    # soft_min = -temp * log(sum(exp(-candidate/temp)))
    # To avoid overflow, subtract the min first
    min_val = np.min(all_candidates)
    shifted = all_candidates - min_val
    exp_terms = np.exp(-shifted / temp)
    log_sum = np.log(np.sum(exp_terms))
    soft_min_val = min_val - temp * log_sum
    
    # Gradient of soft_min with respect to c:
    # d(soft_min)/d(candidate_k) = exp(-candidate_k/temp) / sum(exp(-candidate_j/temp))
    # Then chain rule for each candidate to get gradient w.r.t. c
    
    weights = exp_terms / np.sum(exp_terms)  # shape (M + 3N,)
    
    # Gradient for pairs
    # d(pair_candidate)/d(c_i) = 0.5 * (c_i - c_j) / dist_ij
    grad = np.zeros_like(c)
    
    # Pair gradients
    # weights[:M] correspond to upper triangle pairs
    # We need to map back to (i, j) indices
    # For each pair (i, j) with i < j:
    # d(soft_min)/d(c_i) += weights[k] * 0.5 * (c_i - c_j) / dist_ij
    # d(soft_min)/d(c_j) += weights[k] * 0.5 * (c_j - c_i) / dist_ij
    
    pair_weights = weights[:len(pair_dists)]
    
    # Efficient gradient computation for pairs
    # For each pair (i, j), contribution to grad[i] is w * 0.5 * (c_i - c_j)/d
    # and to grad[j] is w * 0.5 * (c_j - c_i)/d = -w * 0.5 * (c_i - c_j)/d
    
    # We'll accumulate contributions
    # Using the fact that for pair (i,j):
    # term = w * 0.5 / d * (c_i - c_j)
    # grad[i] += term
    # grad[j] -= term
    
    # Vectorized approach:
    # For each pair index k, we have (i_k, j_k)
    # Let's create arrays
    ii, jj = iu
    d_safe = np.maximum(pair_dists, 1e-12)
    
    # term_k = pair_weights[k] * 0.5 / d_safe[k]
    term_coeff = pair_weights * 0.5 / d_safe  # shape (M,)
    
    # For each k, term_vec = term_coeff[k] * (c[i_k] - c[j_k])
    # grad[i_k] += term_vec
    # grad[j_k] -= term_vec
    
    # Use np.add.at for accumulation
    term_vecs = term_coeff[:, np.newaxis] * (c[ii] - c[jj])  # shape (M, 3)
    
    np.add.at(grad, ii, term_vecs)
    np.add.at(grad, jj, -term_vecs)
    
    # Wall gradients
    # d(wall_candidate)/d(c_ik) = -sign(c_ik)
    # So d(soft_min)/d(c_ik) += weights[M + k] * (-sign(c_ik))
    
    wall_weights = weights[len(pair_dists):]  # shape (3N,)
    wall_grad = -wall_weights.reshape(N, 3) * np.sign(c)
    
    grad += wall_grad
    
    # We want to MAXIMIZE soft_min, so we minimize -soft_min
    objective = -soft_min_val
    gradient = -grad.flatten()
    
    return objective, gradient

def _polish_obj_and_grad(x):
    """
    High-precision polish: minimize the negative of the exact minimum radius.
    Uses a very low temperature soft-min as a proxy for the exact min.
    """
    return _soft_min_obj_and_grad(x, temp=1e-8)

def _optimize_soft_min(init_pts, temp, max_iter=5000, time_limit=30.0):
    """
    Run L-BFGS-B optimization on the soft-min objective with bounds.
    """
    t_start = time.time()
    
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    def objective(x):
        if time.time() - t_start > time_limit:
            raise TimeoutError()
        return _soft_min_obj_and_grad(x, temp)
    
    try:
        result = minimize(
            objective,
            init_pts.flatten(),
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
            options={'maxiter': max_iter, 'gtol': 1e-10, 'disp': False}
        )
        opt_pts = result.x.reshape(N, 3)
        return opt_pts, result.success
    except TimeoutError:
        return init_pts, False
    except Exception:
        return init_pts, False

def construct():
    start_time = time.time()
    deadline = 145.0  # Leave 5s margin for 150s limit
    
    # 1. Initialization: 5x5x4 grid
    xs = np.array([-0.4, -0.2, 0.0, 0.2, 0.4])
    ys = np.array([-0.4, -0.2, 0.0, 0.2, 0.4])
    zs = np.array([-0.375, -0.125, 0.125, 0.375])
    
    centers = []
    for z in zs:
        for y in ys:
            for x in xs:
                centers.append([x, y, z])
    centers = np.array(centers)
    
    if len(centers) != N:
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        centers = grid[:N]

    best_pts = centers.copy()
    best_r = _score(best_pts)
    
    rng = np.random.RandomState(42)
    
    # Strategy: Anneal the temperature of the soft-min objective
    # Start with higher temp (smoother landscape, easier to navigate)
    # Gradually decrease temp (sharper min, more accurate)
    
    temps = [0.05, 0.03, 0.02, 0.015, 0.01, 0.008, 0.005, 0.003, 0.002, 0.001]
    
    current_pts = best_pts.copy()
    
    for temp in temps:
        if time.time() > deadline - 10.0:
            break
        
        # Add small noise to break symmetry
        noise = rng.normal(0, 0.003, size=(N, 3))
        init_pts = np.clip(current_pts + noise, -0.49, 0.49)
        
        remaining = deadline - (time.time() - start_time) - 5.0
        if remaining < 5.0:
            break
        
        max_iter = 2000
        if temp < 0.01:
            max_iter = 3000
        
        opt_pts, success = _optimize_soft_min(
            init_pts, 
            temp, 
            max_iter=max_iter, 
            time_limit=remaining
        )
        
        if np.all(np.isfinite(opt_pts)):
            opt_pts = np.clip(opt_pts, -0.5 + 1e-9, 0.5 - 1e-9)
            r_opt = _score(opt_pts)
            
            if r_opt > best_r:
                best_r = r_opt
                best_pts = opt_pts.copy()
                current_pts = opt_pts.copy()
    
    # Random restarts with larger noise if time permits
    if time.time() < deadline - 15.0:
        for i in range(5):
            if time.time() > deadline - 5.0:
                break
            
            noise = rng.normal(0, 0.015, size=(N, 3))
            init_pts = np.clip(best_pts + noise, -0.49, 0.49)
            
            remaining = deadline - (time.time() - start_time) - 5.0
            if remaining < 5.0:
                break
            
            opt_pts, success = _optimize_soft_min(
                init_pts, 
                0.005, 
                max_iter=1500, 
                time_limit=remaining
            )
            
            if np.all(np.isfinite(opt_pts)):
                opt_pts = np.clip(opt_pts, -0.5 + 1e-9, 0.5 - 1e-9)
                r_opt = _score(opt_pts)
                
                if r_opt > best_r:
                    best_r = r_opt
                    best_pts = opt_pts.copy()

    # Final Polish: Very low temperature for high precision
    if time.time() < deadline - 3.0 and best_r > 0.1:
        noise = rng.normal(0, 0.001, size=(N, 3))
        init_pts = np.clip(best_pts + noise, -0.49, 0.49)
        
        remaining = deadline - (time.time() - start_time) - 2.0
        if remaining > 2.0:
            opt_pts, success = _optimize_soft_min(
                init_pts, 
                1e-8, 
                max_iter=1000, 
                time_limit=remaining
            )
            if np.all(np.isfinite(opt_pts)):
                opt_pts = np.clip(opt_pts, -0.5 + 1e-9, 0.5 - 1e-9)
                r_opt = _score(opt_pts)
                if r_opt > best_r:
                    best_r = r_opt
                    best_pts = opt_pts.copy()

    return best_pts

# EVOLVE-BLOCK-END
