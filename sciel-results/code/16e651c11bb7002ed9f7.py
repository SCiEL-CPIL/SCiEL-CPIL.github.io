import numpy as np
import time
from scipy.optimize import minimize

N = 16
D = 2

# EVOLVE-BLOCK-START

def _pairwise_dists_sq(P):
    """Return vector of squared pairwise distances for upper triangle."""
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    iu = np.triu_indices(N, 1)
    return dists_sq[iu]

def _score(P):
    """Calculate (max_dist / min_dist)^2."""
    d_sq = _pairwise_dists_sq(P)
    min_d_sq = np.min(d_sq)
    max_d_sq = np.max(d_sq)
    if min_d_sq < 1e-12:
        return 1e10
    return max_d_sq / min_d_sq

def _soft_max_min_objective(params, beta_max=50.0, beta_min=50.0, lambda_min=100.0):
    """
    Vectorized soft objective:
    Minimize soft_max(d_sq) + lambda * soft_penalty(d_sq < 1)
    
    params: flat array of length 2*(N-2) for points 2..N-1.
    Points 0 and 1 are fixed at (0,0) and (1,0).
    """
    P = np.zeros((N, 2))
    P[0] = [0.0, 0.0]
    P[1] = [1.0, 0.0]
    P[2:] = params.reshape(-1, 2)
    
    # Compute pairwise squared distances
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    d_sq = np.sum(diff**2, axis=2)
    iu = np.triu_indices(N, 1)
    d_sq_iu = d_sq[iu]
    
    # Soft max of d_sq (approximates max distance squared)
    max_d_sq = np.max(d_sq_iu)
    # Shift for numerical stability
    shifted = d_sq_iu - max_d_sq
    # Clip to avoid overflow in exp
    shifted = np.clip(shifted, -50, 0) 
    soft_max_val = max_d_sq + (1.0 / beta_max) * np.log(np.sum(np.exp(beta_max * shifted)))
    
    # Soft penalty for min distance
    # We want d_sq >= 1. Penalty if d_sq < 1.
    # Use soft min of (1 - d_sq) where d_sq < 1, or simply penalize d_sq < 1 directly.
    # A smooth penalty: sum( exp(beta_min * (1 - d_sq)) / beta_min ) for d_sq < 1
    # Or simpler: sum( max(0, 1 - d_sq)^2 ) is not smooth.
    # Let's use: sum( (1 - d_sq) / (1 + exp(-beta_min * (d_sq - 1))) ) -> sigmoid-like penalty
    
    # Actually, let's stick to a standard soft-min of the distances to ensure they are large.
    # We want to maximize min(d_sq).
    # So minimize -soft_min(d_sq).
    min_d_sq = np.min(d_sq_iu)
    shifted_min = min_d_sq - d_sq_iu
    shifted_min = np.clip(shifted_min, -50, 0)
    soft_min_val = min_d_sq - (1.0 / beta_min) * np.log(np.sum(np.exp(beta_min * shifted_min)))
    
    # Objective: Minimize soft_max - lambda * soft_min
    # This balances keeping diameter small and separation large.
    # Since we fixed min dist to 1, we really just want to minimize max dist 
    # while ensuring no dist drops below 1.
    # If we use (max - min), it scales. But we fixed min=1.
    # So we minimize soft_max, with a heavy penalty if soft_min < 1.
    
    penalty = 0.0
    if soft_min_val < 1.0:
        # Quadratic penalty for violating min distance
        penalty = lambda_min * (1.0 - soft_min_val) ** 2
    
    return soft_max_val + penalty

def _soft_max_min_gradient(params, beta_max=50.0, beta_min=50.0, lambda_min=100.0):
    """
    Compute objective and gradient for the soft objective.
    """
    P = np.zeros((N, 2))
    P[0] = [0.0, 0.0]
    P[1] = [1.0, 0.0]
    P[2:] = params.reshape(-1, 2)
    
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    d_sq = np.sum(diff**2, axis=2)
    iu = np.triu_indices(N, 1)
    i_idx, j_idx = iu
    d_sq_iu = d_sq[iu]
    
    # --- Soft Max Term ---
    max_d_sq = np.max(d_sq_iu)
    shifted_max = d_sq_iu - max_d_sq
    shifted_max = np.clip(shifted_max, -50, 0)
    exp_max = np.exp(beta_max * shifted_max)
    sum_exp_max = np.sum(exp_max)
    soft_max_val = max_d_sq + (1.0 / beta_max) * np.log(sum_exp_max)
    
    # Gradient of soft_max wrt d_sq_iu
    # d(soft_max)/d(d_sq_k) = exp(beta * (d_k - max)) / sum(exp(...))
    grad_soft_max_dsq = exp_max / sum_exp_max
    
    # --- Soft Min Penalty Term ---
    # We defined penalty as lambda * (1 - soft_min)^2 if soft_min < 1 else 0.
    # Let's compute soft_min and its gradient.
    min_d_sq = np.min(d_sq_iu)
    shifted_min = min_d_sq - d_sq_iu
    shifted_min = np.clip(shifted_min, -50, 0)
    exp_min = np.exp(beta_min * shifted_min)
    sum_exp_min = np.sum(exp_min)
    soft_min_val = min_d_sq - (1.0 / beta_min) * np.log(sum_exp_min)
    
    # Gradient of soft_min wrt d_sq_iu
    # d(soft_min)/d(d_sq_k) = - exp(-beta * (d_k - min)) / sum(exp(-...))
    # Note: shifted_min = min - d. exp(beta * (min - d)).
    # Let w_k = exp(beta_min * (min_d_sq - d_sq_k)) / sum(exp(...))
    # d(soft_min)/d(d_sq_k) = - w_k
    grad_soft_min_dsq = - (exp_min / sum_exp_min)
    
    penalty = 0.0
    grad_penalty_dsq = np.zeros_like(d_sq_iu)
    if soft_min_val < 1.0:
        diff_val = 1.0 - soft_min_val
        penalty = lambda_min * diff_val ** 2
        # d(penalty)/d(soft_min) = -2 * lambda * diff_val
        # d(penalty)/d(d_sq_k) = d(penalty)/d(soft_min) * d(soft_min)/d(d_sq_k)
        grad_penalty_dsq = (-2.0 * lambda_min * diff_val) * grad_soft_min_dsq
    
    obj = soft_max_val + penalty
    
    # Total gradient wrt d_sq_iu
    grad_obj_dsq = grad_soft_max_dsq + grad_penalty_dsq
    
    # --- Backpropagate to P ---
    # d_sq_k = ||P[i] - P[j]||^2
    # d(d_sq_k)/d(P[m]) = 2 * (P[m] - P[j]) if m == i
    #                    = 2 * (P[m] - P[i]) if m == j
    #                    = 0 otherwise
    
    grad_P = np.zeros_like(P)
    for k in range(len(d_sq_iu)):
        g = grad_obj_dsq[k]
        if g == 0: continue
        i, j = i_idx[k], j_idx[k]
        grad_P[i] += 2.0 * (P[i] - P[j]) * g
        grad_P[j] += 2.0 * (P[j] - P[i]) * g
        
    # Extract gradient for params (indices 2 onwards)
    grad_params = grad_P[2:].flatten()
    
    return obj, grad_params

def _local_optimize_vectorized(params, time_budget=5.0):
    """
    Custom vectorized gradient descent with adaptive step size.
    This is faster than scipy.minimize for this specific structure.
    """
    start_time = time.time()
    current_params = params.copy()
    current_obj, current_grad = _soft_max_min_gradient(current_params)
    best_params = current_params.copy()
    best_obj = current_obj
    
    step_size = 0.01
    patience = 0
    max_patience = 20
    
    while time.time() - start_time < time_budget:
        # Simple gradient step
        new_params = current_params - step_size * current_grad
        
        # Check for improvement
        new_obj, new_grad = _soft_max_min_gradient(new_params)
        
        if new_obj < current_obj:
            current_params = new_params
            current_obj = new_obj
            current_grad = new_grad
            patience = 0
            # Increase step size slightly if improving
            step_size *= 1.05
        else:
            patience += 1
            step_size *= 0.5
            if patience > max_patience:
                break
        
        # Update best
        if current_obj < best_obj:
            best_obj = current_obj
            best_params = current_params.copy()
            
    # Final polish with a few iterations of smaller steps
    for _ in range(50):
        if time.time() - start_time > time_budget: break
        step_size = 0.001
        new_params = current_params - step_size * current_grad
        new_obj, new_grad = _soft_max_min_gradient(new_params)
        if new_obj < current_obj:
            current_params = new_params
            current_obj = new_obj
            current_grad = new_grad
            if current_obj < best_obj:
                best_obj = current_obj
                best_params = current_params.copy()
        else:
            break
            
    # Calculate exact score for the best params found
    P_best = np.zeros((N, 2))
    P_best[0] = [0.0, 0.0]
    P_best[1] = [1.0, 0.0]
    P_best[2:] = best_params.reshape(-1, 2)
    exact_score = _score(P_best)
    
    return best_params, exact_score

def _prepare_params(P):
    """
    Fix closest pair at (0,0) and (1,0), return params for remaining 14 points.
    """
    d_sq = _pairwise_dists_sq(P)
    min_idx = np.argmin(d_sq)
    iu = np.triu_indices(N, 1)
    i_min, j_min = iu[0][min_idx], iu[1][min_idx]
    
    p0 = P[i_min]
    p1 = P[j_min]
    
    v = p1 - p0
    norm_v = np.linalg.norm(v)
    if norm_v < 1e-10:
        return None
    
    # Rotation to align v with x-axis
    cos_t = v[0] / norm_v
    sin_t = v[1] / norm_v
    R = np.array([[cos_t, sin_t], [-sin_t, cos_t]])
    
    # Transform: translate so p0 is origin, rotate, scale so p1 is at (1,0)
    transformed = (P - p0) @ R.T / norm_v
    
    # Reorder: i_min -> 0, j_min -> 1, rest in original order
    perm = list(range(N))
    perm[0] = i_min
    perm[1] = j_min
    remaining = [i for i in range(N) if i != i_min and i != j_min]
    perm[2:] = remaining
    
    transformed = transformed[perm]
    params = transformed[2:].flatten()
    return params

def _get_initial_candidates():
    """Generate diverse initial configurations."""
    rng = np.random.default_rng(42)
    candidates = []
    
    # 1. Hexagonal lattice subset (Densest packing)
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.5, np.sqrt(3)/2])
    hex_points = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            p = i * v1 + j * v2
            if np.sum(p**2) <= 16.0 + 1e-9:
                hex_points.append(p)
    hex_points = np.array(hex_points)
    d_sqs = np.sum(hex_points**2, axis=1)
    sorted_indices = np.argsort(d_sqs)
    candidates.append(hex_points[sorted_indices][:16])
    
    # Jittered hex
    for sigma in [0.01, 0.05, 0.1]:
        candidates.append(hex_points[sorted_indices][:16] + rng.normal(0, sigma, (16, 2)))
    
    # 2. 4x4 grid
    grid = np.array([[i - 1.5, j - 1.5] for i in range(4) for j in range(4)], dtype=float)
    candidates.append(grid)
    candidates.append(grid + rng.normal(0, 0.1, (16, 2)))
    
    # 3. Concentric circles (8 inner + 8 outer)
    angles_in = np.linspace(0, 2*np.pi, 8, endpoint=False)
    angles_out = np.linspace(0, 2*np.pi, 8, endpoint=False) + np.pi/8
    inner = np.column_stack([0.5 * np.cos(angles_in), 0.5 * np.sin(angles_in)])
    outer = np.column_stack([1.5 * np.cos(angles_out), 1.5 * np.sin(angles_out)])
    candidates.append(np.vstack([inner, outer]))
    
    # 4. Golden angle spiral (Fibonacci)
    n = 16
    golden_angle = np.pi * (3. - np.sqrt(5.))
    angles = np.arange(n) * golden_angle
    radii = np.sqrt(np.arange(n)) / np.sqrt(n) * 2.0
    spiral = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    candidates.append(spiral)
    candidates.append(spiral + rng.normal(0, 0.1, (16, 2)))
    
    # 5. Random uniform
    for _ in range(5):
        candidates.append(rng.uniform(-2, 2, (16, 2)))
    
    # 6. Random on ring
    for _ in range(3):
        angles = rng.uniform(0, 2*np.pi, 16)
        radii = rng.uniform(0.5, 2.0, 16)
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        candidates.append(pts)
        
    # 7. Two rings optimized for spacing
    # 8 points on r=0.8, 8 points on r=1.6, offset
    r1, r2 = 0.8, 1.6
    a1 = np.linspace(0, 2*np.pi, 8, endpoint=False)
    a2 = np.linspace(0, 2*np.pi, 8, endpoint=False) + np.pi/8
    p1 = np.column_stack([r1*np.cos(a1), r1*np.sin(a1)])
    p2 = np.column_stack([r2*np.cos(a2), r2*np.sin(a2)])
    candidates.append(np.vstack([p1, p2]))
    
    return candidates

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    candidates = _get_initial_candidates()
    rng = np.random.default_rng(123)
    
    best_params = None
    best_score = float('inf')
    
    # Phase 1: Local optimization from diverse starts using the new vectorized optimizer
    for P0 in candidates:
        if time.time() > deadline - 30:
            break
        try:
            params = _prepare_params(P0)
            if params is None:
                continue
            
            remaining_time = deadline - 30 - time.time()
            if remaining_time < 2:
                break
            
            # Allocate time based on remaining budget
            # We have ~145s total. Candidates ~20. 145/20 ~ 7s each.
            # But we want to spend more time on good candidates.
            # Let's do a quick pass first with 2s each, then refine best.
            time_per_opt = min(remaining_time / 2.0, 3.0)
            
            p_opt, score = _local_optimize_vectorized(params, time_budget=time_per_opt)
            if score < best_score:
                best_score = score
                best_params = p_opt
        except Exception:
            pass
            
    # Phase 2: Refine the best candidate found so far with more time
    if best_params is not None and time.time() < deadline - 10:
        remaining_time = deadline - 10 - time.time()
        # Spend up to 20s on the best candidate
        time_budget_refine = min(remaining_time, 20.0)
        p_opt, score = _local_optimize_vectorized(best_params, time_budget=time_budget_refine)
        if score < best_score:
            best_score = score
            best_params = p_opt
            
    # Phase 3: Random restarts in reduced space if time permits
    while time.time() < deadline - 20:
        try:
            # Generate random points in a reasonable range
            P0 = rng.uniform(-2, 2, (14, 2))
            # Add fixed points
            P_full = np.vstack([[0,0], [1,0], P0])
            params = P0.flatten()
            
            remaining_time = deadline - 20 - time.time()
            if remaining_time < 2:
                break
            time_budget_rand = min(remaining_time / 5.0, 5.0)
            
            p_opt, score = _local_optimize_vectorized(params, time_budget=time_budget_rand)
            if score < best_score:
                best_score = score
                best_params = p_opt
        except Exception:
            pass
            
    # Phase 4: Final high-precision polish on the best candidate
    if best_params is not None and time.time() < deadline - 5:
        remaining_time = deadline - 5 - time.time()
        # Use a very tight gradient step and many iterations
        # We can simulate this by calling the optimizer again with a small budget
        # and relying on the internal "polish" loop
        p_opt, score = _local_optimize_vectorized(best_params, time_budget=min(remaining_time, 5.0))
        if score < best_score:
            best_score = score
            best_params = p_opt
            
    # Construct final P
    if best_params is None:
        # Fallback
        return candidates[0]
    else:
        P_final = np.zeros((N, 2))
        P_final[0] = [0.0, 0.0]
        P_final[1] = [1.0, 0.0]
        P_final[2:] = best_params.reshape(-1, 2)
        
        # Verify distinctness
        d_sq = _pairwise_dists_sq(P_final)
        if np.min(d_sq) < 1e-8:
             # Add tiny perturbation
             P_final = P_final + rng.normal(0, 1e-6, P_final.shape)
             
        return P_final

# EVOLVE-BLOCK-END
