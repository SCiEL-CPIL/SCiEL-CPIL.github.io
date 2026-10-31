import numpy as np
import time
from scipy.optimize import minimize
import itertools

N = 16
D = 2

# EVOLVE-BLOCK-START

def _pairwise_dists_sq(P):
    """Return (N, N) matrix of squared pairwise distances."""
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    return np.sum(diff**2, axis=2)

def _score_exact(P):
    """Calculate (max_dist / min_dist)^2."""
    d_sq = _pairwise_dists_sq(P)
    iu = np.triu_indices(N, 1)
    d_sq_upper = d_sq[iu]
    min_d_sq = np.min(d_sq_upper)
    max_d_sq = np.max(d_sq_upper)
    if min_d_sq < 1e-12:
        return 1e10
    return max_d_sq / min_d_sq

def _get_hexagonal_guess():
    """
    Generate a good initial guess based on the hexagonal lattice.
    We pick the 16 points from the 19-point hexagonal cluster (rings 0-2)
    by removing 3 points from the outer ring.
    """
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.5, np.sqrt(3)/2])
    
    hex_points = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            p = i * v1 + j * v2
            d_sq = np.sum(p**2)
            if d_sq <= 4.0 + 1e-9:
                hex_points.append(p)
    
    hex_points = np.array(hex_points)
    
    if len(hex_points) != 19:
        # Fallback if generation logic fails
        return hex_points[:16]

    d_sqs = np.sum(hex_points**2, axis=1)
    outer_indices = np.where(np.isclose(d_sqs, 4.0))[0]
    
    # We need to remove 3 points from the 12 outer points to get 16 points.
    # We will return a list of candidate configurations (indices to keep)
    # But for simplicity in this function, let's just return the full 19 and let the caller handle selection
    # Actually, the current program structure expects a single P. 
    # I will modify the construct function to iterate over combinations.
    # For now, return the 19 points.
    return hex_points

def _soft_objective_and_grad(params, beta):
    """
    Objective and gradient for the smooth surrogate.
    params: flattened array of shape (28,), representing points 2..15.
    Points 0 and 1 are fixed at (0,0) and (1,0).
    """
    P = np.zeros((N, 2))
    P[0] = [0.0, 0.0]
    P[1] = [1.0, 0.0]
    P[2:] = params.reshape(-1, 2)
    
    # Calculate squared distances
    d_sq = _pairwise_dists_sq(P)
    iu = np.triu_indices(N, 1)
    d_sq_upper = d_sq[iu]
    num_pairs = len(d_sq_upper)
    
    # Log-sum-exp for stability
    # Soft Max
    max_val = np.max(d_sq_upper)
    exp_max = np.exp(beta * (d_sq_upper - max_val))
    log_sum_max = np.log(np.sum(exp_max)) + beta * max_val
    soft_max = log_sum_max / beta
    
    # Soft Min
    min_val = np.min(d_sq_upper)
    exp_min = np.exp(-beta * (d_sq_upper - min_val))
    log_sum_min = np.log(np.sum(exp_min)) - beta * min_val
    soft_min = -log_sum_min / beta
    
    if soft_min < 1e-12:
        return 1e10, np.zeros_like(params)
        
    obj = soft_max / soft_min
    
    # Gradient calculation
    # f = M / m
    # df/dM = 1/m
    # df/dm = -M/m^2
    # dM/d(d_k^2) = (1/S_+) * exp(beta * d_k^2)
    # dm/d(d_k^2) = (1/S_-) * exp(-beta * d_k^2)
    
    S_plus = np.sum(exp_max)
    S_minus = np.sum(exp_min)
    
    # We need to map d_k^2 gradients back to points.
    # d_k^2 corresponds to pair (i, j).
    # grad_d_sq[k] is the gradient of the objective w.r.t d_k^2.
    grad_d_sq = np.zeros(num_pairs)
    
    term_M = exp_max / S_plus
    term_m = exp_min / S_minus
    
    # df/d(d_k^2) = (1/m) * term_M[k] - (M/m^2) * term_m[k]
    # = (term_M[k] - M/m * term_m[k]) / m
    ratio = soft_max / soft_min
    grad_d_sq = (term_M - ratio * term_m) / soft_min
    
    # Now, accumulate gradients for each point.
    # For each pair k (i, j), d_k^2 = ||P_i - P_j||^2
    # d(d_k^2)/d(P_i) = 2(P_i - P_j)
    # d(d_k^2)/d(P_j) = 2(P_j - P_i)
    
    grad_P = np.zeros((N, 2))
    
    # Vectorized accumulation is tricky with indices, loop is fast enough for 120 pairs
    for k in range(num_pairs):
        i, j = iu[0][k], iu[1][k]
        g_dk2 = grad_d_sq[k]
        diff = P[i] - P[j]
        # Contribution to P[i]
        grad_P[i] += 2.0 * g_dk2 * diff
        # Contribution to P[j]
        grad_P[j] += 2.0 * g_dk2 * (-diff)
        
    # We only return gradients for the free parameters (points 2..15)
    # Points 0 and 1 are fixed, so their gradients are discarded.
    grad_params = grad_P[2:].flatten()
    
    return obj, grad_params

def construct():
    start_time = time.time()
    deadline = start_time + 115 # Safety margin
    
    # 1. Generate Hexagonal Candidates
    hex_points_19 = _get_hexagonal_guess()
    
    if len(hex_points_19) != 19:
        # Fallback
        return hex_points_19[:16]
        
    d_sqs = np.sum(hex_points_19**2, axis=1)
    outer_indices = np.where(np.isclose(d_sqs, 4.0))[0]
    
    # 2. Pre-filter: Evaluate combinations of removing 3 points from the outer ring
    # C(12, 3) = 220 combinations.
    best_combos = []
    for remove_idx in itertools.combinations(outer_indices, 3):
        keep_idx = [i for i in range(19) if i not in remove_idx]
        pts = hex_points_19[keep_idx]
        score = _score_exact(pts)
        best_combos.append((score, pts))
        
    best_combos.sort(key=lambda x: x[0])
    
    best_score = float('inf')
    best_P = None
    
    # 3. Refine top K candidates using L-BFGS-B with smooth surrogate
    K = 5 # Number of candidates to refine
    
    for i in range(min(K, len(best_combos))):
        if time.time() > deadline - 30:
            break
            
        score_init, init_pts = best_combos[i]
        
        # Fix symmetry: p0=(0,0), p1=(1,0)
        # Find the pair with minimum distance in init_pts
        diff = init_pts[:, np.newaxis, :] - init_pts[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        iu = np.triu_indices(N, 1)
        d_sq = dists_sq[iu]
        
        min_idx = np.argmin(d_sq)
        p, q = np.unravel_index(min_idx, (N, N))
        
        if p > q:
            p, q = q, p
        
        p0 = init_pts[p]
        p1 = init_pts[q]
        
        v = p1 - p0
        norm_v = np.linalg.norm(v)
        
        if norm_v < 1e-8:
            continue
            
        # Rotation matrix to align v with x-axis
        cos_t = v[0] / norm_v
        sin_t = v[1] / norm_v
        R_mat = np.array([[cos_t, sin_t], [-sin_t, cos_t]])
        
        # Transform all points
        transformed = (init_pts - p0) @ R_mat.T / norm_v
        
        # Reorder points so that p is at index 0 and q is at index 1
        perm = list(range(N))
        perm[0] = p
        perm[1] = q
        remaining = [idx for idx in range(N) if idx != p and idx != q]
        perm[2:] = remaining
        
        transformed = transformed[perm]
        
        init_params = transformed[2:].flatten()
        
        # Optimization loop with increasing beta
        # Start with low beta for global landscape, increase for precision
        betas = [5.0, 20.0, 50.0, 100.0]
        
        current_params = init_params.copy()
        
        for beta in betas:
            if time.time() > deadline - 20:
                break
                
            try:
                res = minimize(
                    _soft_objective_and_grad,
                    current_params,
                    args=(beta,),
                    method='L-BFGS-B',
                    jac=True, # Returns tuple (f, g)
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
                )
                
                if res.success or res.fun < best_score:
                    # Reconstruct P to check exact score
                    P_test = np.zeros((N, 2))
                    P_test[0] = [0.0, 0.0]
                    P_test[1] = [1.0, 0.0]
                    P_test[2:] = res.x.reshape(-1, 2)
                    
                    exact_score = _score_exact(P_test)
                    
                    if exact_score < best_score:
                        best_score = exact_score
                        best_P = P_test.copy()
                        current_params = res.x.copy()
                    else:
                        # If exact score is bad but surrogate was good, 
                        # maybe we stuck in a local min of the surrogate that is bad for exact.
                        # Stick with previous best_params if this one didn't improve exact score.
                        # But we should update current_params to the res.x if it's better in surrogate?
                        # Actually, if exact score didn't improve, we should probably revert to the last known good exact configuration's params?
                        # For simplicity, if exact score doesn't improve, we keep current_params from previous beta.
                        # However, res.x might be a better starting point for the next beta even if exact score is slightly worse due to noise?
                        # Let's stick to the one that improved the exact score.
                        pass
                        
            except Exception:
                pass

    # 4. Final Polish with Nelder-Mead on the exact objective if we have time
    if best_P is not None and time.time() < deadline - 10:
        try:
            # Flatten best_P
            params_nm = best_P.flatten()
            
            def exact_obj(params):
                P = params.reshape(-1, 2)
                return _score_exact(P)
                
            res_nm = minimize(
                exact_obj,
                params_nm,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
            
            if res_nm.fun < best_score:
                best_score = res_nm.fun
                best_P = res_nm.x.reshape(-1, 2)
        except Exception:
            pass
            
    if best_P is None:
        # Fallback to best unoptimized lattice subset
        _, best_P = best_combos[0]
        
    # Final validation
    final_score = _score_exact(best_P)
    if final_score > 1e6:
        return np.array([[i - 1.5, j - 1.5] for i in range(4) for j in range(4)], dtype=float)
        
    return best_P

# EVOLVE-BLOCK-END
