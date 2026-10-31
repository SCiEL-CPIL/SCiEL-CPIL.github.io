"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16
D = 2

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    def get_sq_dists(pts):
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        sq_dists = np.sum(diff**2, axis=2)
        iu = np.triu_indices(N, 1)
        return sq_dists[iu]

    def score_raw(pts):
        d_sq = get_sq_dists(pts)
        d_min = np.sqrt(d_sq.min())
        d_max = np.sqrt(d_sq.max())
        if d_min < 1e-12:
            return 1e10
        return (d_max / d_min) ** 2

    def normalize(pts):
        pts = pts - np.mean(pts, axis=0)
        scale = np.sqrt(np.mean(np.sum(pts**2, axis=1)))
        if scale > 1e-9:
            pts = pts / scale
        return pts

    # Strategy: Distance Geometry / Weiszfeld-based Maximization of Min Distance
    # 1. Maximize the minimum pairwise distance (Tammes problem proxy).
    #    We use a gradient ascent approach on the "soft min" distance, or 
    #    iterative Weiszfeld updates which are known to converge to local maxima 
    #    of the minimum distance.
    # 2. Once a good "packing" is found, minimize the maximum distance (or the ratio)
    #    using L-BFGS-B on the raw ratio or a smoothed version, starting from the packed state.
    
    def weiszfeld_step(pts, iterations=1000):
        """
        Iterative improvement of minimum distance using Weiszfeld's algorithm logic.
        For each point, move it away from its nearest neighbors, weighted by inverse distance.
        """
        pts = pts.copy()
        for _ in range(iterations):
            # Compute distances to all other points
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            d_sq = np.sum(diff**2, axis=2)
            np.fill_diagonal(d_sq, np.inf)
            d = np.sqrt(d_sq)
            
            # For each point i, calculate the gradient of the minimum distance
            # Approximation: Move point i in the direction that increases the distance 
            # to its closest neighbors.
            # Weiszfeld-like update:
            # x_i_new = x_i + sum_{j != i} w_ij * (x_i - x_j) / ||x_i - x_j||^2 ?
            # Actually, standard Weiszfeld is for minimizing sum of distances.
            # To maximize min distance, we can use gradient ascent on min(d_ij).
            # The gradient of min(d_ij) w.r.t x_i is (x_i - x_j_nearest) / d_ij_nearest.
            
            # Let's use a "repulsive" force model which is more robust:
            # Force on i from j: k / d_ij^2 * (x_i - x_j)
            # This tends to equalize distances.
            
            forces = np.zeros_like(pts)
            # Use a strong repulsion for close pairs
            # d is (N, N)
            # Mask diagonal
            mask = ~np.eye(N, dtype=bool)
            
            # Inverse square law repulsion, but only for pairs closer than some threshold?
            # Or just global repulsion.
            # Let's try: F_ij = (x_i - x_j) / d_ij^3
            # This is the gradient of -1/d_ij. Minimizing -1/d is maximizing 1/d, which minimizes d.
            # We want to MAXIMIZE d. So we want to move AWAY.
            # Gradient of d_ij w.r.t x_i is (x_i - x_j)/d_ij.
            # So we move in direction (x_i - x_j)/d_ij.
            
            # To focus on the MINIMUM distance, we weight by exp(-d_ij / T) or similar.
            # Or simply: for each point, find its nearest neighbor and move away from it.
            # This is steepest ascent on the min-distance function.
            
            nearest_idx = np.argmin(d, axis=1)
            nearest_d = d[np.arange(N), nearest_idx]
            
            # Avoid division by zero
            nearest_d = np.maximum(nearest_d, 1e-8)
            
            # Direction to move: away from nearest neighbor
            # v_i = (x_i - x_nearest) / nearest_d
            nearest_pts = pts[nearest_idx]
            vecs = pts - nearest_pts
            # Normalize
            vecs = vecs / nearest_d[:, np.newaxis]
            
            # Step size
            # Adaptive step: small enough to not jump over the minimum, large enough to progress.
            step_size = 0.1
            pts += step_size * vecs
            
            # Normalize to prevent drift
            pts = normalize(pts)
            
        return pts

    def optimize_min_dist(pts, max_iters=5000):
        """
        Refine the configuration to maximize the minimum distance using gradient ascent 
        on the soft-min distance.
        """
        def soft_min_obj(x, T=0.1):
            pts = x.reshape(N, 2)
            d_sq = get_sq_dists(pts)
            d = np.sqrt(np.maximum(d_sq, 1e-10))
            # Soft min: -T * log(sum(exp(-d/T)))
            m = np.min(d)
            return -T * (np.log(np.sum(np.exp(-(d - m) / T))) + m / T) # Wait, log-sum-exp stability
            
            # Correct soft min:
            # f(x) = -T * log( sum( exp(-d_i / T) ) )
            # To be stable:
            # m = min(d)
            # f(x) = -T * ( m/T + log(sum(exp(-(d_i - m)/T))) )
            #      = -m - T * log(sum(exp(-(d_i - m)/T)))
            # We want to MAXIMIZE f(x), so MINIMIZE -f(x).
            # -f(x) = m + T * log(sum(exp(-(d_i - m)/T)))
            # Note: exp(-(d_i - m)/T) = exp(m/T) * exp(-d_i/T).
            # The term inside log is dominated by the smallest d.
            # Let's just compute -soft_min directly.
            
            neg_d = -d
            m_neg = np.max(neg_d) # This is -min(d)
            log_sum_exp = m_neg + np.log(np.sum(np.exp((neg_d - m_neg) / T)))
            soft_min_val = -T * log_sum_exp
            return -soft_min_val # We minimize this to maximize soft_min

        def soft_min_grad(x, T=0.1):
            pts = x.reshape(N, 2)
            d_sq = get_sq_dists(pts)
            d = np.sqrt(np.maximum(d_sq, 1e-10))
            iu = np.triu_indices(N, 1)
            d_vec = d[iu]
            
            neg_d = -d_vec
            m_neg = np.max(neg_d)
            exp_neg = np.exp((neg_d - m_neg) / T)
            sum_exp = np.sum(exp_neg)
            w = exp_neg / sum_exp # Weights for soft max of -d (which is soft min of d)
            
            # Gradient of soft_min(d) w.r.t d_i is w_i.
            # We are minimizing -soft_min(d), so gradient is -w_i.
            # d/dx_k soft_min(d) = sum_i w_i * (d/dx_k d_i)
            # d/dx_k d_ij = (x_i - x_j)/d_ij if k=i, -(x_i-x_j)/d_ij if k=j
            
            grad = np.zeros_like(pts)
            for idx, (i, j) in enumerate(zip(iu[0], iu[1])):
                w_val = -w[idx] # Negative because we minimize -soft_min
                if d_vec[idx] < 1e-8:
                    continue
                diff_vec = pts[i] - pts[j]
                grad[i] += w_val * diff_vec / d_vec[idx]
                grad[j] -= w_val * diff_vec / d_vec[idx]
                
            return grad.ravel()

        x = pts.ravel()
        current_T = 0.5
        for _ in range(5): # Anneal T
            if time.time() > deadline:
                break
            try:
                res = minimize(
                    soft_min_obj,
                    x,
                    args=(current_T,),
                    jac=soft_min_grad,
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                x = res.x
            except Exception:
                pass
            current_T *= 0.5
        
        return x.reshape(N, 2)

    def optimize_ratio(pts, max_iters=1000):
        """
        Refine the configuration to minimize the max/min ratio.
        """
        def ratio_obj(x, T=0.1):
            pts = x.reshape(N, 2)
            d_sq = get_sq_dists(pts)
            d = np.sqrt(np.maximum(d_sq, 1e-10))
            
            # Soft max
            m = np.max(d)
            log_sum_exp_max = m + np.log(np.sum(np.exp((d - m) / T)))
            soft_max = T * log_sum_exp_max
            
            # Soft min
            neg_d = -d
            m_neg = np.max(neg_d)
            log_sum_exp_neg = m_neg + np.log(np.sum(np.exp((neg_d - m_neg) / T)))
            soft_min = -T * log_sum_exp_neg
            
            if soft_min < 1e-8:
                return 1e10
            return soft_max / soft_min

        def ratio_grad(x, T=0.1):
            pts = x.reshape(N, 2)
            d_sq = get_sq_dists(pts)
            d = np.sqrt(np.maximum(d_sq, 1e-10))
            iu = np.triu_indices(N, 1)
            d_vec = d[iu]
            
            # Weights for soft max
            m = np.max(d_vec)
            exp_max = np.exp((d_vec - m) / T)
            sum_exp_max = np.sum(exp_max)
            w_max = exp_max / sum_exp_max
            
            # Weights for soft min
            neg_d = -d_vec
            m_neg = np.max(neg_d)
            exp_neg = np.exp((neg_d - m_neg) / T)
            sum_exp_neg = np.sum(exp_neg)
            w_min = exp_neg / sum_exp_neg
            
            # Objective: f = S_max / S_min
            # S_max = T * LSE(d/T)
            # S_min = -T * LSE(-d/T)
            
            # d f / d d_i = (w_max * S_min - S_max * w_min) / S_min^2
            # Note: w_min here is the weight for the soft MIN component.
            # Gradient of S_min w.r.t d_i is -w_min_i (where w_min_i are weights from exp(-d/T))
            # Let's be precise.
            # S_max = sum p_i d_i where p_i = exp(d_i/T)/sum(exp(d_j/T))
            # d S_max / d d_i = p_i
            # S_min = -T log sum exp(-d/T).
            # d S_min / d d_i = - (exp(-d_i/T) / sum exp(-d_j/T)) * (-1) * T * (-1/T) ?
            # Let L = log sum exp(-d/T). dL/dd_i = (-1/T) exp(-d_i/T) / sum exp(-d/T).
            # S_min = -T L. d S_min / d d_i = -T * (-1/T) * q_i = q_i, where q_i = exp(-d_i/T)/sum(exp(-d_j/T)).
            # Wait, S_min approximates min(d). If d_i decreases, S_min should decrease.
            # Let's check: d_i small -> exp(-d_i/T) large -> q_i large.
            # d S_min / d d_i = q_i. This is positive.
            # So if we increase d_i, S_min increases. Correct.
            
            S_max = T * (m + np.log(np.sum(exp_max)))
            S_min = -T * (m_neg + np.log(np.sum(exp_neg)))
            
            if S_min < 1e-8:
                return np.zeros_like(x)
                
            # grad f = (p_i * S_min - S_max * q_i) / S_min^2
            grad_d = (w_max * S_min - S_max * w_min) / (S_min ** 2)
            
            grad = np.zeros_like(pts)
            for idx, (i, j) in enumerate(zip(iu[0], iu[1])):
                g = grad_d[idx]
                if d_vec[idx] < 1e-8:
                    continue
                diff_vec = pts[i] - pts[j]
                grad[i] += g * diff_vec / d_vec[idx]
                grad[j] -= g * diff_vec / d_vec[idx]
                
            return grad.ravel()

        x = pts.ravel()
        current_T = 0.5
        for _ in range(5):
            if time.time() > deadline:
                break
            try:
                res = minimize(
                    ratio_obj,
                    x,
                    args=(current_T,),
                    jac=ratio_grad,
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                x = res.x
            except Exception:
                pass
            current_T *= 0.5
        
        return x.reshape(N, 2)

    # Initial configurations
    initial_configs = []
    
    # 1. Regular Grid (4x4)
    x = np.arange(4) - 1.5
    y = np.arange(4) - 1.5
    xx, yy = np.meshgrid(x, y)
    base_grid = np.column_stack([xx.ravel(), yy.ravel()])
    initial_configs.append(base_grid.copy())
    
    # 2. Ring (8 inner, 8 outer)
    angles_inner = np.linspace(0, 2*np.pi, 8, endpoint=False)
    angles_outer = np.linspace(0, 2*np.pi, 8, endpoint=False) + np.pi/8
    inner_pts = np.column_stack([0.5 * np.cos(angles_inner), 0.5 * np.sin(angles_inner)])
    outer_pts = np.column_stack([1.0 * np.cos(angles_outer), 1.0 * np.sin(angles_outer)])
    initial_configs.append(np.vstack([inner_pts, outer_pts]))
    
    # 3. Random
    initial_configs.append(np.random.uniform(-2, 2, (N, 2)))
    
    # 4. Hexagonal packing approximation
    # Generate points in a hexagonal grid pattern
    hex_pts = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            if i*i + j*j < 4:
                x_coord = i + 0.5 * j
                y_coord = np.sqrt(3)/2 * j
                hex_pts.append([x_coord, y_coord])
    if len(hex_pts) >= N:
        hex_pts = np.array(hex_pts[:N])
        initial_configs.append(hex_pts)
    else:
        # Pad with random
        hex_pts = np.array(hex_pts)
        while len(hex_pts) < N:
            hex_pts = np.vstack([hex_pts, np.random.uniform(-2, 2, 1)])
        initial_configs.append(hex_pts)

    best_overall = None
    best_overall_score = 1e10

    for init_pts in initial_configs:
        if time.time() > deadline:
            break
            
        init_pts = normalize(init_pts)
        
        # Phase 1: Maximize Min Distance (Packing)
        # Use Weiszfeld-like steps first for robustness, then gradient refinement
        packed_pts = weiszfeld_step(init_pts, iterations=200)
        packed_pts = optimize_min_dist(packed_pts)
        
        # Phase 2: Minimize Ratio
        # Start from the packed configuration
        optimized_pts = optimize_ratio(packed_pts)
        
        # Final polish
        final_pts = normalize(optimized_pts)
        final_score = score_raw(final_pts)
        
        if final_score < best_overall_score:
            best_overall_score = final_score
            best_overall = final_pts

    # If we have time, do a final raw Nelder-Mead polish on the best found
    if time.time() < deadline - 5 and best_overall is not None:
        def raw_obj(x):
            pts = x.reshape(N, 2)
            return score_raw(pts)
        
        try:
            res = minimize(
                raw_obj,
                best_overall.ravel(),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            if res.fun < best_overall_score:
                best_overall = res.x.reshape(N, 2)
                best_overall_score = res.fun
        except Exception:
            pass

    return best_overall

# EVOLVE-BLOCK-END
