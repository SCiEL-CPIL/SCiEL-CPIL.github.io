import numpy as np
import time
import scipy.spatial
import scipy.optimize

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the ratio of the smallest triangle area
    to the convex hull area using Simulated Annealing with a smooth surrogate objective.
    """
    n = 13
    start_time = time.time()
    time_limit = 580  # Safety margin
    
    # Precompute indices for triangles
    idx = np.arange(n)
    i_idx, j_idx, k_idx = np.meshgrid(idx, idx, idx, indexing='ij')
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    tri_i = i_idx[mask]
    tri_j = j_idx[mask]
    tri_k = k_idx[mask]
    # tri_i, tri_j, tri_k are arrays of length 286

    def get_points_from_angles(angles):
        """Convert 13 angles to 13 (x,y) points on unit circle."""
        x = np.cos(angles)
        y = np.sin(angles)
        return np.column_stack((x, y))

    def calculate_metric(points, beta):
        """
        Calculate the smooth surrogate metric.
        beta controls the sharpness of the min approximation.
        """
        try:
            hull = scipy.spatial.ConvexHull(points)
            A_hull = hull.volume
            if A_hull <= 1e-9:
                return 0.0
        except Exception:
            return 0.0
            
        p_i = points[tri_i]
        p_j = points[tri_j]
        p_k = points[tri_k]
        
        # Area = 0.5 * |cross(p_j - p_i, p_k - p_i)|
        v1 = p_j - p_i
        v2 = p_k - p_i
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = 0.5 * np.abs(cross)
        
        if np.min(areas) < 1e-9:
            return 0.0
            
        # Soft-min approximation: -1/beta * log(sum(exp(-beta * areas)))
        # To avoid numerical overflow/underflow, we can subtract the min area first,
        # but since we are maximizing, and areas are positive, we can just use the formula.
        # For stability, let's use a stable log-sum-exp implementation.
        # max_a = np.max(areas)
        # log_sum = max_a + np.log(np.sum(np.exp(-beta * (areas - max_a))))
        # soft_min = -1/beta * log_sum
        
        # Simpler stable version for positive areas:
        # The term exp(-beta * area) decays rapidly.
        # If beta is large, only the smallest area matters.
        
        # Using logsumexp from scipy.special if available, otherwise manual.
        # Manual stable logsumexp for sum(exp(-beta * areas))
        # Let m = min(areas). Then sum(exp(-beta * areas)) = exp(-beta * m) * sum(exp(-beta * (areas - m)))
        # log(sum(...)) = -beta * m + log(sum(exp(-beta * (areas - m))))
        # soft_min = -1/beta * ( -beta * m + log(sum(...)) ) = m - 1/beta * log(sum(...))
        
        m = np.min(areas)
        # areas - m is >= 0, so exp(-beta * (areas - m)) is <= 1
        # This is numerically stable.
        shifted = areas - m
        exp_terms = np.exp(-beta * shifted)
        sum_exp = np.sum(exp_terms)
        
        if sum_exp == 0:
            soft_min = m
        else:
            soft_min = m - (1.0 / beta) * np.log(sum_exp)
            
        return soft_min / A_hull

    def neg_objective_smooth(angles, beta):
        points = get_points_from_angles(angles)
        val = calculate_metric(points, beta)
        return -val

    # Initial Guess: Regular 13-gon
    theta0 = np.linspace(0, 2 * np.pi, n, endpoint=False)
    
    best_score = -1.0
    best_angles = theta0.copy()
    
    # Simulated Annealing / Gradient Descent Loop
    # We increase beta (cooling) to sharpen the minimum
    betas = [10, 50, 100, 200, 500, 1000, 2000]
    
    # We might want to try multiple random starts if time permits, 
    # but L-BFGS-B is fast. Let's stick to the regular start and maybe one perturbed start.
    
    starts = [theta0.copy()]
    
    # Add a slightly perturbed start to potentially escape the symmetric local minimum
    rng = np.random.default_rng(seed=42)
    theta_pert = theta0 + rng.normal(0, 0.1, n)
    starts.append(theta_pert)
    
    current_angles = None
    
    for beta in betas:
        elapsed = time.time() - start_time
        if elapsed > time_limit - 10:
            break
            
        for start in starts:
            if current_angles is not None:
                # Use the previous best result as the start for the next higher beta
                # This is the standard annealing approach
                x0 = current_angles
            else:
                x0 = start
                
            try:
                res = scipy.optimize.minimize(
                    neg_objective_smooth, 
                    x0, 
                    args=(beta,),
                    method='L-BFGS-B',
                    options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                # Evaluate the true metric (hard min) for the result
                # Note: The optimization used the soft metric, but we must report/track the hard metric
                # to ensure we are actually improving the target score.
                # However, for the purpose of guiding the optimization, the soft metric is sufficient.
                # We update best_angles if the soft metric improved, but we should verify with hard metric.
                
                soft_val = -res.fun
                # Calculate hard metric for this candidate
                cand_points = get_points_from_angles(res.x)
                hard_val = calculate_metric(cand_points, beta=10000) # Use large beta for "hard" eval
                
                if hard_val > best_score:
                    best_score = hard_val
                    best_angles = res.x.copy()
                    current_angles = res.x.copy()
                    
            except Exception:
                pass
                
        # After finishing a beta level, if we have a good current_angles, 
        # we can use it as the start for the next beta level for all "starts" 
        # effectively merging them into a single trajectory.
        # To keep it simple and robust, we just let the loop continue.
        # The `current_angles` will be used for the next beta.
        
    # Final Polish with Nelder-Mead on the hard objective
    # This helps to fine-tune the result since L-BFGS-B on the soft objective
    # might stop slightly away from the true hard-minimum optimum.
    
    elapsed = time.time() - start_time
    if elapsed < time_limit - 30:
        try:
            def neg_objective_hard(angles):
                points = get_points_from_angles(angles)
                return -calculate_metric(points, beta=10000)
                
            res_final = scipy.optimize.minimize(
                neg_objective_hard, 
                best_angles, 
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            
            final_hard_val = -res_final.fun
            if final_hard_val > best_score:
                best_score = final_hard_val
                best_angles = res_final.x.copy()
                
        except Exception:
            pass

    final_points = get_points_from_angles(best_angles)
    
    # Ensure validity
    if not np.all(np.isfinite(final_points)):
        # Fallback to regular polygon
        final_points = get_points_from_angles(theta0)

    return final_points

# EVOLVE-BLOCK-END
