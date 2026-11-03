import numpy as np
import time
from scipy.optimize import minimize

N = 300
TIME_LIMIT = 115.0

def generate_hex_init():
    """
    Generate a dense hexagonal lattice, clip to unit circle,
    and select the N points closest to the center.
    """
    spacing = 0.09
    k = int(1.1 / spacing) + 2
    points = []
    for i in range(-k, k + 1):
        for j in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            points.append([x, y])
    
    points = np.array(points)
    r = np.hypot(points[:, 0], points[:, 1])
    
    mask = r < 0.99
    points = points[mask]
    r = r[mask]
    
    idx = np.argsort(r)
    points = points[idx]
    
    if len(points) < N:
        theta = np.random.uniform(0, 2 * np.pi, N)
        rho = np.sqrt(np.random.uniform(0, 0.9, N))
        points = np.column_stack((rho * np.cos(theta), rho * np.sin(theta)))
    else:
        points = points[:N]
        
    return points

def compute_smooth_score(points, k=100.0):
    """
    Compute a smooth approximation of the minimum radius.
    We maximize this score.
    Score = - (1/k) * log( sum(exp(-k * d_ij)) + sum(exp(-k * b_i)) )
    
    This is equivalent to -softmin(k * distances).
    As k -> infinity, it approaches -min(distances).
    We use a moderate k to keep the function smooth enough for optimization
    but sharp enough to approximate the true minimum.
    """
    # Pairwise distances
    diffs = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    
    # We only need the upper triangle to avoid double counting, 
    # but for the log-sum-exp stability, we can just take all i<j.
    # To save memory and time, we can compute the sum of exps directly.
    # However, N=300 means ~45k pairs. This is manageable.
    
    # Get upper triangle indices
    i_upper, j_upper = np.triu_indices(N, k=1)
    pair_dists = dists[i_upper, j_upper]
    
    # Boundary distances
    radii = np.linalg.norm(points, axis=1)
    boundary_dists = 1.0 - radii
    
    # Combine all constraints into one array
    # We want to maximize the minimum of these.
    # Let's define the "violations" or "margins".
    # We want to maximize min(pair_dists/2, boundary_dists).
    # Let's work with the actual radius values.
    
    # To avoid underflow/overflow in exp, we use the log-sum-exp trick.
    # LSE(x) = max(x) + log(sum(exp(x - max(x))))
    # We want -LSE(-k * values) / k ? No.
    # Softmin(x) = -1/k * log(sum(exp(-k*x)))
    # We want to maximize Softmin(x).
    
    # Values to minimize: pair_dists/2 and boundary_dists.
    # Let's collect them.
    vals = np.concatenate([pair_dists / 2.0, boundary_dists])
    
    # To prevent numerical issues with very large k * vals, we shift by max.
    # Actually, we want to find the minimum.
    # Let m = min(vals).
    # Softmin approx m.
    # Function to maximize: f(x) = -1/k * log(sum(exp(-k * x)))
    # Let's compute log(sum(exp(-k * x))).
    # Let y = -k * x.
    # log(sum(exp(y))) = max(y) + log(sum(exp(y - max(y))))
    # max(y) = -k * min(x)
    
    min_val = np.min(vals)
    # Shift x by min_val so that the max exponent is 0
    shifted_vals = vals - min_val
    # exp(-k * shifted_vals) is <= 1, so no overflow.
    log_sum_exp_term = np.log(np.sum(np.exp(-k * shifted_vals)))
    
    # Softmin = min_val - (1/k) * log_sum_exp_term
    # Wait, the formula is:
    # -1/k * log(sum(exp(-k * x)))
    # = -1/k * ( -k * min_val + log(sum(exp(-k * (x - min_val)))) )
    # = min_val - (1/k) * log(sum(exp(-k * (x - min_val))))
    
    smooth_score = min_val - (1.0 / k) * log_sum_exp_term
    
    return smooth_score

def get_min_gap_and_indices(points):
    """
    Returns the exact minimum radius (for final evaluation and tracking).
    """
    diffs = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    
    min_pair = np.min(dists) / 2.0
    
    radii = np.linalg.norm(points, axis=1)
    min_boundary_dist = np.min(1.0 - radii)
    
    return min(min_pair, min_boundary_dist)

def construct():
    start_time = time.time()
    
    # 1. Initialization
    best_points = generate_hex_init()
    best_score = get_min_gap_and_indices(best_points)
    
    # Parameters for Nelder-Mead
    # We optimize a flattened array of 2*N coordinates.
    # Method: 'Nelder-Mead' is derivative-free and works well for moderate dimensions.
    # However, N=300 is 600 dimensions. Nelder-Mead can be slow in high dimensions.
    # Alternative: Use 'Powell' or a custom gradient descent if we had gradients.
    # Given the time limit, we will use Nelder-Mead with a limited number of iterations
    # and multiple restarts.
    
    # To make Nelder-Mead more effective in high dimensions, we might need to
    # reduce the effective dimensionality or use a better method.
    # Let's try 'Nelder-Mead' first. If it's too slow, we might need to switch to 
    # a physics-based relaxation which is O(N) per step vs O(N^2) for distance matrix.
    
    # Actually, computing the smooth score requires O(N^2) distance calculation.
    # N=300 -> 90,000 distances. This is fast in NumPy (~1ms).
    # Nelder-Mead in 600D might take many iterations.
    # Let's try a different approach: **Force-directed relaxation** is often faster 
    # and more robust for packing problems because it handles local interactions well 
    # and doesn't suffer from the "curse of dimensionality" as badly as simplex methods 
    # in high-D.
    
    # Let's pivot the strategy slightly to **Physics-based Force-Directed Optimization** 
    # which is a distinct family from SA and NM. It uses repulsive forces between circles 
    # and attractive forces towards the center/boundary to find a stable equilibrium.
    
    def force_directed_optimize(points, time_budget):
        """
        Optimize points using a physics-based force-directed approach.
        Force = Repulsion (if overlapping) + Attraction (to center/boundary).
        """
        current_points = points.copy()
        current_score = get_min_gap_and_indices(current_points)
        best_local_score = current_score
        best_local_points = current_points.copy()
        
        # Adaptive step size
        step_size = 0.01
        cooling = 0.999
        min_step = 1e-5
        
        start_fd = time.time()
        
        while time.time() - start_fd < time_budget and step_size > min_step:
            # Calculate forces
            # We approximate the gradient of the smooth score or use physical intuition.
            # Force on i:
            # 1. Repulsion from j if dist < 2*r_target. We don't know r_target yet.
            #    Instead, we push all pairs apart to maximize min distance.
            #    Force is proportional to 1/d^2 or similar, directed away.
            # 2. Attraction to boundary if close to center? No, we want to fill space.
            #    Actually, for packing, we want to maximize the minimum distance.
            #    This is equivalent to minimizing the potential energy of a system of 
            #    repulsive particles in a container.
            
            # Let's use a simple repulsive force model:
            # F_ij = k * (1/d_ij^2) * (p_i - p_j)/d_ij  for d_ij < threshold
            # Threshold can be dynamic, e.g., current_best_score * 2 + margin.
            
            forces = np.zeros_like(current_points)
            
            # Compute pairwise distances and forces
            # This is O(N^2).
            diffs = current_points[:, np.newaxis, :] - current_points[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diffs**2, axis=-1))
            np.fill_diagonal(dists, np.inf)
            
            # We want to push points apart.
            # Let's define a target distance based on current best score.
            # target_r = best_local_score
            # If dist < 2 * target_r, apply force.
            # To make it smooth, we can use a potential like V(d) = 1/d^2 or exp(-d).
            
            # Let's use a force that is active for all pairs but stronger for close ones.
            # F = (1/d^3) * diff  (Coulomb-like)
            # This is expensive to compute for all pairs if we do it explicitly in a loop,
            # but vectorized it's fine.
            
            # Avoid division by zero
            safe_dists = np.where(dists < 1e-6, 1e-6, dists)
            
            # Force magnitude: 1/d^2 is strong. Let's use 1/d^3 to be smoother?
            # Actually, for packing, we want to maximize min distance.
            # The gradient of min distance is only non-zero for the closest pairs.
            # A smooth surrogate gradient would be a weighted sum of all pair gradients.
            # Weight w_ij = exp(-k * d_ij).
            
            k_force = 50.0
            weights = np.exp(-k_force * safe_dists)
            np.fill_diagonal(weights, 0)
            
            # Force on i from j is w_ij * (p_i - p_j) / d_ij
            # We need to sum over j.
            # F_i = sum_j w_ij * (p_i - p_j) / d_ij
            # Note: (p_i - p_j) / d_ij is the unit vector from j to i.
            
            # Vectorized computation:
            # unit_vecs = diffs / dists[..., np.newaxis]
            # forces = np.sum(weights[..., np.newaxis] * unit_vecs, axis=1)
            
            unit_vecs = diffs / safe_dists[..., np.newaxis]
            forces = np.sum(weights[..., np.newaxis] * unit_vecs, axis=1)
            
            # Boundary force:
            # If a point is too close to the boundary (radius > 1 - best_local_score),
            # push it inwards.
            radii = np.linalg.norm(current_points, axis=1)
            boundary_margin = 1.0 - radii
            # If boundary_margin < best_local_score, push in.
            # Force = -k_b * (boundary_margin - best_local_score) * (p_i / |p_i|)
            # Simplified: Just push all points slightly towards center to prevent escaping,
            # but the repulsion should handle the packing.
            # Let's add a small centripetal force to keep them in.
            center_force = -0.01 * current_points
            
            total_forces = forces + center_force
            
            # Update positions
            # Limit step size to avoid instability
            max_step = step_size
            force_mags = np.linalg.norm(total_forces, axis=1)
            scale = np.where(force_mags > 1e-6, max_step / np.maximum(force_mags, 1e-6), 0)
            
            new_points = current_points + total_forces * scale[:, np.newaxis]
            
            # Check validity: points must be inside unit circle
            new_radii = np.linalg.norm(new_points, axis=1)
            # If any point is outside, project it back or reject step
            if np.any(new_radii > 1.0):
                # Project back
                scale_back = np.where(new_radii > 1.0, 0.99 / new_radii, 1.0)
                new_points = new_points * scale_back[:, np.newaxis]
            
            new_score = get_min_gap_and_indices(new_points)
            
            if new_score > best_local_score:
                best_local_score = new_score
                best_local_points = new_points.copy()
                current_points = new_points
                # If improved, we can slightly increase step size or keep same
                # step_size = min(step_size * 1.05, 0.05)
            else:
                # If not improved, reduce step size
                step_size *= cooling
                # Optionally, revert to best_local_points if stuck?
                # For now, just keep current_points if it's not worse than best?
                # Actually, if new_score < current_score, we should probably revert
                # to avoid drifting into bad regions.
                if new_score < current_score:
                    current_points = best_local_points.copy()
                else:
                    current_points = new_points
                    current_score = new_score
            
            # Check time
            if time.time() - start_time > TIME_LIMIT - 5:
                break
        
        return best_local_points, best_local_score

    # Multi-start strategy
    # We have 115s. 
    # Let's do 5 restarts with 20s each.
    num_restarts = 5
    time_per_restart = (TIME_LIMIT - 5) / num_restarts
    
    for i in range(num_restarts):
        if time.time() - start_time > TIME_LIMIT - 5:
            break
        
        # Generate new initial point
        # Mix of hex and random
        if i % 2 == 0:
            init_points = generate_hex_init()
        else:
            # Random uniform in circle
            theta = np.random.uniform(0, 2 * np.pi, N)
            rho = np.sqrt(np.random.uniform(0, 0.95, N))
            init_points = np.column_stack((rho * np.cos(theta), rho * np.sin(theta)))
        
        # Optimize
        local_points, local_score = force_directed_optimize(init_points, time_per_restart)
        
        if local_score > best_score:
            best_score = local_score
            best_points = local_points
    
    # Final cleanup
    radii = np.linalg.norm(best_points, axis=1)
    valid = np.isfinite(best_points).all(axis=1) & (radii < 1.0)
    if not np.all(valid):
        theta = np.random.uniform(0, 2 * np.pi, N)
        rho = np.sqrt(np.random.uniform(0, 0.9, N))
        best_points = np.column_stack((rho * np.cos(theta), rho * np.sin(theta)))

    return best_points
