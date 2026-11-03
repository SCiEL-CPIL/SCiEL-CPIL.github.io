import numpy as np
import time
from scipy.optimize import least_squares

N = 300

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin for 150s limit
    
    # Precompute indices for pairwise distances
    # We need to handle the O(N^2) complexity carefully.
    # For N=300, N^2 is 90,000, which is manageable.
    i_idx, j_idx = np.triu_indices(N, k=1)
    num_pairs = len(i_idx)
    
    def compute_score(pts):
        """
        Compute the minimum radius for a given set of points.
        Returns the minimum of:
        1. min_{i<j} |p_i - p_j| / 2
        2. min_i (1 - |p_i|)
        """
        # Pairwise distances
        # Using broadcasting creates an N x N x 2 array, which is ~ 300*300*2*8 bytes = 1.44 MB. Acceptable.
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        min_pair_dist = np.min(dists[i_idx, j_idx]) / 2.0
        
        # Boundary distance
        rad = np.linalg.norm(pts, axis=1)
        min_bound_dist = np.min(1.0 - rad)
        
        return min(min_pair_dist, min_bound_dist)

    def soft_min_grads(pts, k):
        """
        Compute the gradient of the soft-min objective.
        Objective: - (1/k) * log( sum(exp(-k * vals)) ) approximates min(vals)
        Gradient of min(vals) w.r.t. x is a convex combination of gradients of vals,
        weighted by the softmax of -k*vals.
        
        vals are the constraint values:
        - d_ij / 2 for pairs
        - 1 - |p_i| for boundaries
        
        We want to maximize the min, so we maximize the soft-min.
        Gradient of soft-min is sum(weights * grad(val)).
        
        For pair (i,j): val = |p_i - p_j| / 2
        grad_{p_i} (val) = (p_i - p_j) / (2 * |p_i - p_j|)
        grad_{p_j} (val) = (p_j - p_i) / (2 * |p_i - p_j|)
        
        For boundary i: val = 1 - |p_i|
        grad_{p_i} (val) = - p_i / |p_i|
        """
        # 1. Compute all constraint values
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        pair_vals = dists[i_idx, j_idx] / 2.0
        
        rad = np.linalg.norm(pts, axis=1)
        # Avoid division by zero
        rad_safe = np.where(rad > 1e-9, rad, 1.0)
        bound_vals = 1.0 - rad
        
        all_vals = np.concatenate([pair_vals, bound_vals])
        
        # 2. Compute weights using stable log-sum-exp
        # We want weights proportional to exp(-k * vals)
        # To avoid underflow/overflow, subtract max(-k*vals) which is -k*min(vals)
        # Let m = min(vals). Then -k*vals + k*m = k*(m - vals).
        # Since vals >= m, m - vals <= 0. So exp(k*(m-vals)) <= 1.
        
        min_val = np.min(all_vals)
        # Shifted values for stability
        # We want softmax(-k * vals).
        # log(softmax(-k*vals)) = -k*vals - logsumexp(-k*vals)
        # logsumexp(-k*vals) = log(sum(exp(-k*vals)))
        
        # Let's compute the unnormalized weights first: w_i = exp(-k * (vals_i - min_val))
        # This is equivalent to exp(-k*vals_i) / exp(-k*min_val)
        shifted_vals = all_vals - min_val
        # If k is large, exp(-k * shifted) underflows for large shifted.
        # This is good, it focuses on the minimum.
        weights = np.exp(-k * shifted_vals)
        
        sum_weights = np.sum(weights)
        if sum_weights == 0:
            # Fallback if all underflow (should not happen with min_val shift)
            weights = np.ones_like(all_vals) / len(all_vals)
        else:
            weights /= sum_weights
        
        # 3. Compute gradients
        grads = np.zeros_like(pts)
        
        # Pair gradients
        pair_weights = weights[:num_pairs]
        d_pair = dists[i_idx, j_idx]
        
        # Avoid division by zero
        mask = d_pair > 1e-9
        if np.any(mask):
            # grad_{p_i} = (p_i - p_j) / (2 * d)
            # grad_{p_j} = (p_j - p_i) / (2 * d)
            
            # We need to accumulate gradients for each point.
            # Vectorized approach:
            # diff[i,j] = p_i - p_j
            # term = pair_weights / (2 * d_pair)
            # grad_contrib_i = term * diff
            # grad_contrib_j = -term * diff
            
            # We only compute for non-zero distances
            idx_i = i_idx[mask]
            idx_j = j_idx[mask]
            d_masked = d_pair[mask]
            w_masked = pair_weights[mask]
            
            # Coefficient for each pair
            coeff = w_masked / (2.0 * d_masked)
            
            # Difference vectors
            diff_masked = diff[idx_i, idx_j] # shape (num_masked, 2)
            
            # Gradient contribution to i
            grad_i = coeff[:, np.newaxis] * diff_masked
            # Gradient contribution to j
            grad_j = -coeff[:, np.newaxis] * diff_masked
            
            # Accumulate using np.add.at
            np.add.at(grads, idx_i, grad_i)
            np.add.at(grads, idx_j, grad_j)
            
        # Boundary gradients
        bound_weights = weights[num_pairs:]
        # grad = - p_i / |p_i|
        # Note: if p_i is 0, gradient is undefined, but we set rad_safe to 1, so grad is 0.
        bound_grads = - (pts / rad_safe[:, np.newaxis]) * bound_weights[:, np.newaxis]
        grads += bound_grads
        
        return grads

    def optimize(points, time_budget):
        """
        Run gradient ascent optimization on a copy of points.
        Returns the best points found.
        """
        points = points.copy()
        best_points = points.copy()
        best_score = compute_score(points)
        
        # Parameters
        alpha = 0.01
        k = 10.0
        stall_count = 0
        stall_threshold = 50
        max_iter = 20000
        iter_count = 0
        
        # Initial k is small (smooth), we increase it to sharpen
        k_start = 5.0
        k_end = 100.0
        
        while time.time() - start_time < time_budget and iter_count < max_iter:
            # Update k
            progress = min(1.0, iter_count / max_iter)
            k = k_start + (k_end - k_start) * progress
            
            # Compute gradient
            grads = soft_min_grads(points, k)
            
            # Normalize gradient to prevent huge steps
            grad_norm = np.linalg.norm(grads)
            if grad_norm > 0:
                grads /= grad_norm
            
            # Update
            points += alpha * grads
            
            # Project inside circle
            rad = np.linalg.norm(points, axis=1)
            mask = rad > 1.0
            if np.any(mask):
                points[mask] /= rad[mask, np.newaxis]
            
            # Evaluate score
            score = compute_score(points)
            
            if score > best_score:
                best_score = score
                best_points = points.copy()
                stall_count = 0
                # Decay alpha slowly
                alpha *= 0.9999
            else:
                stall_count += 1
                # If stalled, perturb and reset alpha
                if stall_count > stall_threshold:
                    # Random perturbation
                    noise_scale = 0.01
                    points += np.random.normal(0, noise_scale, size=points.shape)
                    rad = np.linalg.norm(points, axis=1)
                    mask = rad > 1.0
                    if np.any(mask):
                        points[mask] /= rad[mask, np.newaxis]
                    
                    # Reset alpha
                    alpha = 0.01
                    stall_count = 0
                    k = k_start # Reset k to smooth again
            
            iter_count += 1
            
            # Periodic re-evaluation of best to ensure we don't lose track
            # (Not strictly necessary if we track best_score, but good for safety)
            
        return best_points, best_score

    # Multi-Start Strategy
    # We will try several initializations.
    # 1. Hexagonal lattice
    # 2. Random uniform
    # 3. Hexagonal lattice with rotation
    
    best_points_overall = None
    best_score_overall = 0.0
    
    # Time allocation
    # We have ~145s. Let's allocate time to different starts.
    # Start 1: Hex lattice (most likely to be good) - 50% time
    # Start 2: Random (explore) - 20% time
    # Start 3: Hex lattice rotated - 20% time
    # Start 4: Random - 10% time
    
    # Helper to generate hex lattice
    def generate_hex_lattice(n_points, radius=1.0):
        s_init = np.sqrt(2 * np.pi / (np.sqrt(3) * n_points))
        s_gen = s_init * 0.95
        y_step = s_gen * np.sqrt(3) / 2
        max_j = int(1.0 / y_step) + 2
        max_i = int(1.0 / s_gen) + 2
        
        points_list = []
        for j in range(-max_j, max_j + 1):
            y = j * y_step
            offset = 0 if j % 2 == 0 else s_gen / 2
            for i in range(-max_i, max_i + 1):
                x = i * s_gen + offset
                if np.hypot(x, y) < 1.0:
                    points_list.append([x, y])
                    
        points = np.array(points_list)
        
        if len(points) > n_points:
            # Select closest to center
            dists = np.linalg.norm(points, axis=1)
            idx = np.argsort(dists)[:n_points]
            points = points[idx]
        elif len(points) < n_points:
            # Add random points
            while len(points) < n_points:
                p = np.random.uniform(-1, 1, 2)
                if np.linalg.norm(p) < 1.0:
                    points = np.vstack([points, p])
            points = points[:n_points]
            
        return points

    # Start 1: Hex Lattice
    t_start = time.time()
    points_hex = generate_hex_lattice(N)
    # Add small random noise to break symmetry
    points_hex += np.random.normal(0, 0.001, size=points_hex.shape)
    # Project
    rad = np.linalg.norm(points_hex, axis=1)
    mask = rad > 1.0
    if np.any(mask):
        points_hex[mask] /= rad[mask, np.newaxis]
        
    time_budget_1 = (time_limit - (time.time() - start_time)) * 0.5
    if time_budget_1 > 5:
        pts, score = optimize(points_hex, time_budget_1)
        if score > best_score_overall:
            best_score_overall = score
            best_points_overall = pts

    # Start 2: Random
    t_start = time.time()
    points_rand = np.random.uniform(-1, 1, size=(N, 2))
    rad = np.linalg.norm(points_rand, axis=1)
    mask = rad > 1.0
    if np.any(mask):
        points_rand[mask] /= rad[mask, np.newaxis]
        
    time_budget_2 = (time_limit - (time.time() - start_time)) * 0.3
    if time_budget_2 > 5:
        pts, score = optimize(points_rand, time_budget_2)
        if score > best_score_overall:
            best_score_overall = score
            best_points_overall = pts

    # Start 3: Hex Lattice Rotated
    t_start = time.time()
    points_hex_rot = generate_hex_lattice(N)
    angle = np.pi / 6 # 30 degrees
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rot_mat = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    points_hex_rot = points_hex_rot @ rot_mat.T
    points_hex_rot += np.random.normal(0, 0.001, size=points_hex_rot.shape)
    rad = np.linalg.norm(points_hex_rot, axis=1)
    mask = rad > 1.0
    if np.any(mask):
        points_hex_rot[mask] /= rad[mask, np.newaxis]
        
    time_budget_3 = (time_limit - (time.time() - start_time)) * 0.3
    if time_budget_3 > 5:
        pts, score = optimize(points_hex_rot, time_budget_3)
        if score > best_score_overall:
            best_score_overall = score
            best_points_overall = pts

    # Final Polish: Use the best points found and run a final, high-precision optimization
    # using a smaller step size and higher k.
    if best_points_overall is not None:
        t_start = time.time()
        remaining_time = time_limit - (time.time() - start_time)
        if remaining_time > 10:
            # Use a tighter optimization loop
            # We can reuse the optimize function but with different params?
            # Or just run optimize again on the best result with the remaining time.
            # The optimize function adapts k and alpha, so it's safe to run again.
            pts, score = optimize(best_points_overall, remaining_time * 0.9)
            if score > best_score_overall:
                best_score_overall = score
                best_points_overall = pts

    return best_points_overall
