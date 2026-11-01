import numpy as np
import time

N = 300

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    # Step 1: Generate hexagonal lattice points
    def hex_lattice_init(n):
        best_points = None
        best_min_dist = 0
        for trial in range(30):
            spacing = 2.0 / (np.sqrt(n) + 1.0) * (1.0 - trial * 0.015)
            if spacing <= 0:
                continue
            k = int(1.0 / spacing) + 2
            points = []
            for j in range(-k, k + 1):
                for i in range(-k, k + 1):
                    x = (i + 0.5 * (j % 2)) * spacing
                    y = j * spacing * np.sqrt(3.0) / 2.0
                    r = np.hypot(x, y)
                    if r <= 1.0 - spacing / 2.0:
                        points.append([x, y])
            if len(points) < n:
                continue
            points = np.array(points)
            dists = np.hypot(points[:, 0], points[:, 1])
            idx = np.argsort(dists)[:n]
            pts_sel = points[idx]
            # Compute min pair distance for this selection
            from scipy.spatial import KDTree
            tree = KDTree(pts_sel)
            pairs = tree.query_pairs(spacing * 1.5)
            if len(pairs) > 0:
                pair_dists = [np.linalg.norm(pts_sel[a] - pts_sel[b]) for a, b in pairs]
                min_pair = min(pair_dists)
            else:
                min_pair = spacing
            min_boundary = 1.0 - np.max(dists[idx])
            min_r = min(min_pair / 2.0, min_boundary)
            if best_points is None or min_r > best_min_dist:
                best_min_dist = min_r
                best_points = pts_sel.copy()
        return best_points

    pts = hex_lattice_init(N)
    
    # Step 2: Compute initial radius
    def compute_radius(pts):
        min_pair = np.min(np.sqrt(np.sum((pts[:, np.newaxis, :] - pts[np.newaxis, :, :]) ** 2, axis=-1).ravel())) / 2.0
        min_boundary = np.min(1.0 - np.hypot(pts[:, 0], pts[:, 1]))
        return min(min_pair, min_boundary)

    best_radius = compute_radius(pts)
    best_pts = pts.copy()
    
    # Step 3: Vectorized optimization using t-epigraph formulation
    # We maximize t subject to pairwise distances >= 2t and boundary distance >= t
    # Smooth surrogate: maximize soft_min( [d_ij/2, 1-|p_i|] )
    # Using log-sum-exp: soft_min(x) = min(x) - T * log(sum(exp(-(x_i - min(x))/T)))
    # To optimize t, we use the epigraph: maximize t - penalty * sum(max(0, 2t - d_ij)^2 + max(0, t - (1-|p_i|))^2)
    # Or simpler: use gradient ascent on soft_min.
    
    # Let's use a direct gradient ascent on the smooth soft-min objective.
    # Objective: F(p) = soft_min( [d_ij/2 for all i<j, 1-|p_i| for all i] )
    # Gradient of soft_min with respect to p_i:
    # Let x = [x_1, ..., x_M] where x_k are the constraints (d_ij/2 or 1-|p_i|)
    # soft_min(x) = min(x) - T * log( sum_k exp(-(x_k - min(x))/T) )
    # d(soft_min)/dx_k = - exp(-(x_k - min(x))/T) / sum_j exp(-(x_j - min(x))/T)
    # Then chain rule to p.
    
    # To make it faster, we can compute the weights w_k = exp(-(x_k - min(x))/T) / sum(exp(...))
    # Then grad_p = sum_k w_k * (d x_k / d p)
    
    # We will use a fixed T that decays slowly.
    
    T_init = 0.005
    T_end = 0.0005
    total_iters = 3000
    best_soft_radius = best_radius
    
    for it in range(total_iters):
        if time.time() > deadline:
            break
            
        # Decay temperature
        progress = it / total_iters
        T = T_init * (T_end / T_init) ** progress
        
        # Compute all pairwise distances and boundary distances
        # d_ij = |p_i - p_j|
        # d_boundary_i = 1 - |p_i|
        
        # Vectorized computation of pairwise distances
        # p has shape (N, 2)
        # diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]  # (N, N, 2)
        # dists = np.sqrt(np.sum(diff ** 2, axis=-1))  # (N, N)
        
        # To save memory, we can compute in chunks or use a more efficient method.
        # For N=300, N^2=90000, which is fine.
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=-1))
        
        # Boundary distances
        r_pts = np.hypot(pts[:, 0], pts[:, 1])
        boundary_dists = 1.0 - r_pts
        
        # Combine all constraints
        # x is a vector of all constraint values
        # Pairwise: d_ij / 2 for i < j
        # Boundary: 1 - |p_i|
        
        # We need the gradient of soft_min with respect to pts.
        # Let's compute the weights w for each constraint.
        
        # x_pairs = dists / 2, but only upper triangle
        # x_bound = boundary_dists
        
        # To avoid duplicating work, let's compute the soft_min and its gradient in one go.
        
        # min_val = min( min(dists/2 for i<j), min(boundary_dists) )
        min_pair = np.min(dists) / 2.0  # Note: dists[i,i]=0, so we need to exclude diagonal
        # Actually, dists has 0 on diagonal. We need to ignore them.
        # Let's mask the diagonal.
        mask = ~np.eye(N, dtype=bool)
        min_pair = np.min(dists[mask]) / 2.0
        min_bound = np.min(boundary_dists)
        min_val = min(min_pair, min_bound)
        
        # Compute weights
        # w_k = exp(-(x_k - min_val)/T) / sum(exp(-(x_j - min_val)/T))
        
        # For pairwise:
        # x_k = dists[i,j] / 2 for i < j
        # d(x_k)/d(p_i) = (p_i - p_j) / dists[i,j]
        # d(x_k)/d(p_j) = (p_j - p_i) / dists[i,j]
        
        # For boundary:
        # x_k = 1 - |p_i|
        # d(x_k)/d(p_i) = - p_i / |p_i|
        
        # Let's compute the sum of exponentials first.
        # sum_exp = sum_{i<j} exp(-(dists[i,j]/2 - min_val)/T) + sum_i exp(-(boundary_dists[i] - min_val)/T)
        
        # To avoid overflow, we shift by min_val.
        # exp(-(x_k - min_val)/T) = exp(-x_k/T) * exp(min_val/T)
        # The exp(min_val/T) factor cancels in the ratio.
        
        # Let's compute unnormalized weights.
        # w_pairs[i,j] = exp(-(dists[i,j]/2 - min_val)/T) for i < j
        # w_bound[i] = exp(-(boundary_dists[i] - min_val)/T)
        
        # Compute dists/2
        dists_half = dists / 2.0
        np.fill_diagonal(dists_half, np.inf)  # Exclude diagonal
        
        # Compute exponents
        exp_pairs = np.exp(-(dists_half - min_val) / T)
        exp_pairs[np.isinf(dists_half)] = 0.0
        
        exp_bound = np.exp(-(boundary_dists - min_val) / T)
        
        sum_exp = np.sum(exp_pairs) + np.sum(exp_bound)
        
        if sum_exp < 1e-10:
            break
            
        # Normalized weights
        w_pairs = exp_pairs / sum_exp
        w_bound = exp_bound / sum_exp
        
        # Compute gradient
        # grad = sum_{i<j} w_pairs[i,j] * (d x_pairs[i,j]/d p) + sum_i w_bound[i] * (d x_bound[i]/d p)
        
        # d x_pairs[i,j]/d p_i = (p_i - p_j) / dists[i,j]
        # d x_pairs[i,j]/d p_j = (p_j - p_i) / dists[i,j]
        
        # Note: dists[i,j] = 2 * dists_half[i,j]
        # So (p_i - p_j) / dists[i,j] = (p_i - p_j) / (2 * dists_half[i,j])
        
        # Let's compute the force on each point.
        # force_i = sum_{j != i} w_pairs[i,j] * (p_i - p_j) / (2 * dists_half[i,j]) - w_bound[i] * p_i / r_pts[i]
        
        # Note: w_pairs is symmetric, w_pairs[i,j] = w_pairs[j,i]
        
        # We can compute this as:
        # force_i = sum_j (w_pairs[i,j] * (p_i - p_j) / (2 * dists_half[i,j]))
        
        # Let's vectorize this.
        # diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]  # (N, N, 2)
        # w_pairs / (2 * dists_half) has shape (N, N)
        # We need to multiply each row i by w_pairs[i,j] / (2 * dists_half[i,j]) and sum over j.
        
        # factor[i,j] = w_pairs[i,j] / (2 * dists_half[i,j])
        # But dists_half[i,i] is inf, so factor[i,i] = 0.
        
        # To avoid division by zero, we can use np.where.
        factor = np.zeros_like(dists_half)
        mask_nonzero = dists_half > 1e-10
        factor[mask_nonzero] = w_pairs[mask_nonzero] / (2 * dists_half[mask_nonzero])
        
        # force_i = sum_j factor[i,j] * (p_i - p_j)
        # = sum_j factor[i,j] * p_i - sum_j factor[i,j] * p_j
        # = p_i * sum_j factor[i,j] - sum_j factor[i,j] * p_j
        
        sum_factor = np.sum(factor, axis=1)  # (N,)
        # sum_j factor[i,j] * p_j = factor @ p  (N, N) @ (N, 2) -> (N, 2)
        force_pairs = pts * sum_factor[:, np.newaxis] - (factor @ pts)
        
        # Boundary force
        # - w_bound[i] * p_i / r_pts[i]
        # r_pts[i] = |p_i|
        # If r_pts[i] is 0, the force is 0 (or undefined, but we can set to 0).
        r_safe = np.where(r_pts < 1e-10, 1.0, r_pts)
        force_bound = -w_bound[:, np.newaxis] * pts / r_safe[:, np.newaxis]
        
        total_force = force_pairs + force_bound
        
        # Adaptive step size
        # We want to increase the soft_min radius.
        # The gradient of soft_min is total_force.
        # We take a step in the direction of total_force.
        
        # To avoid overshooting, we can use a small step size.
        # Let's use a step size that decreases over time.
        step_size = 0.01 * (1.0 - progress) + 0.001
        
        # Normalize the force to prevent large steps
        force_norm = np.linalg.norm(total_force, axis=1, keepdims=True)
        force_norm = np.where(force_norm < 1e-10, 1.0, force_norm)
        total_force_normalized = total_force / force_norm
        
        pts_new = pts + step_size * total_force_normalized
        
        # Keep points inside the disk
        r_new = np.hypot(pts_new[:, 0], pts_new[:, 1])
        mask = r_new > 0.999
        if np.any(mask):
            pts_new[mask] = pts_new[mask] / r_new[mask, np.newaxis] * 0.999
        
        # Check if this improves the radius
        r_new_val = compute_radius(pts_new)
        if r_new_val > best_radius:
            best_radius = r_new_val
            best_pts = pts_new.copy()
            pts = pts_new.copy()
        else:
            # Still accept for exploration but with smaller effective step
            pts = pts_new.copy()
            
    # Final polish: use a smaller step size and more iterations
    if time.time() < deadline - 10:
        # Use a smaller T and more iterations
        T = 0.0005
        for it in range(1000):
            if time.time() > deadline - 5:
                break
            
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff ** 2, axis=-1))
            
            r_pts = np.hypot(pts[:, 0], pts[:, 1])
            boundary_dists = 1.0 - r_pts
            
            min_pair = np.min(dists[~np.eye(N, dtype=bool)]) / 2.0
            min_bound = np.min(boundary_dists)
            min_val = min(min_pair, min_bound)
            
            dists_half = dists / 2.0
            np.fill_diagonal(dists_half, np.inf)
            
            exp_pairs = np.exp(-(dists_half - min_val) / T)
            exp_pairs[np.isinf(dists_half)] = 0.0
            exp_bound = np.exp(-(boundary_dists - min_val) / T)
            
            sum_exp = np.sum(exp_pairs) + np.sum(exp_bound)
            if sum_exp < 1e-10:
                break
                
            w_pairs = exp_pairs / sum_exp
            w_bound = exp_bound / sum_exp
            
            factor = np.zeros_like(dists_half)
            mask_nonzero = dists_half > 1e-10
            factor[mask_nonzero] = w_pairs[mask_nonzero] / (2 * dists_half[mask_nonzero])
            
            sum_factor = np.sum(factor, axis=1)
            force_pairs = pts * sum_factor[:, np.newaxis] - (factor @ pts)
            
            r_safe = np.where(r_pts < 1e-10, 1.0, r_pts)
            force_bound = -w_bound[:, np.newaxis] * pts / r_safe[:, np.newaxis]
            
            total_force = force_pairs + force_bound
            
            step_size = 0.005
            force_norm = np.linalg.norm(total_force, axis=1, keepdims=True)
            force_norm = np.where(force_norm < 1e-10, 1.0, force_norm)
            total_force_normalized = total_force / force_norm
            
            pts_new = pts + step_size * total_force_normalized
            
            r_new = np.hypot(pts_new[:, 0], pts_new[:, 1])
            mask = r_new > 0.999
            if np.any(mask):
                pts_new[mask] = pts_new[mask] / r_new[mask, np.newaxis] * 0.999
            
            r_new_val = compute_radius(pts_new)
            if r_new_val > best_radius:
                best_radius = r_new_val
                best_pts = pts_new.copy()
                pts = pts_new.copy()
            else:
                pts = pts_new.copy()
                
    return best_pts
