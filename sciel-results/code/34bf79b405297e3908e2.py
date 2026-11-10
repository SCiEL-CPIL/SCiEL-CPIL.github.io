import numpy as np
import time
from scipy.optimize import minimize

def _fast_dist_matrix(points):
    """Compute pairwise distance matrix efficiently."""
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))

def compute_metric(points):
    """
    Computes the metric described in the task contract.
    points: (N, 2) array
    Returns: (metric, is_valid)
    """
    N = points.shape[0]
    if N != 20:
        return 0.0, False
    
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    dists_to_centroid = np.linalg.norm(centered, axis=1)
    mean_dist = np.mean(dists_to_centroid)
    if mean_dist <= 0.01:
        return 0.0, False
    
    normalized = centered / mean_dist
    
    # Check strictly convex
    edges = np.roll(normalized, -1, axis=0) - normalized
    cross = edges[:, 0] * np.roll(edges, -1, axis=0)[:, 1] - edges[:, 1] * np.roll(edges, -1, axis=0)[:, 0]
    if np.any(np.abs(cross) < 1e-12):
        return 0.0, False
    if not (np.all(cross > 0) or np.all(cross < 0)):
        return 0.0, False
    
    # Check for repeated points using distance matrix
    dist_matrix = _fast_dist_matrix(normalized)
    np.fill_diagonal(dist_matrix, 0.0)
    min_pair_dist = np.min(dist_matrix[~np.eye(N, dtype=bool)])
    if min_pair_dist < 1e-6:
        return 0.0, False
    
    np.fill_diagonal(dist_matrix, np.inf)
    
    min_score = np.inf
    for i in range(N):
        d = dist_matrix[i]
        d_sorted = np.sort(d)
        
        if d_sorted[0] < 0.05:
            score = 0.0
        else:
            diffs = np.diff(d_sorted)
            min_diff_idx = np.argmin(diffs)
            k = min_diff_idx
            D = (d_sorted[k] + d_sorted[k+1]) / 2.0
            
            if D < 0.1:
                score = 0.0
            else:
                if k == 0:
                    selected = d_sorted[0:4]
                elif k == 17:
                    selected = d_sorted[15:19]
                else:
                    selected = d_sorted[k-1:k+3]
                
                ratios = np.maximum(selected / D, D / selected)
                raw = np.mean(ratios)
                score = min(1.0, 1.0 / raw) if raw > 0 else 0.0
        
        if score < min_score:
            min_score = score
            
    return min_score, True

def compute_metric_smooth(points, temp=0.02):
    """
    Smooth version of the metric for gradient-based optimization.
    Uses smooth approximations of min and argmin operations.
    """
    N = points.shape[0]
    if N != 20:
        return 0.0, False
    
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    dists_to_centroid = np.linalg.norm(centered, axis=1)
    mean_dist = np.mean(dists_to_centroid)
    if mean_dist <= 0.01:
        return 0.0, False
    
    normalized = centered / mean_dist
    
    # Check strictly convex
    edges = np.roll(normalized, -1, axis=0) - normalized
    cross = edges[:, 0] * np.roll(edges, -1, axis=0)[:, 1] - edges[:, 1] * np.roll(edges, -1, axis=0)[:, 0]
    if np.any(np.abs(cross) < 1e-12):
        return 0.0, False
    if not (np.all(cross > 0) or np.all(cross < 0)):
        return 0.0, False
    
    # Check for repeated points using distance matrix
    dist_matrix = _fast_dist_matrix(normalized)
    np.fill_diagonal(dist_matrix, 0.0)
    min_pair_dist = np.min(dist_matrix[~np.eye(N, dtype=bool)])
    if min_pair_dist < 1e-6:
        return 0.0, False
    
    np.fill_diagonal(dist_matrix, np.inf)
    
    scores = np.zeros(N)
    for i in range(N):
        d = dist_matrix[i]
        d_sorted = np.sort(d)
        
        # Smooth min for d_0
        d0_smooth = -temp * np.log(np.sum(np.exp(-d_sorted / temp)))
        if d0_smooth < 0.05:
            scores[i] = 0.0
            continue
        
        # Smooth argmin for diffs
        diffs = np.diff(d_sorted)
        diffs_clipped = np.clip(diffs, 1e-10, None)
        weights = np.exp(-diffs_clipped / temp)
        weights /= np.sum(weights)
        
        # Smooth D via weighted average of candidate D values
        D_candidates = np.zeros(18)
        for k_idx in range(18):
            D_candidates[k_idx] = (d_sorted[k_idx] + d_sorted[k_idx + 1]) / 2.0
        D_smooth = np.sum(weights * D_candidates)
        
        if D_smooth < 0.1:
            scores[i] = 0.0
            continue
        
        # Select four distances: use the most probable window with soft blending
        # Find the most likely k
        k_most_likely = np.argmax(weights)
        
        # Get the primary window
        if k_most_likely == 0:
            primary_sel = d_sorted[0:4].copy()
        elif k_most_likely == 17:
            primary_sel = d_sorted[15:19].copy()
        else:
            primary_sel = d_sorted[k_most_likely-1:k_most_likely+3].copy()
        
        # Blend with neighboring windows based on weights
        blend_sel = primary_sel.copy()
        for j in range(18):
            if j == k_most_likely:
                continue
            if j == 0:
                sel_j = d_sorted[0:4]
            elif j == 17:
                sel_j = d_sorted[15:19]
            else:
                sel_j = d_sorted[j-1:j+3]
            blend_sel += weights[j] * sel_j
        
        # Normalize: primary gets weight 1, others get their weights
        # Actually, let's just use weighted average but with sharpened weights
        weights_sharp = weights ** 2
        weights_sharp /= np.sum(weights_sharp)
        
        window_sums = np.zeros(4)
        for j in range(18):
            if j == 0:
                sel = d_sorted[0:4]
            elif j == 17:
                sel = d_sorted[15:19]
            else:
                sel = d_sorted[j-1:j+3]
            window_sums += weights_sharp[j] * sel
        
        selected = window_sums
        
        # Smooth max(d/D, D/d) using soft-max
        ratio1 = selected / D_smooth
        ratio2 = D_smooth / selected
        log_ratio1 = np.log(np.clip(ratio1, 1e-10, None))
        log_ratio2 = np.log(np.clip(ratio2, 1e-10, None))
        max_log = np.maximum(log_ratio1, log_ratio2)
        soft_max = max_log + temp * np.log(np.exp(log_ratio1 - max_log) + np.exp(log_ratio2 - max_log))
        soft_max_ratio = np.exp(soft_max)
        
        raw = np.mean(soft_max_ratio)
        scores[i] = min(1.0, 1.0 / raw) if raw > 0 else 0.0
    
    # Smooth minimum over vertices
    min_score_smooth = -temp * np.log(np.sum(np.exp(-scores / temp)))
    return max(0.0, min_score_smooth), True

def equidistant_points_polygon20() -> np.ndarray:
    """
    Model-and-optimize approach: parameterize by radii and angular offsets,
    then use gradient-based and derivative-free optimization to maximize the metric.
    """
    start_time = time.time()
    deadline = start_time + 630.0
    safety_margin = 30.0
    
    n = 20
    base_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    
    def get_points(params):
        """
        params: [r_0, ..., r_19, theta_0, ..., theta_19]
        where r_i are radii and theta_i are angular offsets from base_angles
        """
        r = np.abs(params[:n])
        theta = base_angles + params[n:]
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return np.column_stack((x, y))
    
    def objective(params):
        """Objective to minimize: negative smooth score + penalty terms"""
        if time.time() > deadline - safety_margin:
            return 1e10
        
        r = params[:n]
        r_abs = np.abs(r)
        
        # Penalty for small radii to avoid degeneracy
        r_penalty = np.sum(np.maximum(0, 0.1 - r_abs) ** 2)
        
        points = get_points(params)
        score, valid = compute_metric_smooth(points)
        
        if not valid:
            return 100.0 + r_penalty
        
        return -score + 0.001 * r_penalty
    
    # Generate diverse initial candidates
    rng = np.random.default_rng(42)
    candidates = []
    
    # 1. Regular polygon
    r_reg = np.ones(n)
    off_reg = np.zeros(n)
    candidates.append(np.concatenate([r_reg, off_reg]))
    
    # 2. Small perturbations of regular polygon
    for scale in [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]:
        for _ in range(3):
            r_pert = 1.0 + scale * rng.normal(0, 1, n)
            off_pert = scale * 0.5 * rng.normal(0, 1, n)
            candidates.append(np.concatenate([np.abs(r_pert), off_pert]))
    
    # 3. Harmonic modes with various frequencies and amplitudes
    for freq in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        for amp_r in [0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3]:
            for amp_theta in [0.0, 0.03, 0.05, 0.1, 0.15]:
                for phase in [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]:
                    r_h = 1.0 + amp_r * np.sin(freq * base_angles + phase)
                    off_h = amp_theta * np.cos(freq * base_angles + phase)
                    candidates.append(np.concatenate([np.abs(r_h), off_h]))
    
    # 4. Alternating patterns
    for amp in [0.05, 0.1, 0.15, 0.2, 0.3]:
        for period in [2, 3, 4, 5]:
            r_alt = 1.0 + amp * np.sin(2 * np.pi * base_angles * period / (2 * np.pi))
            off_alt = amp * 0.5 * np.cos(2 * np.pi * base_angles * period / (2 * np.pi))
            candidates.append(np.concatenate([np.abs(r_alt), off_alt]))
    
    # 5. Random structured
    for _ in range(50):
        r_rand = 1.0 + 0.25 * rng.normal(0, 1, n)
        off_rand = 0.15 * rng.normal(0, 1, n)
        candidates.append(np.concatenate([np.abs(r_rand), off_rand]))
    
    # 6. Slightly non-uniform angular spacing
    for _ in range(30):
        r_uniform = np.ones(n)
        angle_pert = 0.08 * rng.normal(0, 1, n)
        angle_pert -= np.mean(angle_pert)
        candidates.append(np.concatenate([r_uniform, angle_pert]))
    
    # 7. Two-scale patterns (large radius modulation + small angular perturbation)
    for _ in range(40):
        r_two = 1.0 + 0.2 * np.sin(2 * base_angles) + 0.05 * rng.normal(0, 1, n)
        off_two = 0.05 * np.cos(3 * base_angles) + 0.03 * rng.normal(0, 1, n)
        candidates.append(np.concatenate([np.abs(r_two), off_two]))
    
    # 8. Targeted patterns: emphasize even/odd vertex separation
    for even_r in [0.9, 0.95, 1.0, 1.05, 1.1]:
        for odd_r in [0.9, 0.95, 1.0, 1.05, 1.1]:
            r_target = np.ones(n)
            r_target[0::2] = even_r
            r_target[1::2] = odd_r
            for off_scale in [0.0, 0.02, 0.05]:
                off_target = off_scale * np.sin(2 * base_angles)
                candidates.append(np.concatenate([r_target.copy(), off_target.copy()]))
    
    # 9. Metric-structure-aware patterns: design distances to have clear gaps
    # For a regular 20-gon, distances from a vertex are roughly:
    # d_j = 2*sin(j*pi/20) for j=1..9, then symmetric
    # We want to maximize the gap structure. Try patterns that create
    # distinct distance clusters.
    for n_clusters in [2, 3, 4]:
        for cluster_ratio in [1.1, 1.2, 1.3, 1.5]:
            r_cluster = np.ones(n)
            for j in range(n):
                cluster_id = j % n_clusters
                r_cluster[j] = 1.0 + 0.1 * cluster_ratio * cluster_id / (n_clusters - 1)
            off_cluster = 0.02 * np.sin(n_clusters * base_angles)
            candidates.append(np.concatenate([np.abs(r_cluster), off_cluster]))
    
    # 10. Perturb regular polygon with structured angular shifts to create distance gaps
    for shift_pattern in [0.01, 0.02, 0.03, 0.05, 0.08]:
        for shift_freq in [1, 2, 3, 5]:
            r_sp = np.ones(n)
            off_sp = shift_pattern * np.sin(shift_freq * base_angles)
            candidates.append(np.concatenate([r_sp, off_sp]))
    
    # Evaluate all candidates with the exact metric
    scored_candidates = []
    for i, params in enumerate(candidates):
        if time.time() > deadline - 240:
            break
        
        points = get_points(params)
        score, valid = compute_metric(points)
        if valid:
            scored_candidates.append((score, params, points))
    
    # Sort by score
    scored_candidates.sort(key=lambda x: -x[0])
    
    best_score = scored_candidates[0][0] if scored_candidates else 0.0
    best_params = scored_candidates[0][1].copy() if scored_candidates else None
    
    # Optimize top candidates using multiple methods
    top_k = min(25, len(scored_candidates))
    
    for score, params, points in scored_candidates[:top_k]:
        if time.time() > deadline - 120:
            break
        
        # Try L-BFGS-B with bounds (primary method)
        try:
            bounds = [(0.5, 2.0)] * n + [(-0.5, 0.5)] * n
            res_lbfgsb = minimize(objective, params, method='L-BFGS-B', bounds=bounds,
                                   options={'maxiter': 300, 'gtol': 1e-9, 'disp': False})
            r_final = np.abs(res_lbfgsb.x[:n])
            off_final = res_lbfgsb.x[n:]
            params_final = np.concatenate([r_final, off_final])
            points_final = get_points(params_final)
            score_final, valid_final = compute_metric(points_final)
            
            if valid_final and score_final > best_score:
                best_score = score_final
                best_params = params_final.copy()
        except Exception:
            pass
        
        # Try BFGS with finite differences
        try:
            res_bfgs = minimize(objective, params, method='BFGS',
                                options={'maxiter': 200, 'gtol': 1e-9, 'disp': False})
            r_final = np.abs(res_bfgs.x[:n])
            off_final = res_bfgs.x[n:]
            params_final = np.concatenate([r_final, off_final])
            points_final = get_points(params_final)
            score_final, valid_final = compute_metric(points_final)
            
            if valid_final and score_final > best_score:
                best_score = score_final
                best_params = params_final.copy()
        except Exception:
            pass
        
        # Try Nelder-Mead on exact objective (for non-smooth regions)
        if time.time() > deadline - 40:
            continue
        
        def objective_exact(params_e):
            if time.time() > deadline - safety_margin:
                return 1e10
            r_e = np.abs(params_e[:n])
            r_pen_e = np.sum(np.maximum(0, 0.1 - r_e) ** 2)
            pts_e = get_points(params_e)
            sc_e, val_e = compute_metric(pts_e)
            if not val_e:
                return 100.0 + r_pen_e
            return -sc_e + 0.001 * r_pen_e
        
        try:
            res_nm = minimize(objective_exact, params, method='Nelder-Mead',
                              options={'maxiter': 400, 'adaptive': True, 'xatol': 1e-9, 'fatol': 1e-9})
            r_final = np.abs(res_nm.x[:n])
            off_final = res_nm.x[n:]
            params_final = np.concatenate([r_final, off_final])
            points_final = get_points(params_final)
            score_final, valid_final = compute_metric(points_final)
            
            if valid_final and score_final > best_score:
                best_score = score_final
                best_params = params_final.copy()
        except Exception:
            pass
    
    # Local coordinate descent on vertex positions for fine-tuning (reduced cost)
    if time.time() < deadline - 30 and best_params is not None:
        current_points = get_points(best_params).copy()
        current_score = best_score
        
        for iteration in range(30):
            if time.time() > deadline - 15:
                break
            
            improved = False
            for i in range(n):
                if time.time() > deadline - 15:
                    break
                
                best_pos = current_points[i].copy()
                best_local_score = current_score
                
                # Adaptive multi-scale perturbations in 2D
                # Start with larger steps, then refine
                for scale in [0.02, 0.01, 0.005, 0.001, 0.0005]:
                    if time.time() > deadline - 15:
                        break
                    # Use 7x7 grid for finer search
                    for dx in np.linspace(-scale, scale, 7):
                        for dy in np.linspace(-scale, scale, 7):
                            if dx == 0 and dy == 0:
                                continue
                            new_pos = current_points[i] + np.array([dx, dy])
                            trial_points = current_points.copy()
                            trial_points[i] = new_pos
                            trial_score, trial_valid = compute_metric(trial_points)
                            if trial_valid and trial_score > best_local_score + 1e-10:
                                best_local_score = trial_score
                                best_pos = new_pos
                                improved = True
                
                if improved:
                    current_points[i] = best_pos
                    current_score = best_local_score
            
            if not improved:
                break
        
        if current_score > best_score:
            best_score = current_score
            # Convert back to parameters
            centroid = np.mean(current_points, axis=0)
            centered = current_points - centroid
            r = np.linalg.norm(centered, axis=1)
            theta = np.arctan2(centered[:, 1], centered[:, 0])
            off = theta - base_angles
            off = np.unwrap(off)
            off = (off + np.pi) % (2 * np.pi) - np.pi
            best_params = np.concatenate([r, off])
    
    # Final high-precision polish with adaptive temperature
    if time.time() < deadline - 5 and best_params is not None:
        # Multiple polish passes with tightening tolerance and adaptive smooth metric
        for polish_pass in range(8):
            if time.time() > deadline - 3:
                break
            try:
                tol = 1e-10 / (polish_pass + 1)
                temp_val = 0.02 / (polish_pass + 1)
                
                def objective_adaptive(params_a):
                    if time.time() > deadline - safety_margin:
                        return 1e10
                    r_a = np.abs(params_a[:n])
                    r_pen_a = np.sum(np.maximum(0, 0.1 - r_a) ** 2)
                    pts_a = get_points(params_a)
                    sc_a, val_a = compute_metric_smooth(pts_a, temp=temp_val)
                    if not val_a:
                        return 100.0 + r_pen_a
                    return -sc_a + 0.001 * r_pen_a
                
                res_final = minimize(objective_adaptive, best_params, method='L-BFGS-B',
                                     bounds=[(0.5, 2.0)] * n + [(-0.5, 0.5)] * n,
                                     options={'maxiter': 500, 'gtol': tol, 'disp': False})
                r_p = np.abs(res_final.x[:n])
                off_p = res_final.x[n:]
                params_p = np.concatenate([r_p, off_p])
                points_p = get_points(params_p)
                score_p, valid_p = compute_metric(points_p)
                
                if valid_p and score_p > best_score:
                    best_score = score_p
                    best_params = params_p.copy()
            except Exception:
                pass
        
        # Final exact local optimization: try small random perturbations and keep improvements
        if time.time() < deadline - 2:
            current_points = get_points(best_params).copy()
            current_score = best_score
            rng_final = np.random.default_rng(123)
            
            for _ in range(200):
                if time.time() > deadline - 1:
                    break
                
                # Small random perturbation
                perturbation = 0.001 * rng_final.normal(0, 1, (n, 2))
                trial_points = current_points + perturbation
                trial_score, trial_valid = compute_metric(trial_points)
                
                if trial_valid and trial_score > current_score + 1e-10:
                    current_points = trial_points
                    current_score = trial_score
            
            if current_score > best_score:
                best_score = current_score
                # Convert back to parameters
                centroid = np.mean(current_points, axis=0)
                centered = current_points - centroid
                r = np.linalg.norm(centered, axis=1)
                theta = np.arctan2(centered[:, 1], centered[:, 0])
                off = theta - base_angles
                off = np.unwrap(off)
                off = (off + np.pi) % (2 * np.pi) - np.pi
                best_params = np.concatenate([r, off])
    
    # Return best points
    if best_params is None:
        return np.column_stack((np.cos(base_angles), np.sin(base_angles)))
    
    return get_points(best_params)
