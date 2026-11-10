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

def compute_metric_smooth(points):
    """
    Smooth version of the metric for gradient-based optimization.
    Vectorized inner loop for k to improve speed.
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
    
    # Parameters for smoothing
    temp_gap = 0.03  # Lower temp for sharper k selection
    temp_min = 0.005 # Lower temp for sharper min
    temp_d0 = 0.01   # Temperature for smooth step
    
    scores = np.zeros(N)
    
    for i in range(N):
        d = dist_matrix[i]
        d_sorted = np.sort(d)
        
        # Smooth factor for d_sorted[0] < 0.05
        factor_d0 = 1.0 / (1.0 + np.exp(-(d_sorted[0] - 0.05) / temp_d0))
        
        if factor_d0 < 1e-6:
            scores[i] = 0.0
            continue
            
        # Calculate gaps
        gaps = np.diff(d_sorted)
        
        # Softmax weights for k
        # Subtract max for numerical stability
        exp_gaps = np.exp(-(gaps - np.min(gaps)) / temp_gap)
        weights = exp_gaps / np.sum(exp_gaps)
        
        # Vectorized calculation for all k
        # D[k] = (d_sorted[k] + d_sorted[k+1]) / 2.0
        D_vals = (d_sorted[:-1] + d_sorted[1:]) / 2.0
        
        # Smooth factor for D < 0.1
        factor_D = 1.0 / (1.0 + np.exp(-(D_vals - 0.1) / temp_d0))
        
        k_scores = np.zeros(18)
        
        for k in range(18):
            if factor_D[k] < 1e-6:
                k_scores[k] = 0.0
                continue
                
            if k == 0:
                selected = d_sorted[0:4]
            elif k == 17:
                selected = d_sorted[15:19]
            else:
                selected = d_sorted[k-1:k+3]
                
            D_safe = np.maximum(D_vals[k], 1e-8)
            
            ratios = np.maximum(selected / D_safe, D_safe / selected)
            raw = np.mean(ratios)
            k_scores[k] = min(1.0, 1.0 / raw) * factor_D[k] if raw > 0 else 0.0
        
        # Weighted sum of k_scores
        score_i = np.sum(weights * k_scores) * factor_d0
        scores[i] = score_i
    
    # Smooth minimum over vertices
    max_score = np.max(scores)
    min_score_smooth = max_score - temp_min * np.log(np.sum(np.exp(-(scores - max_score) / temp_min)))
    
    return max(0.0, min_score_smooth), True

def equidistant_points_polygon20() -> np.ndarray:
    """
    Model-and-optimize approach with vectorized smooth metric and refined search.
    """
    start_time = time.time()
    deadline = start_time + 600.0 # Leave 30s safety margin for 630s limit
    
    n = 20
    base_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    
    def get_points(params):
        r = np.abs(params[:n])
        theta = base_angles + params[n:]
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return np.column_stack((x, y))
    
    def objective_smooth(params):
        """Objective to minimize: negative smooth score + penalty terms"""
        if time.time() > deadline - 15.0:
            return 1e10
        
        r = params[:n]
        r_abs = np.abs(r)
        
        # Penalty for small radii
        r_penalty = np.sum(np.maximum(0, 0.1 - r_abs) ** 2)
        
        points = get_points(params)
        score, valid = compute_metric_smooth(points)
        
        if not valid:
            return 100.0 + r_penalty
        
        return -score + 0.001 * r_penalty

    def objective_exact(params):
        """Objective to minimize: negative exact score + penalty terms"""
        if time.time() > deadline - 15.0:
            return 1e10
        
        r = params[:n]
        r_abs = np.abs(r)
        r_penalty = np.sum(np.maximum(0, 0.1 - r_abs) ** 2)
        
        points = get_points(params)
        score, valid = compute_metric(points)
        
        if not valid:
            return 100.0 + r_penalty
        
        return -score + 0.001 * r_penalty

    # Generate diverse initial candidates
    rng = np.random.default_rng(42)
    candidates = []
    
    # 1. Regular polygon
    candidates.append(np.concatenate([np.ones(n), np.zeros(n)]))
    
    # 2. Small perturbations of regular polygon
    for scale in [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]:
        for _ in range(5):
            r_pert = 1.0 + scale * rng.normal(0, 1, n)
            off_pert = scale * 0.5 * rng.normal(0, 1, n)
            candidates.append(np.concatenate([np.abs(r_pert), off_pert]))
    
    # 3. Harmonic modes with various frequencies and amplitudes
    for freq in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        for amp_r in [0.05, 0.1, 0.15, 0.2, 0.3, 0.4]:
            for amp_theta in [0.0, 0.05, 0.1, 0.15, 0.2]:
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
    for _ in range(80):
        r_rand = 1.0 + 0.25 * rng.normal(0, 1, n)
        off_rand = 0.15 * rng.normal(0, 1, n)
        candidates.append(np.concatenate([np.abs(r_rand), off_rand]))
        
    # 6. Slightly non-uniform angular spacing
    for _ in range(50):
        r_uniform = np.ones(n)
        angle_pert = 0.08 * rng.normal(0, 1, n)
        angle_pert -= np.mean(angle_pert)
        candidates.append(np.concatenate([r_uniform, angle_pert]))
        
    # 7. Two-scale patterns
    for _ in range(60):
        r_two = 1.0 + 0.2 * np.sin(2 * base_angles) + 0.05 * rng.normal(0, 1, n)
        off_two = 0.05 * np.cos(3 * base_angles) + 0.03 * rng.normal(0, 1, n)
        candidates.append(np.concatenate([np.abs(r_two), off_two]))
        
    # 8. High frequency noise
    for _ in range(40):
        r_noise = 1.0 + 0.1 * rng.normal(0, 1, n)
        off_noise = 0.1 * rng.normal(0, 1, n)
        candidates.append(np.concatenate([np.abs(r_noise), off_noise]))

    # Evaluate all candidates with the exact metric
    scored_candidates = []
    for i, params in enumerate(candidates):
        if time.time() > deadline - 240:
            break
        
        points = get_points(params)
        score, valid = compute_metric(points)
        if valid:
            scored_candidates.append((score, params, points))
    
    if not scored_candidates:
        return np.column_stack((np.cos(base_angles), np.sin(base_angles)))
        
    # Sort by score
    scored_candidates.sort(key=lambda x: -x[0])
    
    best_score = scored_candidates[0][0]
    best_params = scored_candidates[0][1].copy()
    
    # Optimize top candidates
    # We will use a two-stage optimization:
    # Stage 1: Fast gradient descent (BFGS) on smooth metric
    # Stage 2: Refinement on exact metric (Nelder-Mead)
    
    top_k = min(60, len(scored_candidates))
    
    # First pass: Optimize top candidates with BFGS
    for score, params, points in scored_candidates[:top_k]:
        if time.time() > deadline - 120:
            break
        
        try:
            res_bfgs = minimize(objective_smooth, params, method='BFGS',
                                options={'maxiter': 150, 'gtol': 1e-6, 'disp': False})
            r_final = np.abs(res_bfgs.x[:n])
            off_final = res_bfgs.x[n:]
            params_final = np.concatenate([r_final, off_final])
            points_final = get_points(params_final)
            score_final, valid_final = compute_metric(points_final)
            
            if valid_final and score_final > best_score:
                best_score = score_final
                best_params = params_final.copy()
                # Insert new best at the front to prioritize further refinement
                scored_candidates.insert(0, (score_final, params_final, points_final))
        except Exception:
            pass
        
    # Second pass: Refine the current best and top few with Nelder-Mead on exact metric
    # We re-sort to ensure we are working on the best current ones
    scored_candidates.sort(key=lambda x: -x[0])
    
    refine_k = min(10, len(scored_candidates))
    for score, params, points in scored_candidates[:refine_k]:
        if time.time() > deadline - 60:
            break
            
        # Nelder-Mead on exact metric
        try:
            res_nm = minimize(objective_exact, params, method='Nelder-Mead',
                              options={'maxiter': 200, 'adaptive': True, 'xatol': 1e-9, 'fatol': 1e-9})
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
        
    # Final Polish: High-precision BFGS on smooth metric, then Nelder-Mead on exact
    if time.time() < deadline - 10 and best_params is not None:
        try:
            res_final_bfgs = minimize(objective_smooth, best_params, method='BFGS',
                                      options={'maxiter': 100, 'gtol': 1e-8, 'disp': False})
            r_f = np.abs(res_final_bfgs.x[:n])
            off_f = res_final_bfgs.x[n:]
            params_f = np.concatenate([r_f, off_f])
            points_f = get_points(params_f)
            score_f, valid_f = compute_metric(points_f)
            
            if valid_f and score_f > best_score:
                best_score = score_f
                best_params = params_f.copy()
                
            # One last NM on the exact metric with very tight tolerances
            res_final_nm = minimize(objective_exact, best_params, method='Nelder-Mead',
                                    options={'maxiter': 100, 'adaptive': True, 'xatol': 1e-10, 'fatol': 1e-10})
            r_ff = np.abs(res_final_nm.x[:n])
            off_ff = res_final_nm.x[n:]
            params_ff = np.concatenate([r_ff, off_ff])
            points_ff = get_points(params_ff)
            score_ff, valid_ff = compute_metric(points_ff)
            
            if valid_ff and score_ff > best_score:
                best_score = score_ff
                best_params = params_ff.copy()
                
        except Exception:
            pass
    
    # Return best points
    if best_params is None:
        return np.column_stack((np.cos(base_angles), np.sin(base_angles)))
    
    return get_points(best_params)
