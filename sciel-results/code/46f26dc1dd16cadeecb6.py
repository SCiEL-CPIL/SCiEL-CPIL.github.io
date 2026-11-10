import numpy as np
import time
from scipy.optimize import minimize

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
    
    # Check for repeated points
    diff = normalized[:, np.newaxis, :] - normalized[np.newaxis, :, :]
    dist_matrix = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist_matrix, 0.0)
    min_pair_dist = np.min(dist_matrix[~np.eye(N, dtype=bool)])
    if min_pair_dist < 1e-6:
        return 0.0, False

    # Check strictly convex
    edges = np.roll(normalized, -1, axis=0) - normalized
    cross = edges[:, 0] * np.roll(edges, -1, axis=0)[:, 1] - edges[:, 1] * np.roll(edges, -1, axis=0)[:, 0]
    if np.any(np.abs(cross) < 1e-12):
        return 0.0, False
    if not (np.all(cross > 0) or np.all(cross < 0)):
        return 0.0, False
    
    # Vectorized pairwise distances (reuse dist_matrix)
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

def compute_metric_grad(points):
    """
    Computes the metric and its gradient with respect to vertex positions.
    Returns: (metric, is_valid, grad) where grad is (N, 2)
    """
    N = points.shape[0]
    if N != 20:
        return 0.0, False, np.zeros((N, 2))
    
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    dists_to_centroid = np.linalg.norm(centered, axis=1)
    mean_dist = np.mean(dists_to_centroid)
    if mean_dist <= 0.01:
        return 0.0, False, np.zeros((N, 2))
    
    normalized = centered / mean_dist
    
    # Check for repeated points
    diff = normalized[:, np.newaxis, :] - normalized[np.newaxis, :, :]
    dist_matrix = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist_matrix, 0.0)
    min_pair_dist = np.min(dist_matrix[~np.eye(N, dtype=bool)])
    if min_pair_dist < 1e-6:
        return 0.0, False, np.zeros((N, 2))

    # Check strictly convex
    edges = np.roll(normalized, -1, axis=0) - normalized
    cross = edges[:, 0] * np.roll(edges, -1, axis=0)[:, 1] - edges[:, 1] * np.roll(edges, -1, axis=0)[:, 0]
    if np.any(np.abs(cross) < 1e-12):
        return 0.0, False, np.zeros((N, 2))
    if not (np.all(cross > 0) or np.all(cross < 0)):
        return 0.0, False, np.zeros((N, 2))
    
    np.fill_diagonal(dist_matrix, np.inf)
    
    min_score = np.inf
    min_vertex = 0
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
            min_vertex = i
    
    # Compute gradient for the minimum-scoring vertex using finite differences
    grad = np.zeros((N, 2))
    if min_score > 0:
        h = 1e-6
        for j in range(N):
            for dim in range(2):
                perturbed = points.copy()
                perturbed[j, dim] += h
                score_plus, _ = compute_metric(perturbed)
                
                perturbed = points.copy()
                perturbed[j, dim] -= h
                score_minus, _ = compute_metric(perturbed)
                
                grad[j, dim] = (score_plus - score_minus) / (2 * h)
    
    return min_score, True, grad

def compute_metric_smooth(points, temperature=0.01):
    """
    Smooth surrogate for the metric using soft-min.
    Returns a smooth scalar objective (higher is better).
    """
    N = points.shape[0]
    if N != 20:
        return 0.0
        
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    dists_to_centroid = np.linalg.norm(centered, axis=1)
    mean_dist = np.mean(dists_to_centroid)
    if mean_dist <= 0.01:
        return 0.0
        
    normalized = centered / mean_dist
        
    # Check convexity softly
    edges = np.roll(normalized, -1, axis=0) - normalized
    cross = edges[:, 0] * np.roll(edges, -1, axis=0)[:, 1] - edges[:, 1] * np.roll(edges, -1, axis=0)[:, 0]
        
    # Soft penalty for non-convexity
    cross_abs_min = np.min(np.abs(cross))
    if cross_abs_min < 1e-10:
        return -1.0
    if not (np.all(cross > 0) or np.all(cross < 0)):
        return -1.0
        
    diff = normalized[:, np.newaxis, :] - normalized[np.newaxis, :, :]
    dist_matrix = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist_matrix, np.inf)
        
    # Compute all vertex scores first
    vertex_scores = np.zeros(N)
    for i in range(N):
        d = dist_matrix[i]
        d_sorted = np.sort(d)
            
        if d_sorted[0] < 0.05:
            vertex_scores[i] = 0.0
        else:
            diffs = np.diff(d_sorted)
            k = np.argmin(diffs)
            D = (d_sorted[k] + d_sorted[k+1]) / 2.0
                
            if D < 0.1:
                vertex_scores[i] = 0.0
            else:
                if k == 0:
                    selected = d_sorted[0:4]
                elif k == 17:
                    selected = d_sorted[15:19]
                else:
                    selected = d_sorted[k-1:k+3]
                    
                ratios = np.maximum(selected / D, D / selected)
                raw = np.mean(ratios)
                vertex_scores[i] = min(1.0, 1.0 / raw) if raw > 0 else 0.0
        
    # Use soft-min with a floor to keep gradient information
    eps = 1e-6
    scores_clipped = np.maximum(vertex_scores, eps)
        
    # Soft-min: -temperature * log(sum(exp(-scores/temperature)))
    log_sum_exp = temperature * np.log(np.sum(np.exp(-scores_clipped / temperature)))
    soft_min = -log_sum_exp
        
    # Penalty for zeros, scaled by how many are near-zero
    near_zero = np.sum(vertex_scores < 0.01)
    zero_penalty = 0.05 * near_zero
        
    return soft_min - zero_penalty

def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 600.0
    safety_margin = 15.0
    
    n = 20
    base_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    
    def get_points(r, off):
        r = np.abs(r)
        theta = base_angles + off
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return np.column_stack((x, y))
    
    def points_to_params(points):
        centroid = np.mean(points, axis=0)
        centered = points - centroid
        r = np.linalg.norm(centered, axis=1)
        theta = np.arctan2(centered[:, 1], centered[:, 0])
        off = theta - base_angles
        # Unwrap angles to be continuous
        off = np.unwrap(off)
        # Normalize offsets to [-pi, pi]
        off = (off + np.pi) % (2 * np.pi) - np.pi
        return np.concatenate([r, off])
    
    def objective(params):
        if time.time() > deadline - safety_margin:
            return 1.0
        
        r = params[:n]
        off = params[n:]
        
        r_abs = np.abs(r)
        if np.any(r_abs < 0.1):
            return 1.0 + np.sum(r_abs**2)
            
        points = get_points(r_abs, off)
        score, valid = compute_metric(points)
        if not valid:
            return 1.0 + 0.1 * np.sum(r_abs**2)
        return -score

    def objective_smooth(params, temperature=0.01):
        if time.time() > deadline - safety_margin:
            return 1.0
        
        r = params[:n]
        off = params[n:]
        
        r_abs = np.abs(r)
        if np.any(r_abs < 0.1):
            return 1.0 + np.sum(r_abs**2)
            
        points = get_points(r_abs, off)
        smooth_score = compute_metric_smooth(points, temperature)
        if smooth_score < -0.5:
            return 1.0 + 0.1 * np.sum(r_abs**2)
        return -smooth_score
    
    def local_search(points, max_iter=200, tol=1e-7):
        """
        Perform local search by directly optimizing vertex positions.
        Uses gradient-based optimization with fallback to grid search.
        """
        current_points = points.copy()
        current_score, valid = compute_metric(current_points)
        if not valid:
            return current_points, current_score
        
        # Phase 1: Gradient-based optimization
        for iteration in range(min(max_iter, 100)):
            if time.time() > deadline - safety_margin:
                break
            
            score, valid, grad = compute_metric_grad(current_points)
            if not valid:
                break
            
            if score > current_score + tol:
                current_score = score
            
            # Check if gradient is small enough to stop
            if np.max(np.abs(grad)) < 1e-8:
                break
            
            # Adaptive step size based on gradient magnitude
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-10:
                break
            
            # Use line search to find good step size
            best_step = 0.01
            best_trial_score = current_score
            
            for step in [0.05, 0.02, 0.01, 0.005, 0.002]:
                trial_points = current_points + step * grad
                trial_score, trial_valid = compute_metric(trial_points)
                if trial_valid and trial_score > best_trial_score:
                    best_trial_score = trial_score
                    best_step = step
            
            if best_step > 0 and best_trial_score > current_score + tol:
                current_points = current_points + best_step * grad
                current_score = best_trial_score
            else:
                break
        
        # Phase 2: Fine-tuning with grid search if time remains
        if time.time() < deadline - safety_margin:
            step_size = 0.005
            for iteration in range(min(50, max_iter - 100)):
                if time.time() > deadline - safety_margin:
                    break
                
                improved = False
                for i in range(n):
                    if time.time() > deadline - safety_margin:
                        break
                    
                    best_pos = current_points[i].copy()
                    best_score = current_score
                    
                    for dx in np.linspace(-step_size, step_size, 3):
                        for dy in np.linspace(-step_size, step_size, 3):
                            if dx == 0 and dy == 0:
                                continue
                            new_pos = current_points[i] + np.array([dx, dy])
                            trial_points = current_points.copy()
                            trial_points[i] = new_pos
                            trial_score, trial_valid = compute_metric(trial_points)
                            if trial_valid and trial_score > best_score + tol:
                                best_score = trial_score
                                best_pos = new_pos
                                improved = True
                    
                    if improved:
                        current_points[i] = best_pos
                        current_score = best_score
                        step_size *= 0.7
                    else:
                        step_size *= 1.3
                        step_size = min(step_size, 0.02)
                
                if not improved:
                    break
        
        return current_points, current_score
    
    best_score = -1.0
    best_points = None
    
    rng = np.random.default_rng(12345)
    
    # Generate a variety of initial candidates
    candidates = []
    
    # 1. Regular polygon
    r0 = np.ones(n)
    off0 = np.zeros(n)
    candidates.append(np.concatenate([r0, off0]))
    
    # 2. Perturbed regular polygons with various noise scales
    for noise_scale in [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]:
        r_noise = rng.normal(0, noise_scale, n)
        off_noise = rng.normal(0, noise_scale * 0.5, n)
        candidates.append(np.concatenate([1.0 + r_noise, off_noise]))
    
    # 3. Harmonic perturbations (mode shapes)
    for freq in range(1, 10):
        for amp in [0.01, 0.05, 0.1, 0.2]:
            phase = rng.uniform(0, 2 * np.pi)
            r_harm = 1.0 + amp * np.sin(freq * base_angles + phase)
            off_harm = amp * 0.5 * np.cos(freq * base_angles + phase)
            candidates.append(np.concatenate([r_harm, off_harm]))
    
    # 4. Random structured perturbations
    for _ in range(50):
        r_rand = 1.0 + 0.3 * rng.random(n)
        off_rand = 0.1 * rng.random(n)
        candidates.append(np.concatenate([r_rand, off_rand]))
    
    # 5. Alternating radius patterns
    for amp in [0.05, 0.1, 0.2]:
        r_alt = np.array([1.0 + amp if i % 2 == 0 else 1.0 - amp for i in range(n)])
        off_alt = np.zeros(n)
        candidates.append(np.concatenate([r_alt, off_alt]))
    
    # Evaluate all candidates and keep the best ones
    scored_candidates = []
    for params in candidates:
        if time.time() > deadline - 60:
            break
        r_abs = np.abs(params[:n])
        off = params[n:]
        points = get_points(r_abs, off)
        score, valid = compute_metric(points)
        if valid:
            scored_candidates.append((score, params, points))
    
    # Sort by score descending
    scored_candidates.sort(key=lambda x: -x[0])
    
    # Take top 20 candidates for local optimization
    top_k = min(20, len(scored_candidates))
    for score, params, points in scored_candidates[:top_k]:
        if time.time() > deadline - 30:
            break
        
        # First, try Nelder-Mead with smooth surrogate
        try:
            res = minimize(objective_smooth, params, method='Nelder-Mead',
                           options={'maxiter': 800, 'adaptive': True, 'xatol': 1e-7, 'fatol': 1e-7})
            
            r_final = np.abs(res.x[:n])
            off_final = res.x[n:]
            points_final = get_points(r_final, off_final)
            score_final, valid_final = compute_metric(points_final)
            
            if valid_final and score_final > best_score:
                best_score = score_final
                best_points = points_final.copy()
                
        except Exception:
            pass
        
        # Then, refine with exact metric using Nelder-Mead
        if time.time() > deadline - 30:
            break
        try:
            res2 = minimize(objective, params, method='Nelder-Mead',
                            options={'maxiter': 400, 'adaptive': True, 'xatol': 1e-8, 'fatol': 1e-8})
            
            r_final2 = np.abs(res2.x[:n])
            off_final2 = res2.x[n:]
            points_final2 = get_points(r_final2, off_final2)
            score_final2, valid_final2 = compute_metric(points_final2)
            
            if valid_final2 and score_final2 > best_score:
                best_score = score_final2
                best_points = points_final2.copy()
                
        except Exception:
            pass
        
        # Then, do gradient-based local search directly on vertex positions
        if time.time() > deadline - 20:
            break
        
        # Use the best point from NM as starting point for local search
        start_pts = points_final2 if valid_final2 else (points_final if valid_final else points)
        ls_points, ls_score = local_search(start_pts, max_iter=150)
        
        if ls_score > best_score:
            best_score = ls_score
            best_points = ls_points.copy()
    
    # If we still have time, do more random restarts with local search
    if best_points is None or best_score < 1.47:
        for restart in range(80):
            if time.time() > deadline - 30:
                break
            
            # Random initialization
            r_init = 1.0 + 0.15 * rng.normal(0, 1, n)
            off_init = 0.08 * rng.normal(0, 1, n)
            params_init = np.concatenate([r_init, off_init])
            
            r_abs = np.abs(params_init[:n])
            points_init = get_points(r_abs, params_init[n:])
            
            # Quick gradient-based local search
            ls_points, ls_score = local_search(points_init, max_iter=80)
            
            if ls_score > best_score:
                best_score = ls_score
                best_points = ls_points.copy()
    
    if best_points is None:
        # Fallback to regular polygon
        return np.column_stack((np.cos(base_angles), np.sin(base_angles)))
    
    # Final high-precision polish pass
    if time.time() < deadline - 10:
        # Use gradient-based optimization on vertex coordinates directly
        def vertex_objective_grad(x):
            pts = x.reshape(n, 2)
            s, v, g = compute_metric_grad(pts)
            if not v:
                return 1.0, -g
            return -s, -g
        
        try:
            x0 = best_points.flatten()
            current_x = x0
            
            # Gradient descent with adaptive step size
            for iteration in range(200):
                if time.time() > deadline - 5:
                    break
                
                obj_val, grad = vertex_objective_grad(current_x)
                score_check, valid_check = compute_metric(current_x.reshape(n, 2))
                
                if not valid_check:
                    break
                
                if score_check > best_score:
                    best_score = score_check
                    best_points = current_x.reshape(n, 2).copy()
                
                # Check convergence
                if np.max(np.abs(grad)) < 1e-10:
                    break
                
                # Adaptive step size
                grad_norm = np.linalg.norm(grad)
                if grad_norm < 1e-12:
                    break
                
                best_step = 0.01
                best_obj = obj_val
                
                for step in [0.05, 0.02, 0.01, 0.005, 0.002, 0.001]:
                    trial_x = current_x - step * grad
                    trial_obj, _ = vertex_objective_grad(trial_x)
                    if trial_obj < best_obj:
                        best_obj = trial_obj
                        best_step = step
                
                if best_step > 0:
                    current_x = current_x - best_step * grad
                else:
                    break
                    
        except Exception:
            pass
        
        # Additional polish with very tight Nelder-Mead on parameters
        if time.time() < deadline - 5:
            try:
                params_polish = points_to_params(best_points)
                res_nm = minimize(objective, params_polish, method='Nelder-Mead',
                                  options={'maxiter': 500, 'adaptive': True, 'xatol': 1e-12, 'fatol': 1e-12})
                r_p = np.abs(res_nm.x[:n])
                off_p = res_nm.x[n:]
                pts_nm = get_points(r_p, off_p)
                score_nm, valid_nm = compute_metric(pts_nm)
                if valid_nm and score_nm > best_score:
                    best_score = score_nm
                    best_points = pts_nm.copy()
            except Exception:
                pass
        
        # Final gradient-based local search polish
        if time.time() < deadline - 2:
            ls_points, ls_score = local_search(best_points, max_iter=100)
            if ls_score > best_score:
                best_score = ls_score
                best_points = ls_points.copy()
    
    return best_points
