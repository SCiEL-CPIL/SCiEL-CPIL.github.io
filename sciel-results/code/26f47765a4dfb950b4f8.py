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
    
    # Check for repeated points using vectorized distance matrix
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

def compute_metric_with_scores(points):
    """
    Computes the metric and returns per-vertex scores.
    Returns: (metric, is_valid, vertex_scores)
    """
    N = points.shape[0]
    if N != 20:
        return 0.0, False, np.zeros(N)
    
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    dists_to_centroid = np.linalg.norm(centered, axis=1)
    mean_dist = np.mean(dists_to_centroid)
    if mean_dist <= 0.01:
        return 0.0, False, np.zeros(N)
    
    normalized = centered / mean_dist
    
    # Check for repeated points
    diff = normalized[:, np.newaxis, :] - normalized[np.newaxis, :, :]
    dist_matrix = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist_matrix, 0.0)
    min_pair_dist = np.min(dist_matrix[~np.eye(N, dtype=bool)])
    if min_pair_dist < 1e-6:
        return 0.0, False, np.zeros(N)

    # Check strictly convex
    edges = np.roll(normalized, -1, axis=0) - normalized
    cross = edges[:, 0] * np.roll(edges, -1, axis=0)[:, 1] - edges[:, 1] * np.roll(edges, -1, axis=0)[:, 0]
    if np.any(np.abs(cross) < 1e-12):
        return 0.0, False, np.zeros(N)
    if not (np.all(cross > 0) or np.all(cross < 0)):
        return 0.0, False, np.zeros(N)
    
    np.fill_diagonal(dist_matrix, np.inf)
    
    vertex_scores = np.zeros(N)
    for i in range(N):
        d = dist_matrix[i]
        d_sorted = np.sort(d)
        
        if d_sorted[0] < 0.05:
            vertex_scores[i] = 0.0
        else:
            diffs = np.diff(d_sorted)
            min_diff_idx = np.argmin(diffs)
            k = min_diff_idx
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
    
    min_score = np.min(vertex_scores)
    return min_score, True, vertex_scores

def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 600.0
    safety_margin = 20.0
    
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
        off = np.unwrap(off)
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
    
    def local_search_focused(points, max_iter=150, tol=1e-8):
        """
        Focused local search that prioritizes vertices with lowest scores.
        """
        current_points = points.copy()
        current_score, valid, v_scores = compute_metric_with_scores(current_points)
        if not valid:
            return current_points, current_score
        
        # Sort vertices by score (lowest first) to focus on weak vertices
        vertex_order = np.argsort(v_scores)
        
        for iteration in range(max_iter):
            if time.time() > deadline - safety_margin:
                break
                
            improved = False
            for i in vertex_order:
                if time.time() > deadline - safety_margin:
                    break
                
                best_pos = current_points[i].copy()
                best_score = current_score
                
                # Adaptive step size based on current score
                step_sizes = [0.02, 0.01, 0.005, 0.002, 0.001]
                found_improvement = False
                
                for step in step_sizes:
                    if found_improvement:
                        break
                    for dx in np.linspace(-step, step, 5):
                        for dy in np.linspace(-step, step, 5):
                            if abs(dx) < 1e-10 and abs(dy) < 1e-10:
                                continue
                            new_pos = current_points[i] + np.array([dx, dy])
                            trial_points = current_points.copy()
                            trial_points[i] = new_pos
                            trial_score, trial_valid = compute_metric(trial_points)
                            if trial_valid and trial_score > best_score + tol:
                                best_score = trial_score
                                best_pos = new_pos
                                improved = True
                                found_improvement = True
                
                if found_improvement:
                    current_points[i] = best_pos
                    current_score = best_score
            
            if not improved:
                break
        
        return current_points, current_score
    
    def local_search_fine(points, max_iter=100, tol=1e-10):
        """
        Fine-grained local search for final polish.
        """
        current_points = points.copy()
        current_score, valid = compute_metric(current_points)
        if not valid:
            return current_points, current_score
        
        for iteration in range(max_iter):
            if time.time() > deadline - safety_margin:
                break
                
            improved = False
            for i in range(n):
                if time.time() > deadline - safety_margin:
                    break
                
                best_pos = current_points[i].copy()
                best_score = current_score
                
                # Very fine grid
                for dx in np.linspace(-0.001, 0.001, 7):
                    for dy in np.linspace(-0.001, 0.001, 7):
                        if abs(dx) < 1e-10 and abs(dy) < 1e-10:
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
            
            if not improved:
                break
        
        return current_points, current_score
    
    best_score = -1.0
    best_points = None
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initial candidates
    candidates = []
    
    # 1. Regular polygon
    candidates.append(np.concatenate([np.ones(n), np.zeros(n)]))
    
    # 2. Perturbed regular polygons
    for noise_scale in [0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2]:
        r_noise = rng.normal(0, noise_scale, n)
        off_noise = rng.normal(0, noise_scale * 0.3, n)
        candidates.append(np.concatenate([1.0 + r_noise, off_noise]))
    
    # 3. Harmonic perturbations - these target specific distance structures
    for freq in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
        for amp in [0.01, 0.02, 0.05, 0.1, 0.15]:
            phase = rng.uniform(0, 2 * np.pi)
            r_harm = 1.0 + amp * np.sin(freq * base_angles + phase)
            off_harm = amp * 0.3 * np.cos(freq * base_angles + phase)
            candidates.append(np.concatenate([r_harm, off_harm]))
    
    # 4. Alternating patterns
    for amp in [0.02, 0.05, 0.1, 0.15]:
        r_alt = np.array([1.0 + amp if i % 2 == 0 else 1.0 - amp for i in range(n)])
        candidates.append(np.concatenate([r_alt, np.zeros(n)]))
    
    # 5. Clustered patterns - group vertices into clusters
    for cluster_size in [4, 5]:
        for cluster_amp in [0.05, 0.1]:
            r_cluster = np.ones(n)
            for c in range(n // cluster_size):
                start = c * cluster_size
                end = start + cluster_size
                r_cluster[start:end] = 1.0 + cluster_amp * (0.5 if c % 2 == 0 else -0.5)
            candidates.append(np.concatenate([r_cluster, np.zeros(n)]))
    
    # 6. Random structured
    for _ in range(30):
        r_rand = 1.0 + 0.15 * rng.normal(0, 1, n)
        off_rand = 0.05 * rng.normal(0, 1, n)
        candidates.append(np.concatenate([r_rand, off_rand]))
    
    # Evaluate all candidates
    scored_candidates = []
    for params in candidates:
        if time.time() > deadline - 120:
            break
        r_abs = np.abs(params[:n])
        off = params[n:]
        points = get_points(r_abs, off)
        score, valid = compute_metric(points)
        if valid:
            scored_candidates.append((score, params, points))
    
    scored_candidates.sort(key=lambda x: -x[0])
    
    # Optimize top candidates
    top_k = min(15, len(scored_candidates))
    for score, params, points in scored_candidates[:top_k]:
        if time.time() > deadline - 60:
            break
        
        # Phase 1: Nelder-Mead on (r, theta) parameterization
        try:
            res = minimize(objective, params, method='Nelder-Mead',
                           options={'maxiter': 600, 'adaptive': True, 'xatol': 1e-7, 'fatol': 1e-7})
            
            r_final = np.abs(res.x[:n])
            off_final = res.x[n:]
            points_final = get_points(r_final, off_final)
            score_final, valid_final = compute_metric(points_final)
            
            if valid_final and score_final > best_score:
                best_score = score_final
                best_points = points_final.copy()
                
        except Exception:
            pass
        
        # Phase 2: Focused local search on vertex coordinates
        if time.time() > deadline - 30:
            break
        
        start_pts = points_final if valid_final else points
        ls_points, ls_score = local_search_focused(start_pts, max_iter=100)
        
        if ls_score > best_score:
            best_score = ls_score
            best_points = ls_points.copy()
    
    # Additional random restarts if time permits
    if best_points is None or best_score < 1.45:
        for restart in range(30):
            if time.time() > deadline - 30:
                break
            
            r_init = 1.0 + 0.1 * rng.normal(0, 1, n)
            off_init = 0.03 * rng.normal(0, 1, n)
            params_init = np.concatenate([r_init, off_init])
            
            r_abs = np.abs(params_init[:n])
            points_init = get_points(r_abs, params_init[n:])
            
            ls_points, ls_score = local_search_focused(points_init, max_iter=50)
            
            if ls_score > best_score:
                best_score = ls_score
                best_points = ls_points.copy()
    
    if best_points is None:
        return np.column_stack((np.cos(base_angles), np.sin(base_angles)))
    
    # Final fine polish
    if time.time() < deadline - 15:
        fine_points, fine_score = local_search_fine(best_points, max_iter=50)
        if fine_score > best_score:
            best_score = fine_score
            best_points = fine_points.copy()
    
    # One more NM polish on the best solution
    if time.time() < deadline - 10:
        try:
            params_polish = points_to_params(best_points)
            res_polish = minimize(objective, params_polish, method='Nelder-Mead',
                                  options={'maxiter': 200, 'adaptive': True, 'xatol': 1e-9, 'fatol': 1e-9})
            r_p = np.abs(res_polish.x[:n])
            off_p = res_polish.x[n:]
            pts_polish = get_points(r_p, off_p)
            score_polish, valid_polish = compute_metric(pts_polish)
            if valid_polish and score_polish > best_score:
                best_score = score_polish
                best_points = pts_polish.copy()
        except Exception:
            pass
    
    return best_points
