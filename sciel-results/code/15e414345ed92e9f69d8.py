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
    min_pair_dist = np.inf
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(normalized[i] - normalized[j])
            if d < min_pair_dist:
                min_pair_dist = d
    if min_pair_dist < 1e-6:
        return 0.0, False

    # Check strictly convex
    edges = np.roll(normalized, -1, axis=0) - normalized
    cross = edges[:, 0] * np.roll(edges, -1, axis=0)[:, 1] - edges[:, 1] * np.roll(edges, -1, axis=0)[:, 0]
    if np.any(np.abs(cross) < 1e-12):
        return 0.0, False
    if not (np.all(cross > 0) or np.all(cross < 0)):
        return 0.0, False
    
    # Vectorized pairwise distances
    diff = normalized[:, np.newaxis, :] - normalized[np.newaxis, :, :]
    dist_matrix = np.linalg.norm(diff, axis=2)
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

def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 570.0
    
    n = 20
    base_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    
    def get_points(r, off):
        r = np.abs(r)
        theta = base_angles + off
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return np.column_stack((x, y))
    
    def objective(params):
        if time.time() > deadline:
            return 1.0
        
        r = params[:n]
        off = params[n:]
        
        r_abs = np.abs(r)
        if np.any(r_abs < 0.1):
            return 1.0 + 0.1 * np.sum(r_abs**2)
            
        points = get_points(r_abs, off)
        score, valid = compute_metric(points)
        if not valid:
            return 1.0 + 0.05 * np.sum(r_abs**2)
        return -score
    
    def coordinate_ascent(points, max_iter=300, tol=1e-9):
        """Coordinate ascent on individual vertex positions to maximize min vertex score."""
        current_points = points.copy()
        current_score, valid = compute_metric(current_points)
        if not valid:
            return current_points, current_score
        
        for iteration in range(max_iter):
            if time.time() > deadline - 30:
                break
            
            improved = False
            for i in range(n):
                if time.time() > deadline - 30:
                    break
                
                best_pos = current_points[i].copy()
                best_score_i = current_score
                
                centroid = np.mean(current_points, axis=0)
                radial_dir = current_points[i] - centroid
                radial_norm = np.linalg.norm(radial_dir)
                if radial_norm > 1e-10:
                    radial_dir = radial_dir / radial_norm
                else:
                    radial_dir = np.array([1.0, 0.0])
                
                # Radial adjustments - finer grid
                for dr in [-0.04, -0.02, -0.01, -0.005, 0.005, 0.01, 0.02, 0.04]:
                    new_pos = current_points[i] + dr * radial_dir
                    trial_points = current_points.copy()
                    trial_points[i] = new_pos
                    trial_score, trial_valid = compute_metric(trial_points)
                    if trial_valid and trial_score > best_score_i + tol:
                        best_score_i = trial_score
                        best_pos = new_pos
                        improved = True
                
                # Angular adjustments
                for da in [-0.04, -0.02, -0.01, -0.005, 0.005, 0.01, 0.02, 0.04]:
                    cos_a, sin_a = np.cos(da), np.sin(da)
                    rel = current_points[i] - centroid
                    new_rel = np.array([cos_a * rel[0] - sin_a * rel[1],
                                        sin_a * rel[0] + cos_a * rel[1]])
                    new_pos = centroid + new_rel
                    trial_points = current_points.copy()
                    trial_points[i] = new_pos
                    trial_score, trial_valid = compute_metric(trial_points)
                    if trial_valid and trial_score > best_score_i + tol:
                        best_score_i = trial_score
                        best_pos = new_pos
                        improved = True
                
                # Small 2D adjustments
                for dx in [-0.02, -0.01, 0.01, 0.02]:
                    for dy in [-0.02, -0.01, 0.01, 0.02]:
                        if dx == 0 and dy == 0:
                            continue
                        new_pos = current_points[i] + np.array([dx, dy])
                        trial_points = current_points.copy()
                        trial_points[i] = new_pos
                        trial_score, trial_valid = compute_metric(trial_points)
                        if trial_valid and trial_score > best_score_i + tol:
                            best_score_i = trial_score
                            best_pos = new_pos
                            improved = True
                
                if improved:
                    current_points[i] = best_pos
                    current_score = best_score_i
            
            if not improved:
                break
        
        return current_points, current_score
    
    best_score = -1.0
    best_points = None
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initializations
    initializations = []
    
    # 1. Regular polygon
    initializations.append((np.ones(n), np.zeros(n)))
    
    # 2. Various noise scales
    for noise_scale in [0.02, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]:
        init_r = 1.0 + rng.normal(0, noise_scale, n)
        init_off = rng.normal(0, noise_scale * 0.5, n)
        initializations.append((init_r, init_off))
        
    # 3. Wave patterns (radius modulation)
    for freq in [2, 3, 4, 5]:
        for amp in [0.03, 0.06, 0.1, 0.15, 0.2]:
            init_r = 1.0 + amp * np.sin(freq * base_angles)
            init_off = np.zeros(n)
            initializations.append((init_r, init_off))
    
    # 4. Angle perturbations (wave patterns on angles)
    for freq in [2, 3, 4, 5]:
        for amp in [0.03, 0.06, 0.1]:
            init_r = np.ones(n)
            init_off = amp * np.sin(freq * base_angles)
            initializations.append((init_r, init_off))
    
    # 5. Mixed patterns
    for freq in [2, 3]:
        for amp_r in [0.05, 0.1]:
            for amp_a in [0.03, 0.06]:
                init_r = 1.0 + amp_r * np.sin(freq * base_angles)
                init_off = amp_a * np.cos(freq * base_angles + 0.5)
                initializations.append((init_r, init_off))
    
    # Evaluate all initializations and pick the best few for optimization
    init_scores = []
    for init_r, init_off in initializations:
        points = get_points(init_r, init_off)
        score, valid = compute_metric(points)
        if valid:
            init_scores.append((score, init_r.copy(), init_off.copy()))
    
    # Sort by score descending
    init_scores.sort(key=lambda x: -x[0])
    
    # Take top initializations for optimization
    n_optimizations = min(8, len(init_scores))
    
    for idx in range(n_optimizations):
        if time.time() > deadline - 60:
            break
        
        init_score, init_r, init_off = init_scores[idx]
        
        if init_score > best_score:
            best_score = init_score
            best_points = get_points(init_r, init_off)
        
        init_params = np.concatenate([init_r, init_off])
        
        try:
            # Nelder-Mead optimization
            res = minimize(objective, init_params, method='Nelder-Mead',
                           options={'maxiter': 2000, 'adaptive': True, 'xatol': 1e-4, 'fatol': 1e-4})
            
            if time.time() < deadline - 30:
                final_r = np.abs(res.x[:n])
                final_off = res.x[n:]
                final_points = get_points(final_r, final_off)
                final_score, valid = compute_metric(final_points)
                
                if valid and final_score > best_score:
                    best_score = final_score
                    best_points = final_points
                    
                    # Coordinate ascent refinement
                    if time.time() < deadline - 20:
                        refined_points, refined_score = coordinate_ascent(final_points, max_iter=200)
                        if refined_score > best_score:
                            best_score = refined_score
                            best_points = refined_points
                    
                    # Second Nelder-Mead polish
                    if time.time() < deadline - 15:
                        polish_params = np.concatenate([final_r, final_off])
                        res2 = minimize(objective, polish_params, method='Nelder-Mead',
                                       options={'maxiter': 1500, 'adaptive': True, 'xatol': 1e-6, 'fatol': 1e-6})
                        
                        if time.time() < deadline - 10:
                            polish_r = np.abs(res2.x[:n])
                            polish_off = res2.x[n:]
                            polish_points = get_points(polish_r, polish_off)
                            polish_score, valid2 = compute_metric(polish_points)
                            
                            if valid2 and polish_score > best_score:
                                best_score = polish_score
                                best_points = polish_points
                                
                                # Final coordinate ascent
                                if time.time() < deadline - 5:
                                    refined_points2, refined_score2 = coordinate_ascent(polish_points, max_iter=100)
                                    if refined_score2 > best_score:
                                        best_score = refined_score2
                                        best_points = refined_points2
                                        
        except Exception:
            pass
    
    # If we have a good point, do a final coordinate ascent polish
    if best_points is not None and time.time() < deadline - 5:
        final_refined, final_score = coordinate_ascent(best_points, max_iter=100)
        if final_score > best_score:
            best_score = final_score
            best_points = final_refined
    
    if best_points is None:
        # Fallback: regular polygon
        return np.column_stack((np.cos(base_angles), np.sin(base_angles)))
    
    final_score, valid = compute_metric(best_points)
    if not valid:
        return np.column_stack((np.cos(base_angles), np.sin(base_angles)))
        
    return best_points
