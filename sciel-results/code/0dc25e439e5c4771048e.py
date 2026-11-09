import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

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
    deadline = start_time + 580.0
    
    n = 20
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    
    def get_points(r, off):
        r = np.abs(r)
        theta = angles + off
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
            return 1.0 + np.sum(r_abs**2)
            
        points = get_points(r_abs, off)
        score, valid = compute_metric(points)
        if not valid:
            return 1.0 + 0.1 * np.sum(r_abs**2)
        return -score
    
    def coordinate_ascent(points, max_iter=500, tol=1e-8):
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
                
                # Try to improve vertex i by adjusting its position
                best_pos = current_points[i].copy()
                best_score_i = current_score
                
                # Radial adjustment: move vertex i along its radial direction
                centroid = np.mean(current_points, axis=0)
                radial_dir = current_points[i] - centroid
                radial_norm = np.linalg.norm(radial_dir)
                if radial_norm > 1e-10:
                    radial_dir = radial_dir / radial_norm
                else:
                    radial_dir = np.array([1.0, 0.0])
                
                # Try several radial offsets
                for dr in [-0.05, -0.02, -0.01, 0.0, 0.01, 0.02, 0.05, 0.1]:
                    new_pos = current_points[i] + dr * radial_dir
                    trial_points = current_points.copy()
                    trial_points[i] = new_pos
                    trial_score, trial_valid = compute_metric(trial_points)
                    if trial_valid and trial_score > best_score_i + tol:
                        best_score_i = trial_score
                        best_pos = new_pos
                        improved = True
                
                # Angular adjustment: rotate vertex i around centroid
                if not improved:
                    for da in [-0.05, -0.02, -0.01, 0.0, 0.01, 0.02, 0.05, 0.1]:
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
                
                # Free 2D adjustment: try small perturbations in x and y
                if not improved:
                    for dx in [-0.03, -0.01, 0.01, 0.03]:
                        for dy in [-0.03, -0.01, 0.01, 0.03]:
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
    best_params = None
    
    rng = np.random.default_rng(42)
    
    # Start with regular polygon
    init_points = np.column_stack((np.cos(angles), np.sin(angles)))
    init_score, valid = compute_metric(init_points)
    if valid:
        best_score = init_score
        best_points = init_points.copy()
    
    # Phase 1: Differential evolution for global exploration
    # Parameterize by radii (r) and angular offsets (off)
    # r in [0.5, 1.5], off in [-0.2, 0.2]
    bounds = [(0.5, 1.5)] * n + [(-0.2, 0.2)] * n
    
    if time.time() < deadline - 150:
        try:
            de_res = differential_evolution(
                objective, bounds,
                maxiter=100, popsize=10,
                tol=1e-4, atol=1e-4,
                seed=42,
                polish=False,
                init='sobol'
            )
            final_r = np.abs(de_res.x[:n])
            final_off = de_res.x[n:]
            final_points = get_points(final_r, final_off)
            final_score, final_valid = compute_metric(final_points)
            
            if final_valid and final_score > best_score:
                best_score = final_score
                best_points = final_points.copy()
                best_params = de_res.x.copy()
        except Exception:
            pass
    
    # Phase 2: Coordinate ascent refinement
    if best_points is not None and time.time() < deadline - 80:
        try:
            ca_points, ca_score = coordinate_ascent(best_points, max_iter=200)
            if ca_score > best_score:
                best_score = ca_score
                best_points = ca_points.copy()
        except Exception:
            pass
    
    # Phase 3: Nelder-Mead polishing from best found
    if best_params is not None and time.time() < deadline - 30:
        try:
            res2 = minimize(objective, best_params, method='Nelder-Mead',
                           options={'maxiter': 2000, 'adaptive': True, 'xatol': 1e-7, 'fatol': 1e-7})
            
            final_r2 = np.abs(res2.x[:n])
            final_off2 = res2.x[n:]
            final_points2 = get_points(final_r2, final_off2)
            final_score2, valid2 = compute_metric(final_points2)
            
            if valid2 and final_score2 > best_score:
                best_score = final_score2
                best_points = final_points2.copy()
                best_params = res2.x.copy()
        except Exception:
            pass
    
    # Phase 4: Additional coordinate ascent if time allows
    if best_points is not None and time.time() < deadline - 20:
        try:
            ca_points2, ca_score2 = coordinate_ascent(best_points, max_iter=100)
            if ca_score2 > best_score:
                best_score = ca_score2
                best_points = ca_points2.copy()
        except Exception:
            pass
    
    if best_points is None:
        return init_points
        
    return best_points
