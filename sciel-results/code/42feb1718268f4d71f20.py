import numpy as np
import time
from scipy.optimize import minimize
from scipy.optimize import differential_evolution

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
            return 1.0
            
        points = get_points(r_abs, off)
        score, valid = compute_metric(points)
        if not valid:
            return 1.0 + np.sum(r_abs**2)
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
    
    # Start with regular polygon
    init_points = np.column_stack((np.cos(angles), np.sin(angles)))
    init_score, valid = compute_metric(init_points)
    if valid:
        best_score = init_score
        best_points = init_points.copy()
    
    # Coordinate ascent from regular polygon
    if time.time() < deadline - 80:
        ca_points, ca_score = coordinate_ascent(init_points, max_iter=300)
        if ca_score > best_score:
            best_score = ca_score
            best_points = ca_points.copy()
    
    # Differential evolution for global search
    if time.time() < deadline - 60:
        try:
            bounds = [(0.5, 2.0)] * n + [(-0.5, 0.5)] * n
            de_result = differential_evolution(
                objective, bounds, maxiter=50, tol=1e-5,
                popsize=15, seed=42, polish=False,
                init='sobol', mutation=(0.3, 1.0), recombination=0.7
            )
            
            if time.time() < deadline - 40:
                final_r = np.abs(de_result.x[:n])
                final_off = de_result.x[n:]
                final_points = get_points(final_r, final_off)
                final_score, valid = compute_metric(final_points)
                
                if valid and final_score > best_score:
                    best_score = final_score
                    best_points = final_points.copy()
                    
                    # Polish with coordinate ascent
                    if time.time() < deadline - 20:
                        ca_points2, ca_score2 = coordinate_ascent(final_points, max_iter=200)
                        if ca_score2 > best_score:
                            best_score = ca_score2
                            best_points = ca_points2.copy()
                            
        except Exception:
            pass
    
    # Nelder-Mead refinement from best found
    if best_points is not None and time.time() < deadline - 30:
        centroid = np.mean(best_points, axis=0)
        centered = best_points - centroid
        r_vals = np.linalg.norm(centered, axis=1)
        off_vals = np.arctan2(centered[:, 1], centered[:, 0]) - angles
        off_vals = np.unwrap(off_vals)
        init_params = np.concatenate([r_vals, off_vals])
        
        try:
            res = minimize(objective, init_params, method='Nelder-Mead',
                           options={'maxiter': 5000, 'adaptive': True, 'xatol': 1e-5, 'fatol': 1e-5})
            
            if time.time() < deadline - 10:
                final_r = np.abs(res.x[:n])
                final_off = res.x[n:]
                final_points = get_points(final_r, final_off)
                final_score, valid = compute_metric(final_points)
                
                if valid and final_score > best_score:
                    best_score = final_score
                    best_points = final_points.copy()
                    
                    # Final polish
                    if time.time() < deadline - 5:
                        ca_points3, ca_score3 = coordinate_ascent(final_points, max_iter=150)
                        if ca_score3 > best_score:
                            best_score = ca_score3
                            best_points = ca_points3.copy()
                            
        except Exception:
            pass
    
    # Additional Nelder-Mead restarts if time permits
    rng = np.random.default_rng(123)
    for restart in range(2):
        if time.time() > deadline - 25:
            break
        
        init_r = 1.0 + rng.normal(0, 0.1, n)
        init_off = rng.normal(0, 0.1, n)
        init_params = np.concatenate([init_r, init_off])
        
        try:
            res = minimize(objective, init_params, method='Nelder-Mead',
                           options={'maxiter': 3000, 'adaptive': True, 'xatol': 1e-5, 'fatol': 1e-5})
            
            if time.time() < deadline - 10:
                final_r = np.abs(res.x[:n])
                final_off = res.x[n:]
                final_points = get_points(final_r, final_off)
                final_score, valid = compute_metric(final_points)
                
                if valid and final_score > best_score:
                    best_score = final_score
                    best_points = final_points.copy()
                    
                    if time.time() < deadline - 5:
                        ca_pts, ca_sc = coordinate_ascent(final_points, max_iter=100)
                        if ca_sc > best_score:
                            best_score = ca_sc
                            best_points = ca_pts.copy()
                            
        except Exception:
            pass
    
    if best_points is None:
        return init_points
    
    final_score, valid = compute_metric(best_points)
    if not valid:
        return init_points
        
    return best_points
