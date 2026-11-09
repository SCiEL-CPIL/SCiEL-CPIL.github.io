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
        # Penalize very small radii to maintain structure
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
                
                # Radial adjustment
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
                
                # Angular adjustment
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
                
                # Free 2D adjustment
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
    
    # Generate diverse initializations
    initializations = []
    
    # 1. Regular polygon
    initializations.append((np.ones(n), np.zeros(n)))
    
    # 2. Various noise scales
    for noise_scale in [0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7]:
        init_r = 1.0 + rng.normal(0, noise_scale, n)
        init_off = rng.normal(0, noise_scale * 0.5, n)
        initializations.append((init_r, init_off))
        
    # 3. Wave patterns
    for freq in [2, 3, 4, 5]:
        for amp in [0.05, 0.1, 0.15, 0.2]:
            init_r = 1.0 + amp * np.sin(freq * angles)
            init_off = amp * 0.3 * np.cos(freq * angles)
            initializations.append((init_r, init_off))
    
    # 4. Alternating patterns
    for amp in [0.05, 0.1, 0.15]:
        pattern = np.array([1.0 + amp if i % 2 == 0 else 1.0 - amp for i in range(n)])
        init_off = np.zeros(n)
        initializations.append((pattern, init_off))
        
    # 5. Random structured perturbations
    for _ in range(10):
        init_r = 1.0 + 0.1 * rng.random(n)
        init_off = 0.05 * rng.random(n)
        initializations.append((init_r, init_off))
        
    # 6. High frequency waves
    for freq in [6, 8, 10]:
        for amp in [0.05, 0.1]:
            init_r = 1.0 + amp * np.sin(freq * angles)
            init_off = amp * 0.5 * np.cos(freq * angles)
            initializations.append((init_r, init_off))

    n_initializations = len(initializations)
    
    # Phase 1: Quick Nelder-Mead exploration from diverse starts
    # Limit time per run to allow many restarts
    per_restart_time = 10.0
    
    for i in range(n_initializations):
        if time.time() > deadline - 60:
            break
            
        init_r, init_off = initializations[i]
        init_params = np.concatenate([init_r, init_off])
        
        try:
            # Use a timeout-like mechanism by limiting maxiter
            # Adaptive Nelder-Mead is good for non-smooth objectives
            res = minimize(objective, init_params, method='Nelder-Mead',
                           options={'maxiter': 800, 'adaptive': True, 'xatol': 1e-5, 'fatol': 1e-5})
            
            final_r = np.abs(res.x[:n])
            final_off = res.x[n:]
            final_points = get_points(final_r, final_off)
            final_score, valid = compute_metric(final_points)
            
            if valid and final_score > best_score:
                best_score = final_score
                best_points = final_points.copy()
                best_params = res.x.copy()
                
        except Exception:
            pass
            
    # Phase 2: Differential Evolution for global search if time permits
    # DE is good for finding global basins, but slow. Run a short DE.
    if time.time() < deadline - 80:
        try:
            bounds = [(0.5, 2.0)] * n + [(-0.5, 0.5)] * n
            # Use a small population and few iterations to fit in time
            de_result = differential_evolution(
                objective, bounds, maxiter=30, tol=1e-4,
                popsize=10, seed=42, polish=False,
                init='sobol', mutation=(0.5, 1.0), recombination=0.8
            )
            
            if time.time() < deadline - 40:
                final_r = np.abs(de_result.x[:n])
                final_off = de_result.x[n:]
                final_points = get_points(final_r, final_off)
                final_score, valid = compute_metric(final_points)
                
                if valid and final_score > best_score:
                    best_score = final_score
                    best_points = final_points.copy()
                    best_params = de_result.x.copy()
                    
        except Exception:
            pass
            
    # Phase 3: Polish the best result found
    if best_points is not None and time.time() < deadline - 20:
        # Coordinate Ascent Polish
        ca_points, ca_score = coordinate_ascent(best_points, max_iter=200)
        if ca_score > best_score:
            best_score = ca_score
            best_points = ca_points.copy()
            
        # Further Nelder-Mead polish from the polished point
        if best_params is not None and time.time() < deadline - 10:
            try:
                # Re-extract params from best_points for a final tight NM run
                centroid = np.mean(best_points, axis=0)
                centered = best_points - centroid
                r_vals = np.linalg.norm(centered, axis=1)
                off_vals = np.arctan2(centered[:, 1], centered[:, 0]) - angles
                off_vals = np.unwrap(off_vals)
                init_params = np.concatenate([r_vals, off_vals])
                
                res_final = minimize(objective, init_params, method='Nelder-Mead',
                               options={'maxiter': 1000, 'adaptive': True, 'xatol': 1e-8, 'fatol': 1e-8})
                
                final_r = np.abs(res_final.x[:n])
                final_off = res_final.x[n:]
                final_points = get_points(final_r, final_off)
                final_score, valid = compute_metric(final_points)
                
                if valid and final_score > best_score:
                    best_score = final_score
                    best_points = final_points.copy()
                    
            except Exception:
                pass

    if best_points is None:
        # Fallback to regular polygon
        return np.column_stack((np.cos(angles), np.sin(angles)))
        
    return best_points
