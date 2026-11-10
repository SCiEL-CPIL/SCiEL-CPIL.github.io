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
    
    # Vectorized pairwise distances
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
    
    # Check convexity
    edges = np.roll(normalized, -1, axis=0) - normalized
    cross = edges[:, 0] * np.roll(edges, -1, axis=0)[:, 1] - edges[:, 1] * np.roll(edges, -1, axis=0)[:, 0]
    
    cross_abs_min = np.min(np.abs(cross))
    if cross_abs_min < 1e-10:
        return -1.0
    if not (np.all(cross > 0) or np.all(cross < 0)):
        return -1.0
    
    diff = normalized[:, np.newaxis, :] - normalized[np.newaxis, :, :]
    dist_matrix = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist_matrix, np.inf)
    
    # Compute all vertex scores
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
    
    # Soft-min with floor
    eps = 1e-6
    scores_clipped = np.maximum(vertex_scores, eps)
    
    log_sum_exp = temperature * np.log(np.sum(np.exp(-scores_clipped / temperature)))
    soft_min = -log_sum_exp
    
    zero_penalty = 0.1 * np.sum(vertex_scores == 0)
    
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
        off = np.unwrap(off)
        off = (off + np.pi) % (2 * np.pi) - np.pi
        return np.concatenate([r, off])
    
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

    def objective_exact(params):
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

    def coordinate_ascent(points, max_iter=100, tol=1e-8):
        """
        Fast coordinate ascent: optimize each vertex position independently
        using 1D golden-section search along x and y axes.
        """
        current_points = points.copy()
        current_score, valid = compute_metric(current_points)
        if not valid:
            return current_points, current_score
        
        # Golden section ratio
        gr = (np.sqrt(5) - 1) / 2
        
        def search_1d(f, a, b, max_iter=50):
            """Golden section search for maximum of f on [a, b]"""
            c = b - gr * (b - a)
            d = a + gr * (b - a)
            fc = f(c)
            fd = f(d)
            for _ in range(max_iter):
                if fc > fd:
                    b = d
                    d = c
                    fd = fc
                    c = b - gr * (b - a)
                    fc = f(c)
                else:
                    a = c
                    c = d
                    fc = fd
                    d = a + gr * (b - a)
                    fd = f(d)
                if abs(b - a) < 1e-6:
                    break
            return (a + b) / 2
        
        for iteration in range(max_iter):
            if time.time() > deadline - safety_margin:
                break
            
            improved = False
            for i in range(n):
                if time.time() > deadline - safety_margin:
                    break
                
                # Search along x-axis
                def fx(x):
                    trial = current_points.copy()
                    trial[i, 0] = x
                    s, v = compute_metric(trial)
                    return -s if v else 1.0 + abs(x - current_points[i, 0])
                
                x_opt = search_1d(fx, current_points[i, 0] - 0.1, current_points[i, 0] + 0.1, max_iter=30)
                trial = current_points.copy()
                trial[i, 0] = x_opt
                s_x, v_x = compute_metric(trial)
                
                # Search along y-axis from current best
                def fy(y):
                    trial2 = current_points.copy()
                    trial2[i, 1] = y
                    s2, v2 = compute_metric(trial2)
                    return -s2 if v2 else 1.0 + abs(y - current_points[i, 1])
                
                y_opt = search_1d(fy, current_points[i, 1] - 0.1, current_points[i, 1] + 0.1, max_iter=30)
                trial2 = current_points.copy()
                trial2[i, 1] = y_opt
                s_y, v_y = compute_metric(trial2)
                
                # Pick the better improvement
                best_trial = trial if (v_x and s_x > s_y or not v_y) else trial2
                best_score_i = max(s_x if v_x else -np.inf, s_y if v_y else -np.inf)
                
                if best_score_i > current_score + tol:
                    current_points = best_trial
                    current_score = best_score_i
                    improved = True
            
            if not improved:
                break
        
        return current_points, current_score

    best_score = -1.0
    best_points = None
    
    rng = np.random.default_rng(12345)
    
    # Generate diverse initial candidates
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
    
    # 6. Clustered patterns (groups of close vertices)
    for cluster_size in [2, 3, 4]:
        for n_clusters in range(1, 5):
            if n_clusters * cluster_size > n:
                continue
            r_cl = np.ones(n)
            off_cl = np.zeros(n)
            for c in range(n_clusters):
                start = c * cluster_size
                end = start + cluster_size
                if end <= n:
                    r_cl[start:end] = 1.0 + 0.05 * rng.uniform(-1, 1)
                    off_cl[start:end] = 0.02 * rng.uniform(-1, 1)
            candidates.append(np.concatenate([r_cl, off_cl]))

    # Evaluate all candidates and keep the best ones
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
    
    # Sort by score descending
    scored_candidates.sort(key=lambda x: -x[0])
    
    # Take top candidates for optimization
    top_k = min(30, len(scored_candidates))
    for score, params, points in scored_candidates[:top_k]:
        if time.time() > deadline - 30:
            break
        
        # Phase 1: Optimize using the smooth surrogate with L-BFGS-B
        try:
            res = minimize(objective_smooth, params, method='L-BFGS-B',
                           options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6})
            
            r_final = np.abs(res.x[:n])
            off_final = res.x[n:]
            points_final = get_points(r_final, off_final)
            
            # Phase 2: Polish with exact metric using coordinate ascent
            if time.time() > deadline - 20:
                break
                
            ca_points, ca_score = coordinate_ascent(points_final, max_iter=100)
            
            if ca_score > best_score:
                best_score = ca_score
                best_points = ca_points.copy()
                
        except Exception:
            pass
    
    # If we still have time, do more random restarts with the smooth surrogate
    if best_points is None or best_score < 1.46:
        for restart in range(100):
            if time.time() > deadline - 30:
                break
            
            # Random initialization
            r_init = 1.0 + 0.2 * rng.normal(0, 1, n)
            off_init = 0.1 * rng.normal(0, 1, n)
            params_init = np.concatenate([r_init, off_init])
            
            try:
                res = minimize(objective_smooth, params_init, method='L-BFGS-B',
                               options={'maxiter': 100, 'ftol': 1e-8})
                
                r_final = np.abs(res.x[:n])
                off_final = res.x[n:]
                points_final = get_points(r_final, off_final)
                
                if time.time() > deadline - 20:
                    break
                    
                ca_points, ca_score = coordinate_ascent(points_final, max_iter=50)
                
                if ca_score > best_score:
                    best_score = ca_score
                    best_points = ca_points.copy()
                    
            except Exception:
                pass
    
    # Final polish: one more round of coordinate ascent on the best solution
    if best_points is not None and time.time() < deadline - safety_margin:
        polished, polished_score = coordinate_ascent(best_points, max_iter=200)
        if polished_score > best_score:
            best_score = polished_score
            best_points = polished.copy()
    
    if best_points is None:
        # Fallback to regular polygon
        return np.column_stack((np.cos(base_angles), np.sin(base_angles)))
        
    return best_points
