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
    if np.any(cross == 0):
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
    deadline = start_time + 550.0
    
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

    best_score = -1.0
    best_points = None
    best_params = None
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Broad exploration with many quick restarts
    # Use shorter maxiter to allow more diverse starting points
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
    for _ in range(5):
        init_r = 1.0 + 0.1 * rng.random(n)
        init_off = 0.05 * rng.random(n)
        initializations.append((init_r, init_off))
    
    n_initializations = len(initializations)
    
    # Phase 1: Quick exploration - short Nelder-Mead runs
    per_restart_time = 15.0  # seconds per restart
    max_restarts = int((deadline - start_time - 60) / per_restart_time)
    
    for i in range(min(n_initializations, max_restarts)):
        if time.time() > deadline - 60:
            break
            
        init_r, init_off = initializations[i]
        init_params = np.concatenate([init_r, init_off])
        
        try:
            res = minimize(objective, init_params, method='Nelder-Mead',
                           options={'maxiter': 500, 'adaptive': True, 'xatol': 1e-4, 'fatol': 1e-4})
            
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
    
    # Phase 2: Focused polishing from best found
    if best_params is not None and time.time() < deadline - 30:
        try:
            # Polish with tighter tolerances
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
    
    # Phase 3: Try a few more restarts from near-best if time allows
    if best_params is not None and time.time() < deadline - 20:
        for _ in range(3):
            if time.time() > deadline - 10:
                break
            try:
                perturbation = 0.01 * rng.normal(0, 1, 2*n)
                perturbed_params = best_params + perturbation
                res3 = minimize(objective, perturbed_params, method='Nelder-Mead',
                               options={'maxiter': 500, 'adaptive': True, 'xatol': 1e-6, 'fatol': 1e-6})
                
                final_r3 = np.abs(res3.x[:n])
                final_off3 = res3.x[n:]
                final_points3 = get_points(final_r3, final_off3)
                final_score3, valid3 = compute_metric(final_points3)
                
                if valid3 and final_score3 > best_score:
                    best_score = final_score3
                    best_points = final_points3.copy()
                    best_params = res3.x.copy()
            except Exception:
                pass
    
    if best_points is None:
        return np.column_stack((np.cos(angles), np.sin(angles)))
        
    return best_points
