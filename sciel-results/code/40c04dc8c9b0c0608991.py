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
    # Vectorized pairwise distance check for efficiency
    # Since N=20, even O(N^2) is fast, but let's be careful.
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
            # Penalize invalid points strongly but smoothly
            # The penalty should guide the optimizer back to valid regions
            # Using sum of squared radii as a secondary penalty keeps it "compact"
            return 1.0 + 0.1 * np.sum(r_abs**2)
        return -score

    best_score = -1.0
    best_points = None
    
    # Strategy: Multiple restarts with Nelder-Mead
    # We will try a variety of initial conditions.
    
    initializations = []
    
    # 1. Regular polygon
    initializations.append((np.ones(n), np.zeros(n)))
    
    # 2. Regular polygon with small noise
    rng = np.random.default_rng(42)
    for noise_scale in [0.05, 0.1, 0.2, 0.3, 0.5]:
        init_r = 1.0 + rng.normal(0, noise_scale, n)
        init_off = rng.normal(0, noise_scale, n)
        initializations.append((init_r, init_off))
        
    # 3. Elliptic perturbation (stretch x or y)
    # This might help if the optimal shape is not close to circular
    # But our parameterization is radial/angular, so elliptic is hard to represent directly without changing angles.
    # We can simulate it by adjusting r and off slightly to approximate an ellipse.
    # For simplicity, let's stick to random radial/angular perturbations which cover a wide space.
    
    # 4. Randomized "waves" in radius
    for freq in [2, 3, 4]:
        init_r = 1.0 + 0.1 * np.sin(freq * angles)
        init_off = 0.05 * np.cos(freq * angles)
        initializations.append((init_r, init_off))

    n_initializations = len(initializations)
    
    # Allocate time for each restart
    # We have ~550s. Let's say we spend 40s on each restart.
    # 550 / 40 = ~13 restarts. We have 1 + 5 + 3 = 9 initializations.
    # We can afford to run each one with a decent maxiter.
    
    for i, (init_r, init_off) in enumerate(initializations):
        if time.time() > deadline - 30:
            break
            
        init_params = np.concatenate([init_r, init_off])
        
        try:
            # Use Nelder-Mead with adaptive settings
            # xatol and fatol are tightened as we go
            res = minimize(objective, init_params, method='Nelder-Mead',
                           options={'maxiter': 2000, 'adaptive': True, 'xatol': 1e-5, 'fatol': 1e-5})
            
            if time.time() > deadline - 20:
                break
                
            final_r = np.abs(res.x[:n])
            final_off = res.x[n:]
            final_points = get_points(final_r, final_off)
            final_score, valid = compute_metric(final_points)
            
            if valid and final_score > best_score:
                best_score = final_score
                best_points = final_points.copy()
                
                # If we found a good solution, try to polish it with a tighter optimizer
                if time.time() < deadline - 15:
                    # Re-optimize from the best found so far with tighter tolerances
                    res2 = minimize(objective, res.x, method='Nelder-Mead',
                                    options={'maxiter': 1000, 'adaptive': True, 'xatol': 1e-7, 'fatol': 1e-7})
                    
                    final_r2 = np.abs(res2.x[:n])
                    final_off2 = res2.x[n:]
                    final_points2 = get_points(final_r2, final_off2)
                    final_score2, valid2 = compute_metric(final_points2)
                    
                    if valid2 and final_score2 > best_score:
                        best_score = final_score2
                        best_points = final_points2.copy()

        except Exception:
            pass
    
    if best_points is None:
        # Fallback to regular polygon
        return np.column_stack((np.cos(angles), np.sin(angles)))
        
    return best_points
