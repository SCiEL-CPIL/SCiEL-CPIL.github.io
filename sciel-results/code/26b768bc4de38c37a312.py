import numpy as np
import time
from scipy.optimize import minimize

def _vertex_score(v, pts):
    """Compute the vertex score for vertex v given all points pts."""
    diffs = pts - v
    dists = np.sqrt(np.sum(diffs**2, axis=1))
    mask = np.ones(len(pts), dtype=bool)
    mask[np.argmin(dists)] = False
    d = np.sort(dists[mask])
    
    # Check for too-close points
    if np.any(d < 0.05):
        return 0.0
        
    gaps = np.diff(d)
    k = int(np.argmin(gaps))
    D = (d[k] + d[k + 1]) / 2.0
    
    # Check for small gap center
    if D < 0.1:
        return 0.0
        
    if k == 0:
        selected = d[0:4]
    elif k == 17:
        selected = d[15:19]
    else:
        selected = d[k - 1:k + 3]
        
    ratios = np.maximum(selected / D, D / selected)
    raw = np.mean(ratios)
    return min(1.0, 1.0 / raw)

def _min_vertex_score(pts):
    scores = [_vertex_score(pts[i], pts) for i in range(len(pts))]
    return min(scores)

def _is_valid(pts):
    if len(pts) != 20:
        return False
    if not np.all(np.isfinite(pts)):
        return False
    for i in range(20):
        for j in range(i + 1, 20):
            if np.linalg.norm(pts[i] - pts[j]) < 1e-12:
                return False
    signs = []
    for i in range(20):
        p0 = pts[i]
        p1 = pts[(i + 1) % 20]
        p2 = pts[(i + 2) % 20]
        v1 = p1 - p0
        v2 = p2 - p1
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        if abs(cross) < 1e-14:
            return False
        signs.append(np.sign(cross))
    if not all(s == signs[0] for s in signs):
        return False
    return True

def _normalize(pts):
    centroid = np.mean(pts, axis=0)
    pts_norm = pts - centroid
    dists = np.linalg.norm(pts_norm, axis=1)
    mean_dist = np.mean(dists)
    if mean_dist <= 0.01:
        return pts_norm, False
    pts_norm = pts_norm / mean_dist
    return pts_norm, True

def _params_to_pts(radii):
    n = len(radii)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    return pts

def _smooth_objective(radii):
    """
    Minimize -soft_min(scores) + penalties.
    soft_min(x, beta) = -log(sum(exp(-x_i/beta)))/beta
    """
    n = len(radii)
    pts = _params_to_pts(radii)
    norm_pts, valid = _normalize(pts)
    if not valid:
        return 1e6
    
    # Convexity penalty: all cross products should be positive (for CCW order)
    # We assume CCW. If CW, we can flip, but let's enforce CCW.
    cross_prods = []
    for i in range(n):
        p0 = norm_pts[i]
        p1 = norm_pts[(i + 1) % n]
        p2 = norm_pts[(i + 2) % n]
        v1 = p1 - p0
        v2 = p2 - p1
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        cross_prods.append(cross)
    cross_prods = np.array(cross_prods)
    
    # Penalty for non-convexity (cross <= 0)
    # Use a smooth penalty: sum of min(cross, 0)^2
    neg_cross = np.minimum(cross_prods, 0)
    convexity_penalty = np.sum(neg_cross**2) * 100.0
    
    if convexity_penalty > 1e-3:
        return 1e6 + convexity_penalty
        
    # Compute scores
    scores = np.array([_vertex_score(norm_pts[i], norm_pts) for i in range(n)])
    
    # Soft min
    beta = 0.1
    # Avoid log(0) or overflow
    # scores are in [0, 1]
    # exp(-s/beta) can be large if s is small.
    # If any score is 0, soft_min is ~0.
    if np.any(scores < 1e-6):
        # If any score is effectively 0, the min is 0.
        # We want to push them away from 0.
        # Return a large penalty? Or just use the raw min for gradient?
        # Let's use a hard min if any are 0 to guide away, but that's non-smooth.
        # Better: return -min(scores) + penalty for 0s?
        # The soft min handles this naturally if we shift scores.
        # Let's just compute soft min carefully.
        pass
        
    # To avoid numerical issues, shift scores by a small epsilon if needed?
    # No, scores are bounded.
    # exp(-0/0.1) = 1. exp(-1/0.1) = exp(-10) ~ 4e-5.
    # Sum is ~ 20 * 1 = 20 if all 0. -log(20)/0.1 = -30.
    # If all 1: exp(-10) * 20 ~ 1e-3. -log(1e-3)/0.1 = 69.
    # Wait, soft min of {1, 1, ...} should be 1.
    # Formula: -1/beta * log(sum(exp(-x_i/beta)))
    # If x_i = 1: sum = 20 * exp(-10). log = log(20) - 10.
    # -10 * (log(20) - 10) = -3 * 10 + 100 = 70? No.
    # -10 * (2.99 - 10) = -10 * -7.01 = 70.1.
    # This is not 1.
    
    # The standard soft min is: m(x, beta) = -beta * log(sum(exp(-x_i/beta)))
    # If x_i = 1: sum = 20 * exp(-10). log = log(20) - 10.
    # -0.1 * (log(20) - 10) = -0.1 * 2.99 + 1 = -0.299 + 1 = 0.701.
    # As beta -> 0, m -> min(x).
    # As beta -> inf, m -> mean(x).
    # So we want small beta for min, but large enough for smoothness.
    
    beta = 0.05
    shifted = -scores / beta
    # Max shift to prevent overflow
    shifted_max = np.max(shifted)
    if shifted_max > 500:
        return 1e6
    sum_exp = np.sum(np.exp(shifted - shifted_max))
    if sum_exp == 0:
        return 1e6
    soft_min_val = -(beta * (np.log(sum_exp) - shifted_max))
    
    # We want to maximize soft_min_val, so minimize -soft_min_val
    return -soft_min_val + convexity_penalty

def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 600.0
    n = 20
    
    # Initial best: Regular polygon
    init_radii = np.ones(n)
    init_pts = _params_to_pts(init_radii)
    norm_init, v = _normalize(init_pts)
    
    best_score = 0.0
    best_pts = init_pts.copy()
    
    if v and _is_valid(norm_init):
        best_score = _min_vertex_score(norm_init)
        
    # Multi-start L-BFGS-B
    # Radii bounds: [0.5, 1.5] relative to 1.0
    # We can scale radii, but normalization handles scale.
    # However, extreme ratios might cause issues.
    # Let's try bounds [0.1, 10.0]
    
    bounds = [(0.1, 10.0)] * n
    
    # Number of restarts
    num_restarts = 50
    rng = np.random.default_rng(42)
    
    for i in range(num_restarts):
        if time.time() > deadline - 60:
            break
            
        # Initialize
        if i == 0:
            x0 = init_radii.copy()
        else:
            # Random perturbation around 1.0
            # Log-uniform?
            x0 = 1.0 + rng.normal(0, 0.2, n)
            # Ensure within bounds
            x0 = np.clip(x0, 0.1, 10.0)
            
        try:
            res = minimize(
                _smooth_objective,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if res.success or res.fun < 1e5:
                trial_radii = res.x
                trial_pts = _params_to_pts(trial_radii)
                norm_trial, v_trial = _normalize(trial_pts)
                
                if v_trial and _is_valid(norm_trial):
                    trial_score = _min_vertex_score(norm_trial)
                    if trial_score > best_score:
                        best_score = trial_score
                        best_pts = trial_pts.copy()
                        
        except Exception:
            pass
            
    # Final Polish: If we found a good candidate, try to refine it with tighter tolerance
    if best_score > 0:
        # Use the radii that produced best_pts?
        # We didn't store best_radii. Let's recover or re-optimize from best_pts.
        # Recovering radii from pts:
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        # pts = r * [cos, sin]
        # r_i = sqrt(x^2 + y^2)
        best_radii = np.linalg.norm(best_pts, axis=1)
        
        try:
            res_polish = minimize(
                _smooth_objective,
                best_radii,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-10}
            )
            
            if res_polish.success or res_polish.fun < 1e5:
                trial_radii = res_polish.x
                trial_pts = _params_to_pts(trial_radii)
                norm_trial, v_trial = _normalize(trial_pts)
                
                if v_trial and _is_valid(norm_trial):
                    trial_score = _min_vertex_score(norm_trial)
                    if trial_score > best_score:
                        best_score = trial_score
                        best_pts = trial_pts.copy()
                        
        except Exception:
            pass
            
    # Fallback
    if best_score == 0.0:
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
        
    return best_pts
