# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_score(polygon):
    """
    Compute the metric score for a given polygon.
    Returns the minimum vertex score.
    """
    n = len(polygon)
    # Normalize: subtract centroid, divide by mean distance to centroid
    centroid = np.mean(polygon, axis=0)
    centered = polygon - centroid
    dists_to_centroid = np.linalg.norm(centered, axis=1)
    mean_dist = np.mean(dists_to_centroid)
    
    if mean_dist <= 0.01:
        return 0.0
        
    normalized = centered / mean_dist
    
    # Check convexity and other validity constraints
    # 1. No repeated points
    for i in range(n):
        for j in range(i+1, n):
            if np.linalg.norm(normalized[i] - normalized[j]) < 1e-6:
                return 0.0
                
    # 2. Strictly convex: turns are nonzero and same sign
    signs = []
    total_turn = 0.0
    for i in range(n):
        p0 = normalized[i]
        p1 = normalized[(i+1) % n]
        p2 = normalized[(i+2) % n]
        
        v1 = p1 - p0
        v2 = p2 - p1
        
        cross = v1[0]*v2[1] - v1[1]*v2[0]
        if abs(cross) < 1e-9:
            return 0.0
        signs.append(np.sign(cross))
        
        # Angle turn
        # Using atan2 of cross and dot
        dot = np.dot(v1, v2)
        angle = np.arctan2(cross, dot)
        total_turn += angle
        
    if len(set(signs)) != 1:
        return 0.0
        
    if abs(abs(total_turn) - 2*np.pi) > 1e-6:
        return 0.0
        
    min_score = 1.0
    
    for i in range(n):
        v = normalized[i]
        # Distances to other vertices
        d = np.linalg.norm(normalized - v, axis=1)
        d[i] = np.inf # Exclude self
        d = d[np.isfinite(d)]
        d = np.sort(d)
        
        # Check d_0 < 0.05
        if d[0] < 0.05:
            return 0.0
            
        # Find k: first index minimizing d_{k+1} - d_k
        gaps = np.diff(d)
        min_gap_idx = np.argmin(gaps)
        k = min_gap_idx
        
        D = (d[k] + d[k+1]) / 2.0
        if D < 0.1:
            return 0.0
            
        # Select 4 distances
        if k == 0:
            dists_4 = d[0:4]
        elif k == n - 2: # k=17 for n=20
            dists_4 = d[n-4:n]
        else:
            dists_4 = d[k-1:k+3]
            
        # Calculate raw
        ratios = np.maximum(dists_4 / D, D / dists_4)
        raw = np.mean(ratios)
        
        score = min(1.0, 1.0 / raw)
        if score < min_score:
            min_score = score
            
    return min_score

def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex polygon with n = 20 vertices.
    """
    start_time = time.time()
    deadline = 600.0
    safety_margin = 10.0
    
    n = 20
    # Initial guess: Regular polygon
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)
    initial_polygon = np.stack([x, y], axis=1)
    
    best_polygon = initial_polygon
    best_score = _compute_score(best_polygon)
    
    # Define objective function for minimization (we want to maximize score, so minimize -score)
    # However, score is non-smooth. We can try to minimize the 'raw' of the worst vertex?
    # Or just use Nelder-Mead on the score directly.
    
    def objective(params):
        if time.time() > deadline - safety_margin:
            return 1e9 # Penalize timeout
        
        x, y = params
        polygon = np.stack([x, y], axis=1)
        
        # Simple validity check to avoid expensive score calc if clearly invalid
        # e.g. check if points are distinct
        # The _compute_score function handles validity and returns 0.0 if invalid.
        # We want to maximize score, so minimize -score.
        # But if score is 0, -score is 0. We want to avoid 0.
        # So we minimize (1 - score).
        score = _compute_score(polygon)
        return 1.0 - score
        
    # Multi-start Nelder-Mead
    rng = np.random.default_rng(42)
    
    num_starts = 5
    for i in range(num_starts):
        if time.time() > deadline - safety_margin:
            break
            
        if i == 0:
            x0 = initial_polygon.flatten()
        else:
            # Perturb the initial polygon
            noise = rng.normal(0, 0.05, size=2*n)
            x0 = initial_polygon.flatten() + noise
            
        # Ensure the perturbed polygon is still roughly convex/valid?
        # Nelder-Mead will handle it, but starting from invalid might be slow.
        # We'll just let the optimizer work.
        
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12})
            
            if res.success or res.fun < 1.0: # If it found a valid point (score > 0)
                best_params = res.x
                best_polygon = np.stack([best_params[:n], best_params[n:]], axis=1)
                current_score = _compute_score(best_polygon)
                
                if current_score > best_score:
                    best_score = current_score
                    
        except Exception as e:
            pass
            
    # Final high-precision polish on the best candidate
    if best_score > 0 and time.time() < deadline - safety_margin:
        def objective_polish(params):
            x, y = params
            polygon = np.stack([x, y], axis=1)
            score = _compute_score(polygon)
            return 1.0 - score
            
        try:
            res_polish = minimize(objective_polish, best_polygon.flatten(), method='Nelder-Mead',
                                  options={'maxiter': 10000, 'xatol': 1e-14, 'fatol': 1e-14})
            if res_polish.fun < 1.0:
                best_params = res_polish.x
                best_polygon = np.stack([best_params[:n], best_params[n:]], axis=1)
                final_score = _compute_score(best_polygon)
                if final_score > best_score:
                    best_score = final_score
        except Exception:
            pass
            
    return best_polygon

# EVOLVE-BLOCK-END
