import numpy as np
from scipy.optimize import differential_evolution

def calculate_score(angles):
    n = len(angles)
    x = np.cos(angles)
    y = np.sin(angles)
    pts = np.stack([x, y], axis=1)
    
    # Check convexity (turning angles)
    # For CCW order, cross product should be positive
    convex_penalty = 0.0
    for i in range(n):
        p0 = pts[i]
        p1 = pts[(i+1)%n]
        p2 = pts[(i+2)%n]
        v1 = p1 - p0
        v2 = p2 - p1
        cross = v1[0]*v2[1] - v1[1]*v2[0]
        if cross <= 0:
            convex_penalty += 1000.0
    
    if convex_penalty > 0:
        return -convex_penalty

    # Normalize
    centroid = np.mean(pts, axis=0)
    pts = pts - centroid
    dists = np.linalg.norm(pts, axis=1)
    mean_dist = np.mean(dists)
    if mean_dist <= 0.01:
        return 0.0
    pts = pts / mean_dist
    
    # Calculate score for each vertex
    min_score = 1.0
    for i in range(n):
        vi = pts[i]
        dists_i = np.linalg.norm(pts - vi, axis=1)
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        dists_i = dists_i[mask]
        
        if np.min(dists_i) < 0.05:
            return 0.0
        
        sorted_dists = np.sort(dists_i)
        gaps = np.diff(sorted_dists)
        k = np.argmin(gaps)
        D = (sorted_dists[k] + sorted_dists[k+1]) / 2.0
        
        if D < 0.1:
            return 0.0
            
        if k == 0:
            idxs = [0, 1, 2, 3]
        elif k == 17:
            idxs = [15, 16, 17, 18]
        else:
            idxs = [k-1, k, k+1, k+2]
            
        selected_dists = sorted_dists[idxs]
        ratios = np.maximum(selected_dists / D, D / selected_dists)
        raw = np.mean(ratios)
        score = min(1.0, 1.0 / raw)
        
        if score < min_score:
            min_score = score
            
    return min_score

def equidistant_points_polygon20() -> np.ndarray:
    n = 20
    
    # We optimize 19 gaps, the 20th is determined
    def obj(gaps_19):
        # gaps_19: array of 19
        # Ensure positivity and that the 20th gap is positive
        if np.any(gaps_19 <= 0):
            return -1000.0
        gap_20 = 2.0 * np.pi - np.sum(gaps_19)
        if gap_20 <= 0:
            return -1000.0
        
        all_gaps = np.append(gaps_19, gap_20)
        
        # Compute angles
        angles = np.zeros(n)
        for i in range(1, n):
            angles[i] = np.sum(all_gaps[:i])
            
        return -calculate_score(angles)
        
    # Bounds for the 19 gaps
    # Min gap: 1e-5
    # Max gap for each: 2*pi - 18*1e-5 (leaving room for 18 others + the 20th)
    # Actually, simpler: each gap in [1e-5, 2*pi - 19*1e-5]
    # But we need sum < 2*pi.
    # DE will handle the box. The obj function penalizes invalid sums.
    lower = [1e-5] * 19
    upper = [2.0 * np.pi - 19 * 1e-5] * 19
    
    # Initial guess: regular polygon
    # x0 is not used by DE directly, but we can seed it or use init='latinhypercube'
    
    try:
        res = differential_evolution(obj, bounds=list(zip(lower, upper)), 
                                      maxiter=200, popsize=20, tol=1e-8, 
                                      seed=42, polish=False)
        if res.success:
            best_gaps_19 = res.x
        else:
            # Fallback to regular if DE fails
            best_gaps_19 = np.full(19, 2.0 * np.pi / n)
            # Check if regular is better than res.x if res failed? 
            # Just use regular if it fails.
    except Exception:
        best_gaps_19 = np.full(19, 2.0 * np.pi / n)
        
    gap_20 = 2.0 * np.pi - np.sum(best_gaps_19)
    if gap_20 <= 0:
        # Fallback
        best_gaps_19 = np.full(19, 2.0 * np.pi / n)
        gap_20 = 2.0 * np.pi - np.sum(best_gaps_19)
        
    all_gaps = np.append(best_gaps_19, gap_20)
    
    angles = np.zeros(n)
    for i in range(1, n):
        angles[i] = np.sum(all_gaps[:i])
        
    x = np.cos(angles)
    y = np.sin(angles)
    polygon = np.stack([x, y], axis=1)
    
    return polygon
