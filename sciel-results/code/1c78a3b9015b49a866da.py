import numpy as np
import time
from scipy.optimize import minimize

def _vertex_score(pts, i):
    """
    Calculate the score for vertex i in polygon pts.
    Returns 0 if invalid conditions are met.
    """
    n = len(pts)
    # Compute distances to other vertices
    diffs = pts - pts[i]
    dists = np.linalg.norm(diffs, axis=1)
    dists[i] = np.inf # Exclude self
    
    # Sort distances
    sorted_dists = np.sort(dists)
    
    # Check min distance
    if sorted_dists[0] < 0.05:
        return 0.0
        
    # Find k: first index minimizing d_{k+1} - d_k
    # d is sorted_dists[0:19]
    d = sorted_dists[:19]
    gaps = np.diff(d)
    
    # Find min gap
    min_gap = np.min(gaps)
    # Handle case where min gap is very small or 0
    # np.argmin returns the first occurrence
    k = np.argmin(gaps)
    
    D = (d[k] + d[k+1]) / 2.0
    
    if D < 0.1:
        return 0.0
        
    # Select 4 distances
    if k == 0:
        sel_dists = d[0:4]
    elif k == 17:
        # d has indices 0..18. k=17 means gap between d[17] and d[18].
        # Task says: d_15..d_18 if k=17.
        # d is 0-indexed 0..18.
        # So d[15], d[16], d[17], d[18]
        sel_dists = d[15:19]
    else:
        # d_{k-1} .. d_{k+2}
        # Indices: k-1, k, k+1, k+2
        sel_dists = d[k-1:k+3]
        
    # Calculate raw score
    # max(d/D, D/d)
    # Avoid division by zero (D > 0.1 checked)
    ratios = np.maximum(sel_dists / D, D / sel_dists)
    raw = np.mean(ratios)
    
    score = min(1.0, 1.0 / raw)
    return score

def _min_vertex_score(pts):
    """
    Calculate the minimum vertex score for the polygon.
    Checks convexity first.
    """
    n = len(pts)
    if n < 3:
        return 0.0
        
    # Check for repeated points
    for i in range(n):
        for j in range(i+1, n):
            if np.linalg.norm(pts[i] - pts[j]) < 1e-9:
                return 0.0
                
    # Check convexity: all cross products of consecutive edges must have the same sign
    # and be non-zero
    signs = []
    for i in range(n):
        p0 = pts[i]
        p1 = pts[(i+1) % n]
        p2 = pts[(i+2) % n]
        v1 = p1 - p0
        v2 = p2 - p1
        cross = v1[0]*v2[1] - v1[1]*v2[0]
        if abs(cross) < 1e-9:
            return 0.0
        signs.append(1 if cross > 0 else -1)
        
    if not all(s == signs[0] for s in signs):
        return 0.0
        
    # Calculate min score
    min_score = 1.0
    for i in range(n):
        s = _vertex_score(pts, i)
        if s < min_score:
            min_score = s
        if min_score == 0.0:
            break
            
    return min_score

def _points_from_angles(angles):
    """
    Generate points on unit circle from sorted angles.
    """
    angles = np.sort(angles)
    x = np.cos(angles)
    y = np.sin(angles)
    return np.column_stack((x, y))

def equidistant_points_polygon20() -> np.ndarray:
    n = 20
    start_time = time.time()
    deadline = start_time + 580.0 # 20s safety margin
    
    # Baseline: Regular polygon
    regular_angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    regular_pts = _points_from_angles(regular_angles)
    best_score = _min_vertex_score(regular_pts)
    best_angles = regular_angles.copy()
    
    # Optimization loop
    rng = np.random.default_rng(42)
    
    # Number of restarts
    # Each Nelder-Mead run should be quick.
    # Let's do 50 restarts if time permits.
    max_restarts = 100
    
    for i in range(max_restarts):
        if time.time() > deadline:
            break
            
        # Generate initial angles
        # Strategy: Regular + random noise
        # Or: Random cyclic angles
        if i % 2 == 0:
            # Perturbed regular
            noise = rng.normal(0, 0.1, n)
            angles = regular_angles + noise
        else:
            # Random cyclic angles
            # Generate n random angles in [0, 2pi)
            angles = rng.uniform(0, 2*np.pi, n)
            
        # Sort angles to ensure cyclic order
        angles = np.sort(angles)
        
        # Define objective for minimizer (minimize negative score)
        def objective(ang_flat):
            ang = np.sort(ang_flat)
            pts = _points_from_angles(ang)
            score = _min_vertex_score(pts)
            return -score
            
        # Nelder-Mead
        try:
            res = minimize(objective, angles, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6})
            
            if res.success:
                best_ang = np.sort(res.x)
                best_pts = _points_from_angles(best_ang)
                curr_score = _min_vertex_score(best_pts)
                
                if curr_score > best_score:
                    best_score = curr_score
                    best_angles = best_ang
                    
        except Exception:
            continue
            
    # Final check
    if time.time() > deadline:
        # Return best found so far
        pass
        
    final_pts = _points_from_angles(best_angles)
    
    # Normalize: subtract centroid, divide by mean distance to centroid
    centroid = np.mean(final_pts, axis=0)
    final_pts = final_pts - centroid
    dists_to_centroid = np.linalg.norm(final_pts, axis=1)
    mean_dist = np.mean(dists_to_centroid)
    
    if mean_dist <= 0.01:
        # Fallback to regular polygon if normalization fails
        return regular_pts
        
    final_pts = final_pts / mean_dist
    
    return final_pts
