import numpy as np
from scipy.optimize import minimize
import time

def calculate_score(pts):
    """
    Calculate the minimum vertex score for a polygon given as (n, 2) array of points.
    Assumes points are in cyclic order.
    """
    n = len(pts)
    if n < 3:
        return 0.0

    # Centroid and normalization
    centroid = np.mean(pts, axis=0)
    pts_norm = pts - centroid
    dist_to_centroid = np.linalg.norm(pts_norm, axis=1)
    mean_dist = np.mean(dist_to_centroid)
    if mean_dist <= 0.01:
        return 0.0
    pts_norm = pts_norm / mean_dist

    # Check for repeated points
    # Compute pairwise distances
    diff = pts_norm[:, np.newaxis, :] - pts_norm[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    
    # Check if any two points are too close (repeated or near-repeated)
    # The task says "no repeated points". 
    # Also "if any is < 0.05 the vertex scores 0"
    
    min_score = np.inf
    
    for i in range(n):
        d = dist[i, :].copy()
        d[i] = 0.0 # Self distance is 0
        
        # Sort distances to other vertices
        d_sorted = np.sort(d)
        
        # d_sorted[0] is 0.0 (self). d_sorted[1..n-1] are distances to others.
        # The task says: "sort the 19 distances from v to the other vertices, d_0 <= ... <= d_18"
        # So we exclude self.
        d_other = d_sorted[1:]
        
        # Check min distance > 0.05
        if d_other[0] < 0.05:
            return 0.0
        
        # Find k: first index minimizing d[k+1] - d[k]
        # d_other has length 19. Indices 0..18.
        # Gaps are d[1]-d[0], ..., d[18]-d[17]. 18 gaps. Indices 0..17.
        gaps = np.diff(d_other)
        k = np.argmin(gaps)
        
        D = (d_other[k] + d_other[k+1]) / 2.0
        if D < 0.1:
            return 0.0
        
        # Determine the 4 distances
        if k == 0:
            d_four = d_other[0:4]
        elif k == 17: # 18 gaps, indices 0..17. Max k is 17.
            # Task: "d_15..d_18 if k = 17"
            d_four = d_other[15:19]
        else:
            # d_{k-1} .. d_{k+2}
            # k is in 1..16
            # Indices: k-1, k, k+1, k+2
            d_four = d_other[k-1:k+3]
            
        # Calculate raw score
        # raw = mean of max(d/D, D/d)
        ratios = np.maximum(d_four / D, D / d_four)
        raw = np.mean(ratios)
        
        score_i = min(1.0, 1.0 / raw)
        
        if score_i < min_score:
            min_score = score_i
            
    return min_score

def is_strictly_convex(pts):
    """
    Check if the polygon is strictly convex.
    1. No repeated points.
    2. Every turn between consecutive edges is nonzero and of one sign.
    3. Total turning is +-2*pi.
    """
    n = len(pts)
    if n < 3:
        return False
    
    # Check repeated points
    for i in range(n):
        for j in range(i+1, n):
            if np.linalg.norm(pts[i] - pts[j]) < 1e-12:
                return False
                
    # Check turns
    # Edge i is pts[i] to pts[(i+1)%n]
    # Turn at vertex i is cross product of edge (i-1, i) and edge (i, i+1)
    # Actually, let's look at the cross product of consecutive edges.
    # e_i = pts[(i+1)%n] - pts[i]
    # cross(e_i, e_{i+1})
    # If all crosses are same sign and non-zero, it's strictly convex (for simple polygon).
    
    crosses = []
    for i in range(n):
        p0 = pts[i]
        p1 = pts[(i+1) % n]
        p2 = pts[(i+2) % n]
        
        v1 = p1 - p0
        v2 = p2 - p1
        
        cross = v1[0]*v2[1] - v1[1]*v2[0]
        crosses.append(cross)
        
    crosses = np.array(crosses)
    
    # Check non-zero
    if np.any(np.abs(crosses) < 1e-12):
        return False
        
    # Check same sign
    if np.any(crosses > 0) and np.any(crosses < 0):
        return False
        
    return True

def gaps_to_angles(gaps):
    n = len(gaps)
    gaps = np.abs(gaps) + 1e-10 # Ensure positive
    s = np.sum(gaps)
    gaps = gaps / s * 2 * np.pi
    angles = np.zeros(n)
    angles[1:] = np.cumsum(gaps[:-1])
    return angles

def angles_to_pts(angles):
    x = np.cos(angles)
    y = np.sin(angles)
    return np.column_stack((x, y))

def equidistant_points_polygon20() -> np.ndarray:
    n = 20
    start_time = time.time()
    deadline = start_time + 600.0 # 10 minutes
    
    # Initial guess: regular polygon
    gaps0 = np.ones(n)
    
    def objective_gaps(gaps):
        angles = gaps_to_angles(gaps)
        pts = angles_to_pts(angles)
        if not is_strictly_convex(pts):
            return 10.0 # Penalty
        score = calculate_score(pts)
        return -score
    
    best_gaps = gaps0.copy()
    best_score = -objective_gaps(best_gaps)
    
    # Run Nelder-Mead with multiple restarts or long run
    # Since n=20, Nelder-Mead can be slow.
    # Let's try a single long run first, then maybe polish.
    
    # We can also try BFGS with gradients if we implement them, but Nelder-Mead is easier.
    # Let's try to be smarter with the objective.
    # The "min vertex score" is the bottleneck.
    
    # Let's run Nelder-Mead.
    # maxiter: 5000 might be enough for 20D?
    # xatol, fatol: 1e-8
    
    res = minimize(objective_gaps, best_gaps, method='Nelder-Mead', 
                   options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-10, 'disp': False})
    
    if time.time() > deadline - 60:
        # Safety check
        pass
    
    if res.x is not None:
        score_res = -res.fun
        if score_res > best_score:
            best_score = score_res
            best_gaps = res.x
    
    # Polish with tighter tolerance?
    # Maybe run again from best_gaps with smaller initial simplex?
    
    # Let's try a second pass with BFGS if we had gradients.
    # Since we don't, let's just do another Nelder-Mead with smaller steps.
    
    # Check time
    if time.time() < deadline - 30:
        # Try to improve further
        # Maybe perturb best_gaps slightly and run again?
        # Or just run minimize again with tighter tolerances.
        
        # Let's try to use the best_gaps as start and run again with smaller maxiter but tighter tol
        res2 = minimize(objective_gaps, best_gaps, method='Nelder-Mead', 
                       options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12, 'disp': False})
        
        if res2.x is not None:
            score_res2 = -res2.fun
            if score_res2 > best_score:
                best_score = score_res2
                best_gaps = res2.x
    
    best_angles = gaps_to_angles(best_gaps)
    pts = angles_to_pts(best_angles)
    
    # Final validation
    if not is_strictly_convex(pts):
        # Fallback to regular polygon
        angles0 = np.linspace(0, 2*np.pi, n, endpoint=False)
        pts = angles_to_pts(angles0)
        
    return pts
