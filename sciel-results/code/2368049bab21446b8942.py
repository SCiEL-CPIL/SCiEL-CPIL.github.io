import numpy as np
from scipy.optimize import minimize

def compute_score(polygon):
    # polygon is (20, 2)
    # 1. Normalize
    centroid = np.mean(polygon, axis=0)
    p = polygon - centroid
    dists_to_centroid = np.linalg.norm(p, axis=1)
    mean_dist = np.mean(dists_to_centroid)
    if mean_dist <= 0.01:
        return 0.0
    p = p / mean_dist
    
    # 2. Check Convexity
    # Calculate cross products of consecutive edges
    # v0, v1, v2 ...
    # Edge e_i = v_{i+1} - v_i
    # Cross z = e_i x e_{i+1}
    # All must be same sign and non-zero
    n = len(p)
    cross_vals = []
    for i in range(n):
        v0 = p[i]
        v1 = p[(i+1)%n]
        v2 = p[(i+2)%n]
        e1 = v1 - v0
        e2 = v2 - v1
        cross = e1[0]*e2[1] - e1[1]*e2[0]
        cross_vals.append(cross)
    cross_vals = np.array(cross_vals)
    if np.any(cross_vals == 0):
        return 0.0
    if not (np.all(cross_vals > 0) or np.all(cross_vals < 0)):
        return 0.0
        
    # 3. Calculate Vertex Scores
    min_score = 1.0
    for i in range(n):
        v = p[i]
        # Distances to all others
        d = np.linalg.norm(p - v, axis=1)
        d[i] = np.inf # exclude self
        d_sorted = np.sort(d)
        
        # Check min distance
        if d_sorted[0] < 0.05:
            min_score = 0.0
            continue
            
        # Find k
        gaps = np.diff(d_sorted)
        min_gap_idx = np.argmin(gaps)
        k = min_gap_idx
        
        D = (d_sorted[k] + d_sorted[k+1]) / 2.0
        if D < 0.1:
            min_score = 0.0
            continue
            
        # Select 4 distances
        if k == 0:
            ds = d_sorted[0:4]
        elif k == 17: # n-3 ? n=20, indices 0..18. k=17 means d_17, d_18.
            # "d_15..d_18 if k = 17"
            ds = d_sorted[15:19]
        else:
            # "d_{k-1}..d_{k+2}"
            ds = d_sorted[k-1:k+3]
            
        # Calculate raw
        ratios = np.maximum(ds / D, D / ds)
        raw = np.mean(ratios)
        score = min(1.0, 1.0 / raw)
        if score < min_score:
            min_score = score
            
    return min_score

def equidistant_points_polygon20() -> np.ndarray:
    n = 20
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    
    # Initial guess: regular polygon
    r_init = np.ones(n)
    
    # We assume 5-fold symmetry for optimization to reduce dimensions
    # r_k = r_{k+5}
    # So we optimize r_0, r_1, r_2, r_3, r_4
    def objective(r_vec):
        r = np.tile(r_vec, 4) # length 20
        x = r * np.cos(angles)
        y = r * np.sin(angles)
        poly = np.column_stack((x, y))
        return -compute_score(poly) # Minimize negative score
        
    # Optimize
    # Use Nelder-Mead
    result = minimize(objective, r_init[:5], method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6})
    
    best_r = result.x
    # Ensure radii are positive
    best_r = np.maximum(best_r, 0.5)
    
    r_full = np.tile(best_r, 4)
    x = r_full * np.cos(angles)
    y = r_full * np.sin(angles)
    polygon = np.column_stack((x, y))
    
    return polygon
