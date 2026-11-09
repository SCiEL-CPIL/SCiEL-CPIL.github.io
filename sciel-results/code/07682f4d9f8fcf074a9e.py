import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

def compute_vertex_scores(polygon):
    """
    Computes the vertex scores for a given polygon.
    Returns an array of scores for each vertex.
    """
    n = len(polygon)
    scores = np.zeros(n)
    
    # Precompute pairwise distances? n=20, so 20x20 is small.
    # dist[i, j] is distance between vertex i and j
    # Using broadcasting
    diff = polygon[:, np.newaxis, :] - polygon[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    
    for i in range(n):
        # Get distances from vertex i to all others
        d = dist[i].copy()
        d[i] = np.inf # Exclude self
        
        # Sort distances
        d_sorted = np.sort(d)
        
        # Check for repeated points (distance < 0.05)
        if d_sorted[0] < 0.05:
            scores[i] = 0.0
            continue
            
        # Find k: first index minimizing d_{k+1} - d_k
        # d_sorted has 19 elements (indices 0..18)
        # Gaps are d_sorted[1:] - d_sorted[:-1] (length 18)
        # We need k in 0..17? 
        # "k be the first index minimizing d_{k+1} - d_k"
        # d_k and d_{k+1} exist for k=0..17.
        gaps = d_sorted[1:] - d_sorted[:-1]
        min_gap = np.min(gaps)
        k = np.argmin(gaps)
        
        # D = (d_k + d_{k+1})/2
        D = (d_sorted[k] + d_sorted[k+1]) / 2.0
        
        if D < 0.1:
            scores[i] = 0.0
            continue
            
        # Select 4 distances
        if k == 0:
            d_sel = d_sorted[0:4]
        elif k == 17:
            # d_{15}..d_{18}
            d_sel = d_sorted[15:19]
        else:
            # d_{k-1}..d_{k+2}
            d_sel = d_sorted[k-1:k+3]
            
        # raw = mean of max(d/D, D/d)
        ratios = np.maximum(d_sel / D, D / d_sel)
        raw = np.mean(ratios)
        
        scores[i] = min(1.0, 1.0 / raw)
        
    return scores

def is_strictly_convex(polygon, tol=1e-6):
    """
    Checks if the polygon is strictly convex and in CCW order.
    Returns True if valid, False otherwise.
    """
    n = len(polygon)
    # Check for repeated points
    # This is already handled by score=0, but for validity check:
    # "no repeated points"
    for i in range(n):
        for j in range(i+1, n):
            if np.linalg.norm(polygon[i] - polygon[j]) < 1e-5:
                return False
                
    # Check convexity
    # Calculate cross products of consecutive edge vectors
    for i in range(n):
        p0 = polygon[i]
        p1 = polygon[(i+1) % n]
        p2 = polygon[(i+2) % n]
        
        v1 = p1 - p0
        v2 = p2 - p1
        
        # 2D cross product
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        
        # For CCW convex, cross should be > 0
        if cross <= tol:
            return False
            
    return True

def objective(params):
    """
    Objective function for optimization.
    params: 10 floats representing 5 vertices (x0,y0, ..., x4,y4)
    Returns negative of the minimum vertex score (to be minimized).
    """
    # Reconstruct 20 vertices with 4-fold rotational symmetry
    base = params.reshape(5, 2)
    vertices = []
    
    # Rotation matrix for 90 degrees
    R = np.array([[0, -1], [1, 0]])
    
    for k in range(4):
        # Rotate base vertices by k * 90 degrees
        # R^k
        if k == 0:
            rot_base = base
        elif k == 1:
            rot_base = base @ R.T
        elif k == 2:
            rot_base = base @ (R @ R).T
        else:
            rot_base = base @ (R @ R @ R).T
            
        vertices.append(rot_base)
        
    polygon = np.vstack(vertices)
    
    # Normalize: subtract centroid, divide by mean distance
    centroid = np.mean(polygon, axis=0)
    polygon = polygon - centroid
    
    dists_to_centroid = np.linalg.norm(polygon, axis=1)
    mean_dist = np.mean(dists_to_centroid)
    
    if mean_dist <= 0.01:
        return 100.0 # Invalid
        
    polygon = polygon / mean_dist
    
    # Check convexity
    if not is_strictly_convex(polygon):
        return 100.0 # Invalid
        
    # Calculate scores
    scores = compute_vertex_scores(polygon)
    min_score = np.min(scores)
    
    return -min_score

def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    time_limit = 600.0
    safety_margin = 10.0
    
    # Initial guess: Regular 20-gon
    n = 20
    radius = 1.0
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    reg_polygon = np.stack([x, y], axis=1)
    
    # Extract 5 base vertices from regular polygon for initialization
    # The regular polygon has 20-fold symmetry, so it fits 4-fold.
    # Vertices 0, 1, 2, 3, 4 are the base.
    base_init = reg_polygon[:5].copy()
    params_init = base_init.flatten()
    
    best_score = -objective(params_init)
    best_params = params_init
    
    # Run Differential Evolution
    # Bounds: [-1.5, 1.5] for x and y
    bounds = [(-1.5, 1.5)] * 10
    
    if time.time() - start_time < time_limit - safety_margin:
        try:
            res = differential_evolution(
                objective, 
                bounds, 
                seed=42, 
                maxiter=50, 
                popsize=15,
                tol=1e-4,
                polish=True
            )
            if res.fun < -best_score:
                best_score = -res.fun
                best_params = res.x
        except Exception:
            pass
            
    # Local refinement with Nelder-Mead
    if time.time() - start_time < time_limit - safety_margin:
        try:
            res_local = minimize(
                objective, 
                best_params, 
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6}
            )
            if res_local.fun < -best_score:
                best_score = -res_local.fun
                best_params = res_local.x
        except Exception:
            pass
            
    # Reconstruct final polygon
    base = best_params.reshape(5, 2)
    vertices = []
    R = np.array([[0, -1], [1, 0]])
    
    for k in range(4):
        if k == 0:
            rot_base = base
        elif k == 1:
            rot_base = base @ R.T
        elif k == 2:
            rot_base = base @ (R @ R).T
        else:
            rot_base = base @ (R @ R @ R).T
        vertices.append(rot_base)
        
    polygon = np.vstack(vertices)
    
    # Normalize
    centroid = np.mean(polygon, axis=0)
    polygon = polygon - centroid
    dists_to_centroid = np.linalg.norm(polygon, axis=1)
    mean_dist = np.mean(dists_to_centroid)
    
    if mean_dist <= 0.01:
        # Fallback to regular polygon
        return reg_polygon
        
    polygon = polygon / mean_dist
    
    # Final validity check
    if not is_strictly_convex(polygon):
        return reg_polygon
        
    return polygon

if __name__ == "__main__":
    p = equidistant_points_polygon20()
    print(p)
