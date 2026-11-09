import numpy as np
from scipy.optimize import minimize
import time

def equidistant_points_polygon20() -> np.ndarray:
    n = 20
    # Start with regular polygon
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    
    # Parameters: amplitudes of cos(k * theta) for k=1..5
    # r_i = 1 + sum(a_k * cos(k * theta_i))
    # We use 5 harmonics, 5 parameters.
    # Symmetry: We can assume b_k=0 for simplicity (rotation invariance).
    
    def get_polygon(params):
        a = params
        radii = np.ones(n)
        for k in range(1, 6):
            radii += a[k-1] * np.cos(k * angles)
        # Ensure radii are positive
        radii = np.maximum(radii, 0.1)
        x = radii * np.cos(angles)
        y = radii * np.sin(angles)
        return np.stack([x, y], axis=1)

    def calculate_score(polygon):
        # Normalize
        centroid = np.mean(polygon, axis=0)
        poly_norm = polygon - centroid
        dists_to_centroid = np.linalg.norm(poly_norm, axis=1)
        mean_dist = np.mean(dists_to_centroid)
        if mean_dist <= 0.01:
            return -1.0 # Invalid
        
        poly_norm = poly_norm / mean_dist
        
        # Check convexity (simple check: cross products)
        # v_i = poly_norm[i]
        # e1 = v_{i+1} - v_i
        # e2 = v_{i+2} - v_{i+1}
        # cross = e1_x * e2_y - e1_y * e2_x
        # All crosses must have same sign and be non-zero
        
        vertices = poly_norm
        min_score = 1.0
        
        for i in range(n):
            v = vertices[i]
            # Distances to all other vertices
            diffs = vertices - v
            dists = np.linalg.norm(diffs, axis=1)
            # Exclude self
            dists[i] = np.inf
            # Sort
            sorted_dists = np.sort(dists)
            d = sorted_dists[:19] # 19 distances
            
            # Check if any < 0.05
            if d[0] < 0.05:
                return 0.0 # Score 0 for this vertex, so min is 0
            
            # Find k: first index minimizing d[k+1] - d[k]
            gaps = np.diff(d)
            # Handle potential numerical noise in gaps
            k = np.argmin(gaps)
            
            D = (d[k] + d[k+1]) / 2.0
            if D < 0.1:
                return 0.0
            
            # Determine cluster
            if k == 0:
                cluster = d[0:4]
            elif k == 17:
                cluster = d[15:19]
            else:
                cluster = d[k-1:k+3]
            
            # Raw score
            ratios = np.maximum(cluster / D, D / cluster)
            raw = np.mean(ratios)
            score = min(1.0, 1.0 / raw)
            
            if score < min_score:
                min_score = score
                
        return min_score

    def objective(params):
        poly = get_polygon(params)
        score = calculate_score(poly)
        # We want to maximize score, so minimize negative score
        # If score is 0, it's bad.
        # Add a small penalty if convexity is broken?
        # The score function returns 0 if invalid? 
        # The task says "invalid result... scores 0".
        # My calculate_score returns 0 if dist < 0.05 or D < 0.1.
        # It doesn't explicitly check convexity yet.
        
        # Check convexity
        vertices = poly
        centroid = np.mean(vertices, axis=0)
        vertices = vertices - centroid
        # Calculate cross products of consecutive edges
        cross_prods = []
        for i in range(n):
            v1 = vertices[i]
            v2 = vertices[(i+1)%n]
            v3 = vertices[(i+2)%n]
            e1 = v2 - v1
            e2 = v3 - v2
            cross = e1[0]*e2[1] - e1[1]*e2[0]
            cross_prods.append(cross)
        
        cross_prods = np.array(cross_prods)
        # All must be non-zero and same sign
        if np.any(np.abs(cross_prods) < 1e-9):
            return 1.0 # Bad, high cost
        if np.any(cross_prods > 0) and np.any(cross_prods < 0):
            return 1.0 # Bad, non-convex
            
        if score == 0.0:
            return 1.0 # Bad score
            
        return -score

    # Initial guess: Regular polygon (params = 0)
    # But regular polygon has score ~0.67.
    # We can start with a small perturbation to break symmetry?
    # Or just let the optimizer find the direction.
    # Since the landscape might be flat or have symmetries, 
    # a small random perturbation might help escape local minima if any.
    # But the target (d0=d1=d2=d3) is a specific geometry.
    
    x0 = np.zeros(5)
    
    # Use Nelder-Mead for robustness
    start_time = time.time()
    deadline = start_time + 500 # 500s margin
    
    result = None
    best_score = -1
    
    # Run minimization
    try:
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 2000, 'maxfev': 2000})
        
        if time.time() > deadline:
             # Timeout
             pass
        else:
             result = res
             best_score = calculate_score(get_polygon(res.x))
             
    except Exception:
        pass
    
    # If the result is not perfect, try a few random restarts?
    # Or just return the best found.
    
    # To ensure we beat the baseline (1.0), we need score > 0.67?
    # Wait, the baseline score is 1.0 in the prompt?
    # "CURRENT PROGRAM (score 1.0000)"
    # The prompt says the regular polygon scores 1.0?
    # Let me re-read the scoring rule carefully.
    
    # "Vertex score: ... raw = mean ... vertex score = min(1, 1/raw)."
    # For regular 20-gon:
    # d = [0.313, 0.313, 0.618, 0.618, ...]
    # Gaps: 0, 0.305, 0, 0.290 ...
    # First min gap is at k=0 (gap 0).
    # Cluster: d[0..3] = [0.313, 0.313, 0.618, 0.618].
    # D = (0.313 + 0.313)/2 = 0.313.
    # Ratios: 1, 1, 1.97, 1.97.
    # Raw = 1.48.
    # Score = 1/1.48 = 0.67.
    
    # Why does the prompt say score 1.0000?
    # "The search score is value/seed, so 1.0 equals the seed program".
    # Ah, the "score" in the prompt title "score 1.0000" refers to the **relative** score (value/seed).
    # The **raw** value of the regular polygon is ~0.67.
    # So if I produce a polygon with raw score 0.68, my relative score is 0.68/0.67 > 1.0.
    # I need to maximize the raw score.
    
    # So the target is to get raw score close to 1.0.
    # This requires d0=d1=d2=d3.
    
    # Let's refine the optimization.
    # The Nelder-Mead might get stuck.
    # Let's try to manually construct a better starting point or use a gradient-based method.
    
    # Actually, let's just return the result from the optimizer.
    # If it fails, return the regular polygon.
    
    if result is None or result.fun >= 0: # fun >= 0 means bad (since we minimize -score)
         # Fallback to regular polygon
         x = np.cos(angles)
         y = np.sin(angles)
         return np.stack([x, y], axis=1)
    
    best_poly = get_polygon(result.x)
    
    # Final sanity check: normalize and check validity
    centroid = np.mean(best_poly, axis=0)
    best_poly = best_poly - centroid
    dists = np.linalg.norm(best_poly, axis=1)
    mean_dist = np.mean(dists)
    if mean_dist > 0.01:
        best_poly = best_poly / mean_dist
        
    return best_poly
