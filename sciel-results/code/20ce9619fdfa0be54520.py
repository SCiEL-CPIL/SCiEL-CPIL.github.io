import numpy as np
from scipy.optimize import minimize

def calculate_score(angles):
    """
    Calculate the minimum vertex score for a polygon defined by angles on a unit circle.
    """
    n = len(angles)
    # Sort angles to ensure cyclic order? 
    # The input angles should be sorted. 
    # If we optimize angles, we must maintain order.
    # Let's assume angles are sorted in [0, 2*pi).
    
    x = np.cos(angles)
    y = np.sin(angles)
    pts = np.column_stack((x, y))
    
    # Calculate pairwise distances
    # dist[i, j] is distance from i to j
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    
    min_score = np.inf
    
    for i in range(n):
        # Get distances from i to all other vertices
        d = dist[i, :].copy()
        d[i] = np.inf # Exclude self
        
        # Sort distances
        d_sorted = np.sort(d)
        
        # Check for repeated points (should not happen if angles distinct)
        # Check min distance > 0.05
        if d_sorted[0] < 0.05:
            return 0.0 # Or a very small number to penalize
        
        # Find k: first index minimizing d[k+1] - d[k]
        gaps = np.diff(d_sorted)
        k = np.argmin(gaps)
        
        D = (d_sorted[k] + d_sorted[k+1]) / 2.0
        if D < 0.1:
            return 0.0
        
        # Determine the 4 distances
        if k == 0:
            d_four = d_sorted[0:4]
        elif k == n - 2: # k=17 for n=20 (indices 0..18, so max k is 17? No, d has 19 elements. Indices 0..18. Gaps 0..17. So k in 0..17. If k=17, d[17], d[18].)
             # Wait, d_sorted has 19 elements.
             # If k=17, d[17] and d[18] are the pair.
             # Task: "d_15..d_18 if k = 17"
             d_four = d_sorted[15:19]
        else:
            # d_{k-1} .. d_{k+2}
            d_four = d_sorted[k-1:k+3]
            
        # Calculate raw score
        # raw = mean of max(d/D, D/d)
        ratios = np.maximum(d_four / D, D / d_four)
        raw = np.mean(ratios)
        
        score_i = min(1.0, 1.0 / raw)
        
        if score_i < min_score:
            min_score = score_i
            
    return min_score

def objective(angles):
    # We want to maximize min_score, so minimize -min_score
    # Use smooth min approximation for gradient-based methods if needed, 
    # but Nelder-Mead doesn't need gradients.
    # For Nelder-Mead, just returning -score is fine, though non-smooth.
    score = calculate_score(angles)
    return -score

def equidistant_points_polygon20() -> np.ndarray:
    n = 20
    # Initial guess: regular polygon
    angles0 = np.linspace(0, 2*np.pi, n, endpoint=False)
    
    # Optimization
    # We need to ensure angles remain sorted and within [0, 2*pi).
    # Nelder-Mead might step out of bounds.
    # We can use a penalty or transform variables.
    # Let's try optimizing deltas from the regular polygon.
    # delta_i is small.
    # angle_i = base_angle_i + delta_i
    # To maintain order, we can optimize the gaps between angles.
    # Let gaps g_i > 0, sum(g_i) = 2*pi.
    # This ensures convexity and cyclic order.
    
    def gaps_to_angles(gaps):
        # Normalize gaps to sum to 2*pi
        gaps = np.abs(gaps) # Ensure positive
        s = np.sum(gaps)
        if s == 0:
            return np.linspace(0, 2*np.pi, n, endpoint=False)
        gaps = gaps / s * 2 * np.pi
        angles = np.cumsum(gaps)
        angles[0] = 0 # Fix rotation
        # The last angle should be close to 2*pi but not equal.
        # Actually, cumsum of 20 gaps summing to 2*pi gives angles 0, ..., 2*pi.
        # We want 20 points. The 20th point is at 2*pi which is same as 0.
        # So we take the first 20 cumulative sums? No.
        # If we have 20 gaps, we have 20 vertices.
        # v_0 at 0.
        # v_1 at g_0.
        # ...
        # v_{19} at sum(g_0...g_{18}).
        # g_{19} closes the loop.
        # So we optimize 20 gaps.
        angles = np.zeros(n)
        angles[1:] = np.cumsum(gaps[:-1])
        return angles

    def objective_gaps(gaps):
        angles = gaps_to_angles(gaps)
        return -calculate_score(angles)

    # Initial gaps are equal
    gaps0 = np.ones(n)
    
    # Run Nelder-Mead
    # maxiter needs to be reasonable.
    # n=20 parameters.
    res = minimize(objective_gaps, gaps0, method='Nelder-Mead', 
                   options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8})
    
    # Get best angles
    best_gaps = res.x
    best_angles = gaps_to_angles(best_gaps)
    
    # Construct polygon
    x = np.cos(best_angles)
    y = np.sin(best_angles)
    polygon = np.column_stack((x, y))
    
    return polygon
