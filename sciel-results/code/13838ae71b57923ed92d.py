# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize
import time

def equidistant_points_polygon20() -> np.ndarray:
    n = 20
    # We parameterize the angles by 19 weights.
    # Let w = [w_1, ..., w_19].
    # angles[0] = 0
    # angles[i] = sum(w_1...w_i)
    # angles[n-1] should be close to 2*pi.
    # To ensure the polygon closes and is convex, we can just use the cumulative sum
    # and normalize the angles to [0, 2*pi).
    # Actually, for a strictly convex polygon with vertices in cyclic order,
    # it suffices to define the angular gaps between consecutive vertices.
    # Let gaps = [g_1, ..., g_20] where sum(gaps) = 2*pi and g_i > 0.
    # Since sum is fixed, we only need to optimize 19 independent gaps.
    # Or, simpler: optimize 19 weights, compute cumulative sum, normalize to 2*pi.
    
    # Let's use 19 parameters representing the angular gaps between vertex i and i+1 (mod 20),
    # but since the last gap is determined by the others to sum to 2*pi, we optimize 19 values.
    # Let x = [x_1, ..., x_19].
    # gaps = [x_1, ..., x_19, 2*pi - sum(x)]
    # angles = cumsum([0, gaps[0], ..., gaps[18]]) -> 20 angles.
    
    def get_points_from_weights(x):
        # x is an array of 19 values.
        # We map them to positive gaps.
        # Using softmax or exp to ensure positivity?
        # Or just bound them in DE.
        # Let's assume x are the first 19 gaps, and they are positive.
        # The 20th gap is 2*pi - sum(x). It must also be positive.
        
        gaps = np.zeros(n)
        gaps[:19] = x
        gaps[19] = 2.0 * np.pi - np.sum(x)
        
        # If any gap is <= 0, the polygon is not strictly convex (or degenerate).
        # We handle this in the objective function by returning a bad score.
        
        # Compute angles
        angles = np.cumsum(np.concatenate(([0.0], gaps[:19])))
        # The last angle should be 2*pi - gaps[19].
        # cumsum of [0, g1...g19] gives 0, g1, g1+g2... sum(g1..g19).
        # The 20th point is at angle sum(g1..g19).
        # The gap from point 19 to point 0 (angle 0/2pi) is 2*pi - sum(g1..g19) = gaps[19].
        # So the angles are correct for vertices 0..19.
        
        # Ensure angles are in [0, 2*pi)
        # angles[0] = 0
        # angles[19] = sum(x)
        
        # Points on unit circle
        x_coords = np.cos(angles)
        y_coords = np.sin(angles)
        return np.column_stack((x_coords, y_coords))

    def compute_score(polygon):
        # Normalize
        centroid = np.mean(polygon, axis=0)
        centered = polygon - centroid
        dists_to_centroid = np.linalg.norm(centered, axis=1)
        mean_dist = np.mean(dists_to_centroid)
        
        if mean_dist <= 0.01:
            return 0.0
            
        normalized = centered / mean_dist
        
        # Check Convexity
        # For points on a circle in cyclic order with positive angular gaps,
        # the polygon is strictly convex.
        # We just need to ensure all gaps are positive.
        # This is handled by the objective function penalty if we pass invalid weights.
        # But here we just compute the score assuming valid input.
        
        # Check for repeated points (should not happen if gaps > 1e-6)
        
        # Calculate Vertex Scores
        min_vertex_score = 1.0
        
        for i in range(n):
            v = normalized[i]
            # Distances to other vertices
            diffs = normalized - v
            dists = np.linalg.norm(diffs, axis=1)
            dists[i] = np.inf # Exclude self
            
            # Sort distances
            sorted_dists = np.sort(dists)
            
            # Find k minimizing d_{k+1} - d_k
            diffs_sorted = np.diff(sorted_dists)
            k = np.argmin(diffs_sorted)
            
            # Determine the 4 distances to use
            if k == 0:
                d_subset = sorted_dists[0:4]
            elif k == 17: # Max index for diff is 17 (between d_17 and d_18)
                d_subset = sorted_dists[15:19]
            else:
                d_subset = sorted_dists[k-1:k+3]
                
            # Check if any d < 0.05
            if np.any(d_subset < 0.05):
                return 0.0
                
            D_val = (sorted_dists[k] + sorted_dists[k+1]) / 2.0
            
            if D_val < 0.1:
                return 0.0
                
            # raw = mean over these four d of max(d/D, D/d)
            ratios = np.maximum(d_subset / D_val, D_val / d_subset)
            raw = np.mean(ratios)
            
            vertex_score = min(1.0, 1.0 / raw)
            
            if vertex_score < min_vertex_score:
                min_vertex_score = vertex_score
                
        return min_vertex_score

    def objective(x):
        # x: 19 weights
        # Check if valid (all gaps positive)
        gaps = np.zeros(n)
        gaps[:19] = x
        gaps[19] = 2.0 * np.pi - np.sum(x)
        
        if np.any(gaps <= 1e-6):
            return 1.0 # Bad score (we minimize negative score, so 1.0 is bad for -score)
            # Actually, let's return a large penalty.
            # We are minimizing -score. So we want to return large positive for bad inputs.
            return 100.0
            
        polygon = get_points_from_weights(x)
        score = compute_score(polygon)
        
        # If score is 0, it's invalid or very bad.
        # We want to maximize score, so minimize -score.
        return -score

    # Initial guess: regular polygon
    # Regular polygon has equal gaps: 2*pi/20
    gap_reg = 2.0 * np.pi / n
    x0 = np.full(19, gap_reg)
    
    # Bounds for DE
    # Gaps must be positive and sum < 2*pi.
    # Individual gap min: 1e-4
    # Individual gap max: 2*pi - 19*1e-4
    lower_bound = 1e-4
    upper_bound = 2.0 * np.pi - 19 * lower_bound
    
    bounds = [(lower_bound, upper_bound) for _ in range(19)]
    
    start_time = time.time()
    deadline = start_time + 500 # 500s for optimization, 130s buffer
    
    best_score = 0.0
    best_x = x0.copy()
    
    # Evaluate initial
    init_score = compute_score(get_points_from_weights(x0))
    best_score = init_score
    
    # Differential Evolution
    try:
        de_result = scipy.optimize.differential_evolution(
            objective, 
            bounds, 
            seed=42, 
            maxiter=200, 
            popsize=20,
            tol=1e-4,
            polish=False
        )
        
        if de_result.fun < -best_score:
            best_score = -de_result.fun
            best_x = de_result.x
            
    except Exception as e:
        pass # If DE fails, stick with initial
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > deadline - 10:
         polygon = get_points_from_weights(best_x)
         return polygon

    # Polish with Nelder-Mead
    try:
        nm_result = scipy.optimize.minimize(
            objective, 
            best_x, 
            method='Nelder-Mead', 
            options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8}
        )
        
        if nm_result.fun < -best_score:
            best_score = -nm_result.fun
            best_x = nm_result.x
            
    except Exception as e:
        pass
        
    # One more polish round if time permits
    elapsed = time.time() - start_time
    if elapsed < deadline - 20:
        try:
            nm_result2 = scipy.optimize.minimize(
                objective, 
                best_x, 
                method='Nelder-Mead', 
                options={'maxiter': 5000, 'xatol': 1e-9, 'fatol': 1e-9}
            )
            if nm_result2.fun < -best_score:
                best_score = -nm_result2.fun
                best_x = nm_result2.x
        except Exception as e:
            pass

    polygon = get_points_from_weights(best_x)
    
    # Final validation
    final_score = compute_score(polygon)
    if final_score == 0.0:
        # Fallback to regular polygon
        angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        return np.column_stack((np.cos(angles), np.sin(angles)))
        
    return polygon

# EVOLVE-BLOCK-END
