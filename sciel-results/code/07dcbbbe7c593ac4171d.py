import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

def equidistant_points_polygon20() -> np.ndarray:
    """
    Finds a convex 20-gon maximizing the minimum vertex equidistance score.
    Uses radial parameterization r_i and optimizes with Differential Evolution
    followed by local refinement with a smooth surrogate.
    """
    n = 20
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    
    # Time limit guard
    start_time = time.time()
    time_limit = 580  # 20s margin
    
    def get_polygon(radii):
        """Convert radii to polygon vertices."""
        x = radii * np.cos(angles)
        y = radii * np.sin(angles)
        return np.column_stack((x, y))

    def check_convexity(polygon):
        """Returns True if polygon is strictly convex and CCW."""
        # Check for repeated points
        for i in range(n):
            if np.linalg.norm(polygon[i] - polygon[(i+1)%n]) < 1e-6:
                return False
        
        # Check cross products
        for i in range(n):
            p0 = polygon[i]
            p1 = polygon[(i+1)%n]
            p2 = polygon[(i+2)%n]
            v1 = p1 - p0
            v2 = p2 - p1
            cross = v1[0]*v2[1] - v1[1]*v2[0]
            if cross <= 1e-6: # Allow small tolerance, but strictly positive required
                return False
        return True

    def vertex_score(polygon, v_idx):
        """Calculate the score for a specific vertex."""
        v = polygon[v_idx]
        # Distances to all other vertices
        diffs = polygon - v
        dists = np.linalg.norm(diffs, axis=1)
        # Exclude self distance (index v_idx)
        # Create a mask or just remove the element
        dists = np.delete(dists, v_idx)
        dists = np.sort(dists)
        
        # Check for small distances
        if dists[0] < 0.05:
            return 0.0
            
        # Find k: first index minimizing d_{k+1} - d_k
        gaps = np.diff(dists)
        k = np.argmin(gaps)
        
        D = (dists[k] + dists[k+1]) / 2.0
        if D < 0.1:
            return 0.0
            
        # Select 4 distances
        if k == 0:
            selected = dists[0:4]
        elif k == 17: # n-3 = 17
            selected = dists[15:19]
        else:
            selected = dists[k-1:k+3]
            
        # Calculate raw score
        ratios = np.maximum(selected / D, D / selected)
        raw = np.mean(ratios)
        
        return min(1.0, 1.0 / raw)

    def total_score(polygon):
        """Calculate the minimum vertex score."""
        scores = [vertex_score(polygon, i) for i in range(n)]
        return min(scores)

    def objective(radii):
        """Objective function for optimization. Returns negative score."""
        polygon = get_polygon(radii)
        
        # Convexity check
        if not check_convexity(polygon):
            return 100.0 # Large penalty
            
        # Normalization is done in scoring? 
        # The prompt says: "The polygon is normalized: subtract the centroid, divide by the mean distance to it".
        # My vertex_score uses raw coordinates. I should normalize inside the score calculation 
        # or ensure the input is normalized. 
        # Since the score is scale/translation invariant, I can just compute on the raw polygon 
        # BUT the thresholds 0.05 and 0.1 are absolute.
        # Wait, "if that mean is <= 0.01 the score is 0".
        # And "if any is < 0.05".
        # This implies the coordinates MUST be normalized before scoring.
        
        # Let's normalize the polygon inside the objective function to match evaluator.
        centroid = np.mean(polygon, axis=0)
        polygon_norm = polygon - centroid
        mean_dist = np.mean(np.linalg.norm(polygon_norm, axis=1))
        if mean_dist <= 0.01:
            return 100.0
        polygon_norm = polygon_norm / mean_dist
        
        # Now compute score on normalized polygon
        scores = [vertex_score(polygon_norm, i) for i in range(n)]
        min_score = min(scores)
        
        return -min_score

    # Optimization
    # Bounds for radii
    # Regular polygon radius 1.0.
    # Allow variation [0.5, 1.5]
    bounds = [(0.5, 1.5)] * n
    
    # Initial DE
    best_result = None
    best_score = -100.0
    
    if time.time() - start_time < time_limit:
        de_result = differential_evolution(
            objective, 
            bounds, 
            maxiter=50, 
            popsize=10, 
            tol=1e-4, 
            seed=42,
            polish=False # We will do our own polish
        )
        
        if de_result.fun < 100.0: # Valid result
            best_result = de_result.x
            best_score = -de_result.fun

    # Local refinement with smooth surrogate
    if best_result is not None and time.time() - start_time < time_limit - 60:
        # Use log-sum-exp surrogate
        tau = 50.0
        
        def smooth_objective(radii):
            polygon = get_polygon(radii)
            if not check_convexity(polygon):
                return 100.0
            
            centroid = np.mean(polygon, axis=0)
            polygon_norm = polygon - centroid
            mean_dist = np.mean(np.linalg.norm(polygon_norm, axis=1))
            if mean_dist <= 0.01:
                return 100.0
            polygon_norm = polygon_norm / mean_dist
            
            scores = np.array([vertex_score(polygon_norm, i) for i in range(n)])
            
            # Smooth min: -1/tau * log(sum(exp(-tau * S_i)))
            # To avoid overflow, use log-sum-exp trick
            max_s = np.max(scores)
            log_sum = np.log(np.sum(np.exp(-tau * (scores - max_s)))) + (-tau * max_s)
            return -(-1.0/tau * log_sum) # Minimizing negative smooth min
            
        # Try a few refinements with increasing tau
        current_x = best_result
        for t in [10, 50, 100, 200]:
            if time.time() - start_time > time_limit - 30:
                break
            try:
                res = minimize(
                    smooth_objective, 
                    current_x, 
                    method='BFGS', 
                    options={'maxiter': 100, 'gtol': 1e-5}
                )
                if res.fun < 100.0:
                    # Verify with exact score
                    exact_val = objective(res.x)
                    if exact_val < objective(current_x):
                        current_x = res.x
                        best_score = -exact_val
            except Exception:
                pass

    # Final check and return
    if best_result is None:
        # Fallback to regular polygon
        return np.column_stack((np.cos(angles), np.sin(angles)))
    
    # Ensure we return the best one found
    # Re-evaluate to be sure
    final_poly = get_polygon(best_result)
    
    # One last polish with Nelder-Mead on exact objective if time permits
    if time.time() - start_time < time_limit - 10:
        try:
            res = minimize(
                objective, 
                best_result, 
                method='Nelder-Mead', 
                options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6}
            )
            if res.fun < objective(best_result):
                best_result = res.x
                final_poly = get_polygon(best_result)
        except Exception:
            pass
            
    return final_poly
