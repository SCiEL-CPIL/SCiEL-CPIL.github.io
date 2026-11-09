# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize
import time

def equidistant_points_polygon20() -> np.ndarray:
    n = 20
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    
    # Helper to compute the score of a polygon given as (n, 2) array
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
        # Strictly convex: no repeated points, every turn nonzero and same sign.
        # We check cross products of consecutive edges.
        # Edge i: P_{i+1} - P_i
        # Cross product of Edge i and Edge i+1 should be > 0 (assuming CCW) or < 0 (CW).
        # Since we start with CCW, we expect > 0.
        
        edges = np.roll(normalized, -1, axis=0) - normalized
        # Cross product in 2D: e1_x * e2_y - e1_y * e2_x
        cross_products = edges[:, 0] * np.roll(edges, -1, axis=0)[:, 1] - edges[:, 1] * np.roll(edges, -1, axis=0)[:, 0]
        
        if np.any(cross_products <= 1e-9): # Allow small numerical noise, but strictly convex requires > 0
             # If any cross product is <= 0, it's not strictly convex (or is CW).
             # If it's CW, all cross products would be negative.
             if np.all(cross_products < 0):
                 # It's a valid convex polygon, just clockwise. We can flip it or just accept.
                 # But the task says "every turn ... of one sign".
                 # So CW is valid.
                 pass
             else:
                 return 0.0
                 
        # Check for repeated points
        # Min distance between any pair should be > 0
        # This is O(N^2) but N=20 is small.
        # Actually, if cross products are non-zero, points are not collinear adjacent.
        # Non-adjacent repeats?
        # "No repeated points".
        # We can check min pairwise distance.
        # But if it's convex and no 3 collinear, points are distinct.
        # Let's just check min distance between all pairs to be safe.
        # dist_matrix = np.linalg.norm(normalized[:, np.newaxis, :] - normalized[np.newaxis, :, :], axis=2)
        # if np.min(dist_matrix[np.triu_indices(n, k=1)]) < 1e-6: return 0.0
        # This might be slow if called many times? No, 20x20 is tiny.
        
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
            
            # d_0 ... d_18
            # Find k minimizing d_{k+1} - d_k
            # Note: sorted_dists has 19 elements (indices 0 to 18).
            # Differences are between 0-1, 1-2, ..., 17-18. (18 diffs)
            diffs_sorted = np.diff(sorted_dists)
            k = np.argmin(diffs_sorted)
            
            # Determine the 4 distances to use
            if k == 0:
                # d_0 .. d_3
                # Wait, if k=0, the gap is between d_0 and d_1.
                # Task: "d_0..d_3 if k=0"
                d_subset = sorted_dists[0:4]
            elif k == 17: # Max index for diff is 17 (between d_17 and d_18)
                # Task: "d_15..d_18 if k=17"
                d_subset = sorted_dists[15:19]
            else:
                # Task: "d_{k-1}..d_{k+2}"
                # This is 4 elements: index k-1, k, k+1, k+2
                d_subset = sorted_dists[k-1:k+3]
                
            # Check if any d < 0.05
            if np.any(d_subset < 0.05):
                min_vertex_score = 0.0
                break
                
            D = (d_subset[1] + d_subset[2]) / 2.0 # Wait, D is (d_k + d_{k+1})/2
            # The task says: "Let k be the first index minimizing d_{k+1} - d_k and D = (d_k + d_{k+1})/2"
            # So D is the midpoint of the smallest gap.
            D_val = (sorted_dists[k] + sorted_dists[k+1]) / 2.0
            
            if D_val < 0.1:
                min_vertex_score = 0.0
                break
                
            # raw = mean over these four d of max(d/D, D/d)
            # D is D_val
            ratios = np.maximum(d_subset / D_val, D_val / d_subset)
            raw = np.mean(ratios)
            
            vertex_score = min(1.0, 1.0 / raw)
            
            if vertex_score < min_vertex_score:
                min_vertex_score = vertex_score
                
        return min_vertex_score

    def objective(radii):
        # radii is a vector of length 20
        x = radii * np.cos(angles)
        y = radii * np.sin(angles)
        polygon = np.stack([x, y], axis=1)
        
        score = compute_score(polygon)
        
        # If score is 0, it might be due to non-convexity or bad distances.
        # We want to maximize score, so minimize -score.
        # But if score is 0 because of non-convexity, we might want a stronger penalty
        # to guide DE away.
        # However, compute_score already returns 0 for invalid.
        # Let's add a small penalty if not convex to help gradients?
        # DE doesn't use gradients.
        # Just returning -score is fine.
        
        return -score

    # Initial guess: regular polygon
    r0 = np.ones(n)
    
    # Bounds for radii
    # If radii are too small or large, convexity might break or distances become weird.
    # Let's try [0.5, 2.0]
    bounds = [(0.5, 2.0) for _ in range(n)]
    
    start_time = time.time()
    deadline = start_time + 500 # 500s for optimization, 100s buffer
    
    best_score = 0.0
    best_radii = r0.copy()
    
    # Try Differential Evolution
    # Use a smaller population and maxiter to fit in time
    # N=20, popsize=20, maxiter=100 -> 2000 evals.
    # Each eval is fast (N=20).
    
    try:
        de_result = scipy.optimize.differential_evolution(
            objective, 
            bounds, 
            seed=42, 
            maxiter=100, 
            popsize=20,
            tol=1e-4,
            polish=False, # We will polish later
            # update_interval=10,
            # callback=lambda x, convergence: None
        )
        
        if de_result.fun < -best_score:
            best_score = -de_result.fun
            best_radii = de_result.x
            
    except Exception as e:
        pass # If DE fails, stick with initial
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > deadline:
         # Return best so far
         x = best_radii * np.cos(angles)
         y = best_radii * np.sin(angles)
         return np.stack([x, y], axis=1)

    # Polish with Nelder-Mead
    # Use the best radii from DE as starting point
    try:
        nm_result = scipy.optimize.minimize(
            objective, 
            best_radii, 
            method='Nelder-Mead', 
            options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6}
        )
        
        if nm_result.fun < -best_score:
            best_score = -nm_result.fun
            best_radii = nm_result.x
            
    except Exception as e:
        pass
        
    # One more polish round if time permits?
    elapsed = time.time() - start_time
    if elapsed < deadline - 50:
        try:
            nm_result2 = scipy.optimize.minimize(
                objective, 
                best_radii, 
                method='Nelder-Mead', 
                options={'maxiter': 2000, 'xatol': 1e-7, 'fatol': 1e-7}
            )
            if nm_result2.fun < -best_score:
                best_score = -nm_result2.fun
                best_radii = nm_result2.x
        except Exception as e:
            pass

    x = best_radii * np.cos(angles)
    y = best_radii * np.sin(angles)
    polygon = np.stack([x, y], axis=1)
    
    # Final validation
    final_score = compute_score(polygon)
    if final_score == 0.0:
        # Fallback to regular polygon if optimization failed
        return np.stack([np.cos(angles), np.sin(angles)], axis=1)
        
    return polygon

# EVOLVE-BLOCK-END
