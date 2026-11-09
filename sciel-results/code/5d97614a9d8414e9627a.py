# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as opt

def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex polygon with n = 20 vertices that tries to be a counterexample to the
    conjecture "every convex polygon has a vertex v with no 4 other vertices equidistant
    from v". The goal is a polygon where EVERY vertex has (at least) 4 other vertices that
    are approximately equidistant from it. Quality is the minimum, over all vertices, of a
    per-vertex equidistance score in [0, 1] (1.0 = perfect counterexample).

    This implementation searches for an optimal radial perturbation of a regular polygon
    of the form r(theta) = 1 + epsilon * cos(k * theta). We focus on k=5 (5-fold symmetry)
    as it divides 20 and may create favorable distance structures.
    """
    n = 20
    start_time = time.time()
    timeout = 600.0  # seconds
    safety_margin = 10.0

    # Precompute base angles
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    
    # We will search over epsilon for a few harmonic modes k
    # k=5 is a strong candidate due to symmetry with n=20
    # k=4, k=6, k=10 are also divisors or near divisors
    
    best_score = -np.inf
    best_params = None
    best_k = 5
    best_phase = 0.0

    def compute_polygon(eps, k, phase):
        r = 1.0 + eps * np.cos(k * angles + phase)
        x = r * np.cos(angles)
        y = r * np.sin(angles)
        return np.stack([x, y], axis=1)

    def vertex_score(polygon):
        """
        Calculate the score for a single polygon.
        Returns the minimum vertex score.
        """
        # Normalize: subtract centroid, divide by mean distance to centroid
        centroid = np.mean(polygon, axis=0)
        centered = polygon - centroid
        dists_to_centroid = np.linalg.norm(centered, axis=1)
        mean_dist = np.mean(dists_to_centroid)
        
        if mean_dist <= 0.01:
            return 0.0
        
        normalized = centered / mean_dist
        
        # Check strict convexity
        # No repeated points
        if np.any(np.diff(np.sort(np.linalg.norm(normalized, axis=1)) < 1e-9)):
             # This check is a bit weak, let's check pairwise distances
             pass
        
        # Check turns
        # v0, v1, v2 ...
        # Edge vectors
        edges = np.roll(normalized, -1, axis=0) - normalized
        # Cross product z-component
        cross_z = edges[:, 0] * np.roll(edges, -1, axis=0)[:, 1] - edges[:, 1] * np.roll(edges, -1, axis=0)[:, 0]
        
        # All turns must be nonzero and same sign
        if np.any(cross_z == 0):
            return 0.0
        if not (np.all(cross_z > 0) or np.all(cross_z < 0)):
            return 0.0
            
        # Total turning +-2*pi is guaranteed for simple convex polygon
        
        # No repeated points
        for i in range(n):
            for j in range(i+1, n):
                if np.linalg.norm(normalized[i] - normalized[j]) < 1e-9:
                    return 0.0

        min_v_score = 1.0
        
        # Calculate vertex scores
        # Distances between all pairs
        # dist[i, j] is distance between vertex i and j
        diff = normalized[:, np.newaxis, :] - normalized[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        
        # For each vertex i
        for i in range(n):
            # Get distances from i to all other vertices
            d = dists[i].copy()
            d[i] = np.inf # Exclude self
            
            # Sort distances
            d_sorted = np.sort(d)
            
            # d_0 ... d_18
            if d_sorted[0] < 0.05:
                min_v_score = 0.0
                break
                
            # Find k: first index minimizing d_{k+1} - d_k
            diffs = d_sorted[1:] - d_sorted[:-1]
            k = np.argmin(diffs)
            
            D = (d_sorted[k] + d_sorted[k+1]) / 2.0
            
            if D < 0.1:
                min_v_score = 0.0
                break
                
            # Select 4 distances
            if k == 0:
                idxs = [0, 1, 2, 3]
            elif k == 17: # Note: indices are 0..18. k is index in diffs (0..17).
                          # If k=17, it minimizes d_18 - d_17.
                          # The rule says: d_15..d_18 if k=17.
                idxs = [15, 16, 17, 18]
            else:
                idxs = [k-1, k, k+1, k+2]
                
            selected_dists = d_sorted[idxs]
            
            # raw = mean over these four d of max(d/D, D/d)
            ratios = np.maximum(selected_dists / D, D / selected_dists)
            raw = np.mean(ratios)
            
            v_score = min(1.0, 1.0 / raw)
            
            if v_score < min_v_score:
                min_v_score = v_score
                
        return min_v_score

    def objective(params):
        # params: [epsilon, phase]
        eps, phase = params
        # Ensure epsilon is small enough to maintain convexity
        # For r = 1 + eps cos(k theta), convexity requires eps < 1/k roughly?
        # Actually, curvature kappa = r'' + r / (r^2 + r'^2)^(3/2)
        # For small eps, kappa ~ 1 - eps k^2 cos(k theta) ... 
        # So eps < 1/k^2 is a safe bound. For k=5, eps < 0.04.
        # Let's allow up to 0.05 to be safe, but optimizer might go higher.
        # We'll clip in the objective to avoid invalid polygons scoring 0.
        
        if time.time() - start_time > timeout - safety_margin:
            return 1e9 # Penalty to stop optimization

        poly = compute_polygon(eps, best_k, phase)
        score = vertex_score(poly)
        return -score # Minimize negative score

    # Search over a few k values
    # k=5 is primary candidate
    # k=4, k=6, k=10
    candidate_ks = [5, 4, 6, 10, 2, 1]
    
    for k in candidate_ks:
        if time.time() - start_time > timeout - safety_margin:
            break
            
        best_k = k
        # Initial guess: small epsilon
        # Try a few initial points
        initial_guesses = [
            [0.01, 0.0],
            [0.02, 0.0],
            [0.03, 0.0],
            [0.01, np.pi/10],
            [0.02, np.pi/10]
        ]
        
        for guess in initial_guesses:
            if time.time() - start_time > timeout - safety_margin:
                break
            
            try:
                # Bounds: epsilon in [0, 0.05], phase in [0, 2*pi]
                bounds = [(0.0, 0.05), (0.0, 2.0*np.pi)]
                
                res = opt.minimize(objective, guess, method='L-BFGS-B', bounds=bounds, 
                                   options={'maxiter': 50, 'ftol': 1e-6})
                
                if res.fun < 0: # Valid score found
                    score = -res.fun
                    if score > best_score:
                        best_score = score
                        best_params = res.x
                        # best_k is already set in loop
                        best_phase = res.x[1]
                        
            except Exception:
                pass

    # If optimization didn't find anything better than regular polygon, fallback to regular
    if best_score < 1.0 or best_params is None:
        # Return regular polygon
        radius = 1.0
        x = radius * np.cos(angles)
        y = radius * np.sin(angles)
        polygon = np.stack([x, y], axis=1)
        return polygon

    # Construct the best polygon found
    eps, phase = best_params
    polygon = compute_polygon(eps, best_k, phase)
    
    return polygon

# EVOLVE-BLOCK-END
