import numpy as np
import time
from scipy.optimize import minimize

def equidistant_points_polygon20() -> np.ndarray:
    n = 20
    # Initial regular polygon
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)
    polygon = np.stack([x, y], axis=1)
    
    start_time = time.time()
    timeout = 580.0  # Safety margin
    
    def normalize(poly):
        centroid = np.mean(poly, axis=0)
        centered = poly - centroid
        dists = np.linalg.norm(centered, axis=1)
        mean_dist = np.mean(dists)
        if mean_dist < 1e-9:
            return centered
        return centered / mean_dist

    def vertex_score(v, poly):
        # v is a vertex, poly is (n, 2)
        # distances to other vertices
        diffs = poly - v
        dists = np.linalg.norm(diffs, axis=1)
        # remove self distance (index 0 if v is poly[0], but we pass index)
        # actually, let's pass index i
        pass

    def compute_scores(poly):
        """
        Returns array of scores for each vertex.
        """
        scores = np.zeros(n)
        for i in range(n):
            v = poly[i]
            diffs = poly - v
            dists = np.linalg.norm(diffs, axis=1)
            # remove self
            mask = np.arange(n) != i
            d = dists[mask]
            d = np.sort(d)
            
            # Check for zero distances
            if d[0] < 0.05:
                scores[i] = 0.0
                continue
                
            # Find k: first index minimizing d[k+1] - d[k]
            gaps = np.diff(d)
            k = np.argmin(gaps)
            D = (d[k] + d[k+1]) / 2.0
            
            if D < 0.1:
                scores[i] = 0.0
                continue
                
            # Select 4 distances
            if k == 0:
                selected = d[0:4]
            elif k == 17: # n-3
                selected = d[15:19]
            else:
                selected = d[k-1:k+3]
                
            # raw = mean of max(d/D, D/d)
            ratios = np.maximum(selected / D, D / selected)
            raw = np.mean(ratios)
            
            if raw == 0:
                scores[i] = 0.0
            else:
                scores[i] = min(1.0, 1.0 / raw)
                
        return scores

    def objective(params):
        poly = params.reshape(n, 2)
        # Normalize
        poly_norm = normalize(poly)
        scores = compute_scores(poly_norm)
        # We want to maximize min(scores)
        # Minimize -min(scores)
        # To make it smoother, we can use a soft min
        # But min is okay for local search if we start close
        return -np.min(scores)

    def objective_smooth(params, beta=10.0):
        poly = params.reshape(n, 2)
        poly_norm = normalize(poly)
        scores = compute_scores(poly_norm)
        # Soft min: -log(sum(exp(-beta * scores))) / beta
        # Handle zeros in scores
        scores_clipped = np.clip(scores, 1e-6, 1.0)
        soft_min = -np.log(np.sum(np.exp(-beta * scores_clipped))) / beta
        return -soft_min

    best_score = -np.min(compute_scores(normalize(polygon)))
    best_poly = polygon.copy()
    
    # Local optimization
    # Use Nelder-Mead for robustness, or L-BFGS-B
    # Let's try L-BFGS-B with numerical gradients
    x0 = polygon.flatten()
    
    # Run a few optimizations with different random perturbations
    num_tries = 5
    for i in range(num_tries):
        if time.time() - start_time > timeout:
            break
            
        if i == 0:
            x_current = x0
        else:
            # Random perturbation
            noise = np.random.normal(0, 0.01, size=x0.shape)
            x_current = x0 + noise
            
        try:
            res = minimize(objective_smooth, x_current, method='L-BFGS-B', 
                           options={'maxiter': 500, 'ftol': 1e-6, 'gtol': 1e-5})
            
            if res.success or res.fun < objective_smooth(x_current):
                poly_res = res.x.reshape(n, 2)
                # Check convexity
                # Compute cross products of consecutive edges
                # If all cross products have the same sign, it's strictly convex
                edges = np.diff(np.vstack([poly_res, poly_res[0]]), axis=0)
                cross = np.cross(edges[:-1], edges[1:])
                # Check if all positive or all negative
                if np.all(cross > 0) or np.all(cross < 0):
                    scores = compute_scores(normalize(poly_res))
                    min_score = np.min(scores)
                    if min_score > best_score:
                        best_score = min_score
                        best_poly = poly_res
        except Exception:
            pass

    # Final check and return
    # Ensure the returned polygon is the best one found
    return best_poly
