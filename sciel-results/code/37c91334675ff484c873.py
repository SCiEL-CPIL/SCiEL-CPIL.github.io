import numpy as np
import time
import sys
from scipy.optimize import minimize

def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex polygon with n = 20 vertices that tries to be a counterexample to the
    conjecture "every convex polygon has a vertex v with no 4 other vertices equidistant
    from v". The goal is a polygon where EVERY vertex has (at least) 4 other vertices that
    are approximately equidistant from it. Quality is the minimum, over all vertices, of a
    per-vertex equidistance score in [0, 1] (1.0 = perfect counterexample).

    This baseline is the upstream search's starting construction: a regular polygon whose
    vertices lie evenly on a circle (deterministic, strictly convex). The scoring function
    is scale/translation invariant, so the circle radius is arbitrary.

    Returns:
        polygon: np.ndarray of shape (20, 2) with the x,y coordinates of the vertices,
            given in convex (counter-clockwise) order.
    """
    n = 20
    start_time = time.time()
    deadline = 600.0
    
    # Fixed angles for the 20 vertices
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    
    def get_polygon(radii):
        """Convert radii to cartesian coordinates."""
        x = radii * np.cos(angles)
        y = radii * np.sin(angles)
        return np.stack([x, y], axis=1)

    def normalize(polygon):
        """Normalize polygon: subtract centroid, divide by mean distance to centroid."""
        centroid = np.mean(polygon, axis=0)
        centered = polygon - centroid
        dists = np.linalg.norm(centered, axis=1)
        mean_dist = np.mean(dists)
        if mean_dist <= 0.01:
            return None
        return centered / mean_dist

    def is_convex(polygon):
        """Check if polygon is strictly convex."""
        n = len(polygon)
        if n < 3:
            return False
        cross_products = []
        for i in range(n):
            p0 = polygon[i]
            p1 = polygon[(i + 1) % n]
            p2 = polygon[(i + 2) % n]
            v1 = p1 - p0
            v2 = p2 - p1
            cross = v1[0] * v2[1] - v1[1] * v2[0]
            cross_products.append(cross)
        
        # Check if all cross products have the same sign and are non-zero
        # For CCW order, they should be positive
        if any(c == 0 for c in cross_products):
            return False
        if np.all(np.array(cross_products) > 0) or np.all(np.array(cross_products) < 0):
            return True
        return False

    def vertex_score(v, polygon):
        """Calculate the score for a single vertex v in the normalized polygon."""
        # Distances to all other vertices
        dists = np.linalg.norm(polygon - v, axis=1)
        # Exclude self (distance 0)
        dists = np.sort(dists[1:]) # There are 19 other vertices
        
        if len(dists) != 19:
            return 0.0
            
        if np.any(dists < 0.05):
            return 0.0
            
        # Find k: first index minimizing d_{k+1} - d_k
        gaps = np.diff(dists)
        min_gap = np.min(gaps)
        k = np.argmin(gaps)
        
        # D = (d_k + d_{k+1}) / 2
        D = (dists[k] + dists[k+1]) / 2.0
        
        if D < 0.1:
            return 0.0
            
        # Select 4 distances
        if k == 0:
            d_sel = dists[0:4]
        elif k == 17:
            d_sel = dists[15:19]
        else:
            d_sel = dists[k-1:k+3]
            
        # raw = mean of max(d/D, D/d)
        ratios = np.maximum(d_sel / D, D / d_sel)
        raw = np.mean(ratios)
        
        if raw == 0:
            return 0.0
            
        return min(1.0, 1.0 / raw)

    def total_score(polygon):
        """Calculate the minimum vertex score for the polygon."""
        norm_poly = normalize(polygon)
        if norm_poly is None:
            return -1.0 # Invalid
        
        if not is_convex(norm_poly):
            return -1.0 # Invalid
            
        scores = []
        for i in range(len(norm_poly)):
            v = norm_poly[i]
            s = vertex_score(v, norm_poly)
            scores.append(s)
            
        return min(scores)

    # Optimization Strategy
    # We optimize the radii.
    # Start with regular polygon
    best_radii = np.ones(n)
    best_score = total_score(get_polygon(best_radii))
    
    if best_score < 0:
        # Fallback to regular if normalization fails (shouldn't happen)
        return get_polygon(best_radii)

    # Use Nelder-Mead to maximize the score (minimize negative score)
    def objective(radii):
        if np.any(radii <= 0):
            return 100.0 # Penalty
        poly = get_polygon(radii)
        s = total_score(poly)
        if s < 0:
            return 100.0 + (0 - s) # Penalty for invalid
        return -s

    # Multiple restarts to escape local minima
    restarts = 5
    for r in range(restarts):
        if time.time() > deadline - 60.0:
            break
            
        # Initialize
        if r == 0:
            x0 = np.ones(n)
        else:
            # Random perturbation of best found so far
            x0 = best_radii * (1.0 + 0.2 * np.random.normal(0, 1, n))
            x0 = np.clip(x0, 0.5, 2.0)
            
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6})
            
            if res.fun < -best_score: # i.e. -score < -best_score -> score > best_score
                best_radii = res.x
                best_score = -res.fun
        except Exception:
            continue

    final_poly = get_polygon(best_radii)
    
    # Final sanity check
    if not is_convex(normalize(final_poly)):
        # Return regular polygon if optimization broke convexity
        return get_polygon(np.ones(n))
        
    return final_poly
