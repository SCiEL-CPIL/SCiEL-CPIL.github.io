# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of n points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 13.

    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    start_time = time.time()
    max_time = 620.0 # Leave 10s for overhead

    # Helper to compute score
    def score(pts):
        if np.any(np.isnan(pts)) or np.any(np.isinf(pts)):
            return -1e9
        try:
            hull = ConvexHull(pts)
            a_hull = hull.volume
            if a_hull < 1e-10:
                return -1e9
        except Exception:
            return -1e9
            
        min_area = np.inf
        # Unroll loops for speed? No, Python loops for 286 items are fine.
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    p1, p2, p3 = pts[i], pts[j], pts[k]
                    area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
                    if area < min_area:
                        min_area = area
        return min_area / a_hull

    # Objective function for minimizer
    def objective(x):
        pts = x.reshape(n, 2)
        return -score(pts)

    # Initialization: Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    points_reg = np.column_stack((np.cos(angles), np.sin(angles)))
    best_score = score(points_reg)
    best_points = points_reg.copy()

    # Try a few different initializations/perturbations
    rng = np.random.default_rng(seed=42)
    
    for trial in range(5):
        if time.time() - start_time > max_time - 5:
            break
            
        # Perturb the regular polygon
        # Larger perturbations in later trials to explore more?
        # Or just random restarts?
        # Random restarts from scratch might find better global structures.
        if trial == 0:
            x0 = points_reg.flatten() + rng.normal(0, 0.05, points_reg.flatten().shape)
        else:
            # Random points in unit circle
            x0 = rng.uniform(-1, 1, (n, 2)).flatten()
            # Ensure they are not collinear
            if score(x0.reshape(n,2)) < 0:
                continue

        # Define callback to check time
        def callback(xk):
            if time.time() - start_time > max_time - 2:
                raise RuntimeError("Time limit reached")
            # Update best if improved
            current_score = -objective(xk)
            nonlocal best_score, best_points
            if current_score > best_score:
                best_score = current_score
                best_points = xk.reshape(n, 2).copy()

        try:
            result = minimize(objective, x0, method='Nelder-Mead', 
                              options={'maxiter': 20000, 'xatol': 1e-7, 'fatol': 1e-9},
                              callback=callback)
            if result.fun < -best_score:
                best_score = -result.fun
                best_points = result.x.reshape(n, 2)
        except RuntimeError:
            pass
        except Exception:
            pass

    # Final validation
    final_score = score(best_points)
    if final_score <= 0:
        # Fallback
        return points_reg

    return best_points

# EVOLVE-BLOCK-END
