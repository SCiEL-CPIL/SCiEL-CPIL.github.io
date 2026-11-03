"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

# Pre-compute triangle indices for efficiency
def _get_triangle_indices(n):
    idx1 = []
    idx2 = []
    idx3 = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                idx1.append(i)
                idx2.append(j)
                idx3.append(k)
    return np.array(idx1), np.array(idx2), np.array(idx3)

TRI_IDX1, TRI_IDX2, TRI_IDX3 = _get_triangle_indices(N)

def _score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    """
    points = np.asarray(points, dtype=np.float64).reshape(N, 2)
    
    # Check for degenerate hull
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-9:
            return 0.0
    except Exception:
        return 0.0

    # Calculate areas of all triangles
    p1 = points[TRI_IDX1]
    p2 = points[TRI_IDX2]
    p3 = points[TRI_IDX3]
    
    # Area = 0.5 * | (p2-p1) x (p3-p1) |
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    
    return min_area / hull_area

def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120s limit, keep 30s margin
    
    # Initial guess: Regular polygon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    radius = 1.0
    x0 = np.column_stack((radius * np.cos(angles), radius * np.sin(angles))).flatten()
    
    best_points = x0.copy()
    best_score = _score(best_points)
    
    # Objective function for minimizer (we maximize score, so minimize -score)
    def objective(x):
        # Optional: normalize to unit area to help optimizer? 
        # The score is scale invariant, but the optimizer might drift in scale.
        # Let's normalize the points to have a fixed norm or hull area inside the objective?
        # No, it's better to let the optimizer find the shape. 
        # However, if the points expand to infinity, the ratio stays same but numerical issues arise.
        # If they shrink to 0, same.
        # We can fix the scale by normalizing the sum of squared distances to 1.
        pts = x.reshape(N, 2)
        norm = np.sum(pts**2)
        if norm > 1e-9:
            pts = pts / np.sqrt(norm)
        else:
            pts = pts + 1e-6 # avoid zero
        return -_score(pts)

    # Use Nelder-Mead for robustness against non-differentiability of min()
    # Options can be tuned, but defaults are often okay for small N
    options = {
        'maxiter': 5000,
        'xatol': 1e-8,
        'fatol': 1e-8
    }
    
    try:
        res = minimize(objective, x0, method='Nelder-Mead', options=options)
        
        # Check if result is better
        res_pts = res.x.reshape(N, 2)
        # Normalize res_pts to match the scaling used in objective? 
        # The objective normalized internally, but returned score based on that.
        # The returned x is in the "normalized space" if we modified it? 
        # No, minimize changes x. If I normalize inside objective, the gradient is w.r.t x, 
        # but the landscape is flattened in radial direction.
        # It's safer to just evaluate the raw x0 and res.x.
        
        # Actually, since the objective is scale invariant, the optimizer might wander in scale.
        # Let's re-normalize the result to a standard scale (e.g., radius 1) before scoring, 
        # or just score it as is. The score function handles any scale.
        
        res_score = _score(res.x)
        if res_score > best_score:
            best_score = res_score
            best_points = res.x.copy()
            
    except Exception:
        pass

    # If we have time, try a few random restarts
    # The regular polygon is a strong baseline. Random restarts might find local minima that are worse.
    # But maybe a different topology? Unlikely for max-min triangle area.
    # Let's try one more optimization from a slightly perturbed regular polygon if time allows.
    
    if time.time() < deadline - 10:
        # Perturb the best points
        noise = np.random.normal(0, 0.1, best_points.shape)
        x_start = best_points + noise
        try:
            res2 = minimize(objective, x_start, method='Nelder-Mead', options=options)
            res2_score = _score(res2.x)
            if res2_score > best_score:
                best_score = res2_score
                best_points = res2.x.copy()
        except Exception:
            pass

    # Ensure the output is a numpy array of shape (N, 2)
    return best_points.reshape(N, 2)

# EVOLVE-BLOCK-END
