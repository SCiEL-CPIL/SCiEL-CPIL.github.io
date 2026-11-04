"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 130

# EVOLVE-BLOCK-START
def _fast_covering_radius(points):
    """
    Approximate covering radius using a grid of points on the sphere.
    This is much faster than ConvexHull for use in optimization loops.
    """
    n_test = 20000
    i = np.arange(n_test) + 0.5
    z = 1.0 - 2.0 * i / n_test
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    test_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    dots = np.dot(test_pts, points.T) # shape (n_test, N)
    dots = np.clip(dots, -1.0, 1.0)
    angles = np.arccos(dots)
    min_dists = np.min(angles, axis=1)
    return np.max(min_dists)

def construct():
    """Construct 130 points on the unit sphere minimizing covering radius."""
    start_time = time.time()
    deadline = 140.0 # Leave 10s buffer for 150s limit
    
    # Golden-angle (Fibonacci) spiral points on the unit sphere
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Normalize
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    points = points / norms
    
    best_points = points
    best_score = _fast_covering_radius(best_points)
    
    # Optimization loop
    def objective(x):
        pts = x.reshape(N, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        pts = pts / norms
        return _fast_covering_radius(pts)
    
    current_x = points.flatten()
    
    # Set options for Powell
    options = {'maxiter': 1000, 'ftol': 1e-12, 'xtol': 1e-12}
    
    while time.time() - start_time < deadline:
        try:
            res = minimize(objective, current_x, method='Powell', options=options)
            if res.fun < best_score:
                best_score = res.fun
                best_points = res.x.reshape(N, 3)
                # Normalize best points
                norms = np.linalg.norm(best_points, axis=1, keepdims=True)
                norms[norms < 1e-10] = 1.0
                best_points = best_points / norms
                current_x = best_points.flatten()
            else:
                # If no improvement, try a small perturbation restart
                if time.time() - start_time > deadline - 20:
                    break
                # Add small noise to escape local minima
                noise = np.random.normal(0, 0.01, current_x.shape)
                current_x = current_x + noise
        except Exception:
            break
            
    # Final normalization
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    best_points = best_points / norms
    
    return best_points
# EVOLVE-BLOCK-END
