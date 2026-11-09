"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time

def _score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    """
    try:
        # Check for degenerate points
        if np.any(np.isnan(points)):
            return 0.0
            
        hull = ConvexHull(points)
        hull_area = hull.volume  # In 2D, volume is area
        if hull_area < 1e-10:
            return 0.0
        
        # Compute min triangle area
        # Indices for all triples i < j < k
        idx = np.indices((N, N, N)).reshape(3, -1)
        mask = (idx[0] < idx[1]) & (idx[1] < idx[2])
        i, j, k = idx[:, mask]
        
        pi = points[i]
        pj = points[j]
        pk = points[k]
        
        # Area = 0.5 * | (pj - pi) x (pk - pi) |
        v1 = pj - pi
        v2 = pk - pi
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = 0.5 * np.abs(cross)
        min_area = np.min(areas)
        
        return min_area / hull_area
    except Exception:
        return 0.0

def _negative_score(points_flat):
    """
    Objective function for minimization.
    """
    points = points_flat.reshape(N, 2)
    return -_score(points)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_points = None
    best_score = -np.inf
    
    # Initial guesses
    guesses = []
    rng = np.random.default_rng(42)
    
    # 1. Regular 14-gon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    guesses.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # 2. Points on a circle with radial and angular noise
    for _ in range(10):
        angles = np.linspace(0, 2*np.pi, N, endpoint=False) + rng.normal(0, 0.1, N)
        radii = 1.0 + rng.normal(0, 0.05, N)
        guesses.append(np.column_stack([np.cos(angles)*radii, np.sin(angles)*radii]))
        
    # 3. Random points in unit square
    for _ in range(5):
        guesses.append(rng.random((N, 2)))
        
    for guess in guesses:
        if time.time() > deadline - 5:
            break
            
        # Run local optimization
        try:
            # Powell method is robust for non-smooth functions
            res = minimize(_negative_score, guess.flatten(), method='Powell', 
                           options={'maxiter': 2000, 'ftol': 1e-9, 'xtol': 1e-9})
            
            current_score = -res.fun
            if current_score > best_score:
                best_score = current_score
                best_points = res.x.reshape(N, 2)
        except Exception:
            pass

    if best_points is None:
        # Fallback to regular polygon
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        
    return best_points
# EVOLVE-BLOCK-END
