import numpy as np
from scipy.optimize import differential_evolution, minimize
import itertools

# ... existing helper functions ...

def _min_triangle_area_fast(points):
    n = len(points)
    # Pre-compute indices for all triples
    # This can be cached, but n is fixed at 11.
    # idx = np.array(list(itertools.combinations(range(n), 3)))
    # For speed, we can just use the pre-computed indices if we store them globally or pass them.
    # But construct() is called once.
    
    # To be safe and fast, let's compute indices inside or use a global.
    # Since N=11 is fixed, we can compute it once.
    
    p1 = points[idx_global[:, 0]]
    p2 = points[idx_global[:, 1]]
    p3 = points[idx_global[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

# Global indices
N = 11
idx_global = np.array(list(itertools.combinations(range(N), 3)))

def _objective_de(points_flat):
    points = points_flat.reshape(-1, 2)
    # Check validity
    # Penalty for being outside
    penalty = 0.0
    # y >= 0
    if np.any(points[:, 1] < 0):
        penalty += np.sum((points[points[:, 1] < 0, 1]) ** 2)
    # sqrt(3)*x + y <= sqrt(3)
    v2 = np.sqrt(3) * points[:, 0] + points[:, 1] - np.sqrt(3)
    if np.any(v2 > 0):
        penalty += np.sum(v2[v2 > 0] ** 2)
    # -sqrt(3)*x + y <= 0
    v3 = -np.sqrt(3) * points[:, 0] + points[:, 1]
    if np.any(v3 > 0):
        penalty += np.sum(v3[v3 > 0] ** 2)
        
    if penalty > 1e-6:
        return 1e5 + penalty
        
    return -_min_triangle_area_fast(points)

def construct():
    # Bounds for DE
    # x in [0, 1], y in [0, sqrt(3)/2]
    bounds = [(0, 1), (0, np.sqrt(3)/2)] * N
    
    # Initial population? DE generates its own.
    # But we can use a specific seed or strategy.
    
    # Run DE
    res = differential_evolution(
        _objective_de,
        bounds=bounds,
        seed=42,
        maxiter=100,
        tol=1e-8,
        polish=True # This will run L-BFGS-B on the best result
    )
    
    best_points = res.x.reshape(-1, 2)
    
    # Optional: Further polish with Nelder-Mead if needed
    # res.x is already polished by L-BFGS-B if polish=True
    
    return best_points
