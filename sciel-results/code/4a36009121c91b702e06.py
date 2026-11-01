"""Heilbronn convex regions n=14: maximize minimum triangle area / hull area."""
N = 14
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time
import itertools
import warnings

# EVOLVE-BLOCK-START
def _all_triangle_areas(pts):
    """Return array of areas of all triangles formed by triples of points."""
    n = len(pts)
    areas = np.empty((n, n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # area = 0.5 * |cross(pj - pi, pk - pi)|
                v1 = pts[j] - pts[i]
                v2 = pts[k] - pts[i]
                areas[i, j, k] = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
    return areas


def _score(pts):
    """Compute min triangle area / convex hull area."""
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume  # In 2D, volume is area
        if hull_area < 1e-12:
            return 0.0
        # Compute all triangle areas
        n = len(pts)
        min_area = float('inf')
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    v1 = pts[j] - pts[i]
                    v2 = pts[k] - pts[i]
                    area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                    if area < min_area:
                        min_area = area
        return min_area / hull_area
    except Exception:
        return 0.0


def _objective(x):
    """Negative score for minimization."""
    pts = np.array(x).reshape(N, 2)
    s = _score(pts)
    return -s


def construct():
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    # Strategy: Use concentric ring parameterization with optimization
    # Start with a good initial guess: regular polygon + inner points
    
    best_score = 0.0
    best_pts = None
    
    # Initial guess: regular 14-gon on unit circle
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    init_pts = np.column_stack((np.cos(angles), np.sin(angles)))
    
    # Try multiple initial configurations
    initial_configs = [
        # Regular 14-gon
        init_pts.copy(),
        # 7+7: two concentric heptagons
        np.vstack([
            np.column_stack((np.cos(np.linspace(0, 2*np.pi, 7, endpoint=False)),
                            np.sin(np.linspace(0, 2*np.pi, 7, endpoint=False)))),
            0.5 * np.column_stack((np.cos(np.linspace(0, 2*np.pi, 7, endpoint=False) + np.pi/7),
                                   np.sin(np.linspace(0, 2*np.pi, 7, endpoint=False) + np.pi/7)))
        ]),
        # 4+5+5: three rings
        np.vstack([
            np.array([[0.0, 0.0]]),
            np.column_stack((np.cos(np.linspace(0, 2*np.pi, 4, endpoint=False)),
                            np.sin(np.linspace(0, 2*np.pi, 4, endpoint=False)))),
            0.6 * np.column_stack((np.cos(np.linspace(0, 2*np.pi, 5, endpoint=False) + np.pi/5),
                                   np.sin(np.linspace(0, 2*np.pi, 5, endpoint=False) + np.pi/5)))
        ]),
        # 3+4+4+3: four rings
        np.vstack([
            np.array([[0.0, 0.0]]),
            np.column_stack((np.cos(np.linspace(0, 2*np.pi, 3, endpoint=False)),
                            np.sin(np.linspace(0, 2*np.pi, 3, endpoint=False)))),
            0.5 * np.column_stack((np.cos(np.linspace(0, 2*np.pi, 4, endpoint=False) + np.pi/4),
                                   np.sin(np.linspace(0, 2*np.pi, 4, endpoint=False) + np.pi/4))),
            0.8 * np.column_stack((np.cos(np.linspace(0, 2*np.pi, 3, endpoint=False) + np.pi/3),
                                   np.sin(np.linspace(0, 2*np.pi, 3, endpoint=False) + np.pi/3)))
        ]),
    ]
    
    # Add some random perturbations
    rng = np.random.default_rng(42)
    for _ in range(3):
        random_pts = rng.random((N, 2))
        # Normalize to unit area
        try:
            hull = ConvexHull(random_pts)
            area = hull.volume
            if area > 1e-6:
                centroid = random_pts.mean(axis=0)
                centered = random_pts - centroid
                scaled = centered / np.sqrt(area)
                initial_configs.append(scaled)
        except Exception:
            pass
    
    # Optimize each initial configuration
    for config in initial_configs:
        if time.time() > deadline - 10:
            break
        
        x0 = config.flatten()
        
        # Use Nelder-Mead for robustness with nonsmooth objective
        try:
            result = minimize(_objective, x0, method='Nelder-Mead',
                            options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
            if result.fun < 0:
                score = -result.fun
                if score > best_score:
                    best_score = score
                    best_pts = result.x.reshape(N, 2)
        except Exception:
            pass
    
    # If we still have time, do a second round of optimization with more iterations
    if best_pts is not None and time.time() < deadline - 20:
        x0 = best_pts.flatten()
        try:
            result = minimize(_objective, x0, method='Nelder-Mead',
                            options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12})
            if result.fun < 0:
                score = -result.fun
                if score > best_score:
                    best_score = score
                    best_pts = result.x.reshape(N, 2)
        except Exception:
            pass
    
    # If no valid result found, fall back to regular polygon
    if best_pts is None:
        best_pts = init_pts
    
    return best_pts
# EVOLVE-BLOCK-END
