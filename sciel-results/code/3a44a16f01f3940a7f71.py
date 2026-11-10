# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull


def _triangle_areas(points):
    """Compute areas of all triangles formed by triples of points."""
    n = len(points)
    areas = np.zeros((n, n, n))
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                area = abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0
                areas[i, j, k] = area
                areas[j, i, k] = area
                areas[k, j, i] = area
    return areas


def _score(points):
    """Compute A_min / A_hull ratio."""
    try:
        hull = ConvexHull(points)
        a_hull = hull.volume
    except Exception:
        return 0.0
    if a_hull <= 1e-15:
        return 0.0
    n = len(points)
    min_area = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                area = abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0
                if area < min_area:
                    min_area = area
    if min_area <= 0:
        return 0.0
    return min_area / a_hull


def _nelder_mead(f, x0, maxiter=10000, tol=1e-10, alpha=1.0, gamma=2.0, rho=0.5, sigma=2.0):
    """Simple Nelder-Mead implementation."""
    n = len(x0)
    # Initial simplex
    simplex = [np.array(x0, dtype=float)]
    for i in range(n):
        pt = np.array(x0, dtype=float)
        pt[i] += 0.05 * (1.0 if x0[i] == 0 else x0[i])
        simplex.append(pt)
    
    values = [f(pt) for pt in simplex]
    
    for it in range(maxiter):
        # Sort simplex
        idx = np.argsort(values)
        simplex = [simplex[i] for i in idx]
        values = [values[i] for i in idx]
        
        # Check convergence
        if abs(values[-1] - values[0]) < tol:
            break
        
        # Centroid of all but worst
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        x_r = centroid + alpha * (centroid - simplex[-1])
        f_r = f(x_r)
        
        if f_r < values[0]:
            # Expansion
            x_e = centroid + gamma * (x_r - centroid)
            f_e = f(x_e)
            if f_e < f_r:
                simplex[-1] = x_e
                values[-1] = f_e
            else:
                simplex[-1] = x_r
                values[-1] = f_r
        elif f_r < values[-2]:
            simplex[-1] = x_r
            values[-1] = f_r
        else:
            # Contraction
            x_c = centroid + rho * (simplex[-1] - centroid)
            f_c = f(x_c)
            if f_c < values[-1]:
                simplex[-1] = x_c
                values[-1] = f_c
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    values[i] = f(simplex[i])
    
    # Return best point
    idx = np.argmin(values)
    return simplex[idx], values[idx]


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of n points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 13.
    
    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    deadline = time.time() + 600
    
    # Start with regular 13-gon on unit circle
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x0 = np.concatenate([np.cos(angles), np.sin(angles)])
    
    best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    best_score = _score(best_points)
    
    # Optimize full 26D coordinates
    def objective(coords):
        pts = coords.reshape(n, 2)
        return _score(pts)
    
    # Run Nelder-Mead with multiple restarts
    rng = np.random.default_rng(seed=123)
    
    # First, optimize from the regular polygon
    if time.time() < deadline - 30:
        result, val = _nelder_mead(objective, x0, maxiter=5000, tol=1e-12)
        if val > best_score:
            best_score = val
            best_points = result.reshape(n, 2)
    
    # Try a few different initializations
    for restart in range(5):
        if time.time() >= deadline - 60:
            break
        
        # Perturb the best solution
        perturbation = rng.normal(0, 0.1, size=(n, 2))
        start = best_points + perturbation
        start_flat = start.flatten()
        
        result, val = _nelder_mead(objective, start_flat, maxiter=3000, tol=1e-12)
        if val > best_score:
            best_score = val
            best_points = result.reshape(n, 2)
    
    # Final polish
    if time.time() < deadline - 10:
        result, val = _nelder_mead(objective, best_points.flatten(), maxiter=2000, tol=1e-14)
        if val > best_score:
            best_score = val
            best_points = result.reshape(n, 2)
    
    return best_points


# EVOLVE-BLOCK-END
