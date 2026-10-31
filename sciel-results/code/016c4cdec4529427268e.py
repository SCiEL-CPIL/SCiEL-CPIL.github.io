"""Heilbronn convex regions: place 13 points to maximize min triangle area / hull area."""
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, differential_evolution

N = 13
DEADLINE = 115.0  # seconds, safety margin before 120s kill

def _compute_min_triangle_area(points):
    """Compute minimum triangle area among all triples."""
    n = points.shape[0]
    min_area = float('inf')
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
                if area < min_area:
                    min_area = area
    return min_area

def _compute_hull_area(points):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume  # For 2D, volume is area
    except Exception:
        return 0.0

def _score(points):
    """Compute the score: min triangle area / hull area."""
    hull_area = _compute_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_tri = _compute_min_triangle_area(points)
    return min_tri / hull_area

def _objective(params):
    """Objective function for optimization (minimize negative score)."""
    points = params.reshape(N, 2)
    s = _score(points)
    return -s

def _make_grid_4x4_minus_corner():
    """Create a 4x4 grid minus one corner point, centered."""
    # 4x4 grid from -1.5 to 1.5
    coords = np.array([[i - 1.5 for i in range(4)] for j in range(4) for i in range(4)])
    # Remove one corner (top-right in our indexing: index 15)
    # Actually let's remove index 0 (bottom-left) for variety, or just pick one
    # Let's use indices 0-14 (remove the last one, index 15)
    return coords[:13]

def construct():
    start_time = time.time()
    
    # Start with a structured grid
    base_grid = _make_grid_4x4_minus_corner()
    
    # Initial affine transform: identity
    # We'll optimize 4 parameters: a, b, c, d for matrix [[a, b], [c, d]]
    # Applied to the base grid
    
    best_score = 0.0
    best_points = base_grid.copy()
    
    def _affine_score(params):
        """Score given affine transform parameters [a, b, c, d]."""
        a, b, c, d = params
        M = np.array([[a, b], [c, d]])
        points = base_grid @ M.T
        return _score(points)
    
    def _affine_obj(params):
        return -_affine_score(params)
    
    # First, do a coarse grid search over affine parameters
    # a, d in [0.5, 2.0], b, c in [-1.0, 1.0]
    a_vals = np.linspace(0.5, 2.0, 8)
    d_vals = np.linspace(0.5, 2.0, 8)
    b_vals = np.linspace(-1.0, 1.0, 8)
    c_vals = np.linspace(-1.0, 1.0, 8)
    
    for a in a_vals:
        if time.time() - start_time > DEADLINE:
            break
        for d in d_vals:
            if time.time() - start_time > DEADLINE:
                break
            for b in b_vals:
                if time.time() - start_time > DEADLINE:
                    break
                for c in c_vals:
                    if time.time() - start_time > DEADLINE:
                        break
                    s = _affine_score([a, b, c, d])
                    if s > best_score:
                        best_score = s
                        M = np.array([[a, b], [c, d]])
                        best_points = base_grid @ M.T
    
    # Now refine with local optimization from the best point found
    if time.time() - start_time < DEADLINE - 30:
        # Get current best affine params
        # We need to track which params gave the best score
        # Let's just use the best_points and do a full local search
        # Actually, let's re-do: track best params
        pass
    
    # Let me restructure: track best params
    best_score = 0.0
    best_params = np.array([1.0, 0.0, 0.0, 1.0])
    best_points = base_grid.copy()
    
    a_vals = np.linspace(0.5, 2.0, 8)
    d_vals = np.linspace(0.5, 2.0, 8)
    b_vals = np.linspace(-1.0, 1.0, 8)
    c_vals = np.linspace(-1.0, 1.0, 8)
    
    for a in a_vals:
        if time.time() - start_time > DEADLINE:
            break
        for d in d_vals:
            if time.time() - start_time > DEADLINE:
                break
            for b in b_vals:
                if time.time() - start_time > DEADLINE:
                    break
                for c in c_vals:
                    if time.time() - start_time > DEADLINE:
                        break
                    s = _affine_score([a, b, c, d])
                    if s > best_score:
                        best_score = s
                        best_params = np.array([a, b, c, d])
                        M = np.array([[a, b], [c, d]])
                        best_points = base_grid @ M.T
    
    # Local refinement using scipy minimize
    if time.time() - start_time < DEADLINE - 20:
        try:
            res = minimize(
                _affine_obj,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            s = _affine_score(res.x)
            if s > best_score:
                best_score = s
                M = np.array([[res.x[0], res.x[1]], [res.x[2], res.x[3]]])
                best_points = base_grid @ M.T
        except Exception:
            pass
    
    # Also try a few different base grids
    if time.time() - start_time < DEADLINE - 15:
        # Try a different grid: 3x5 arrangement or other patterns
        # Grid 1: 4x4 minus corner (already done)
        # Grid 2: centered hexagonal-like pattern
        hex_points = np.array([
            [0, 0],
            [1, 0], [0.5, 0.866], [-0.5, 0.866], [-1, 0],
            [-0.5, -0.866], [0.5, -0.866],
            [1.5, 0.866], [1.5, -0.866],
            [2.0, 0], [1.0, 1.732], [1.0, -1.732],
            [0, 1.732]
        ])
        # Center it
        hex_points = hex_points - hex_points.mean(axis=0)
        
        best_hex_score = 0.0
        a_vals2 = np.linspace(0.5, 2.0, 6)
        d_vals2 = np.linspace(0.5, 2.0, 6)
        b_vals2 = np.linspace(-1.0, 1.0, 6)
        c_vals2 = np.linspace(-1.0, 1.0, 6)
        
        for a in a_vals2:
            if time.time() - start_time > DEADLINE:
                break
            for d in d_vals2:
                if time.time() - start_time > DEADLINE:
                    break
                for b in b_vals2:
                    if time.time() - start_time > DEADLINE:
                        break
                    for c in c_vals2:
                        if time.time() - start_time > DEADLINE:
                            break
                        M = np.array([[a, b], [c, d]])
                        pts = hex_points @ M.T
                        s = _score(pts)
                        if s > best_hex_score:
                            best_hex_score = s
                            best_hex_points = pts
        
        if best_hex_score > best_score:
            best_score = best_hex_score
            best_points = best_hex_points
    
    # Final local polish on best_points using full 26D search with limited iterations
    if time.time() - start_time < DEADLINE - 10:
        try:
            res = minimize(
                _objective,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 100, 'xatol': 1e-5, 'fatol': 1e-7}
            )
            s = _score(res.x.reshape(N, 2))
            if s > best_score:
                best_score = s
                best_points = res.x.reshape(N, 2)
        except Exception:
            pass
    
    # Check deadline
    if time.time() - start_time > DEADLINE - 5:
        return best_points
    
    return best_points
