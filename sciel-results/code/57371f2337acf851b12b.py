"""Heilbronn triangles: place 11 points in unit equilateral triangle to maximize min triangle area."""
import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

N = 11
SQRT3 = np.sqrt(3.0)
H = SQRT3 / 2.0  # height of unit equilateral triangle


def _clamp_to_triangle(pts):
    """Clamp points to lie inside the equilateral triangle (0,0), (1,0), (0.5, H)."""
    pts = np.array(pts, dtype=float).reshape(-1, 2)
    x, y = pts[:, 0], pts[:, 1]
    # Clamp y to [0, H]
    y = np.clip(y, 0.0, H)
    # For each y, x must satisfy: 0 <= x <= y/H * 1 (left edge) and x >= 1 - y/H (right edge)
    # Left edge: from (0,0) to (0.5, H): x >= (y / H) * 0.5 = y / (2*H) ... wait
    # Actually the triangle has vertices (0,0), (1,0), (0.5, H)
    # Left edge: (0,0) to (0.5, H): x = 0.5 * y / H = y / SQRT3
    # Right edge: (1,0) to (0.5, H): x = 1 - 0.5 * y / H = 1 - y / SQRT3
    # So constraint: y/SQRT3 <= x <= 1 - y/SQRT3
    x_min = y / SQRT3
    x_max = 1.0 - y / SQRT3
    # Ensure x_min <= x_max (should be for y in [0, H])
    x = np.clip(x, x_min, x_max)
    return np.column_stack([x, y])


def _min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by 3 of the N points."""
    n = len(pts)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = pts[i]
                b = pts[j]
                c = pts[k]
                area = 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]))
                if area < min_area:
                    min_area = area
    return min_area


def _objective_flat(flat):
    """Objective function for optimization: negative of min area (we minimize)."""
    pts = _clamp_to_triangle(flat.reshape(N, 2))
    return -_min_triangle_area(pts)


def construct():
    deadline = time.time() + 130.0  # safety margin before 150s kill
    
    best_score = 0.0
    best_pts = None
    
    # Initialize with a reasonable spread
    rng = np.random.default_rng(42)
    
    # Strategy 1: Differential Evolution for global search
    # Variables: 2*N coordinates
    bounds = [(0.0, 1.0)] * (2 * N)
    
    # Use DE with a moderate population
    de_result = differential_evolution(
        _objective_flat,
        bounds=bounds,
        seed=42,
        maxiter=50,
        popsize=15,
        tol=1e-4,
        mutation=(0.3, 1.0),
        recombination=0.7,
        polish=False,
        init='sobol',
        workers=-1
    )
    
    if time.time() > deadline:
        # Return whatever we have
        if best_pts is not None:
            return best_pts
        pts = _clamp_to_triangle(de_result.x.reshape(N, 2))
        return pts
    
    de_pts = _clamp_to_triangle(de_result.x.reshape(N, 2))
    de_score = _min_triangle_area(de_pts)
    
    if de_score > best_score:
        best_score = de_score
        best_pts = de_pts.copy()
    
    # Strategy 2: Local polish with Nelder-Mead from multiple starting points
    starting_points = [de_pts.copy()]
    
    # Add some heuristic starting points
    # Symmetric-ish arrangement
    heuristic_pts = []
    # Place points in a grid-like pattern inside triangle
    for row in range(4):
        for col in range(4):
            if len(heuristic_pts) >= N:
                break
            # Map grid to triangle
            t = (row + 0.5) / 4.0
            s = (col + 0.5) / 4.0
            x = s * (1.0 - t) + 0.5 * t
            y = t * H
            heuristic_pts.append([x, y])
    heuristic_pts = np.array(heuristic_pts[:N])
    heuristic_pts = _clamp_to_triangle(heuristic_pts)
    starting_points.append(heuristic_pts.copy())
    
    # Random starts
    for _ in range(5):
        rand_pts = rng.random((N, 2))
        rand_pts[:, 0] *= 1.0
        rand_pts[:, 1] *= H
        rand_pts = _clamp_to_triangle(rand_pts)
        starting_points.append(rand_pts.copy())
    
    for start in starting_points:
        if time.time() > deadline:
            break
        
        try:
            result = minimize(
                _objective_flat,
                start.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            nm_pts = _clamp_to_triangle(result.x.reshape(N, 2))
            nm_score = _min_triangle_area(nm_pts)
            
            if nm_score > best_score:
                best_score = nm_score
                best_pts = nm_pts.copy()
        except Exception:
            continue
    
    # Strategy 3: More DE runs with different seeds if time permits
    if time.time() < deadline - 30:
        for seed in [123, 456, 789]:
            if time.time() > deadline - 15:
                break
            try:
                de_result2 = differential_evolution(
                    _objective_flat,
                    bounds=bounds,
                    seed=seed,
                    maxiter=40,
                    popsize=12,
                    tol=1e-4,
                    mutation=(0.4, 1.0),
                    recombination=0.8,
                    polish=False,
                    init='sobol',
                    workers=-1
                )
                de_pts2 = _clamp_to_triangle(de_result2.x.reshape(N, 2))
                de_score2 = _min_triangle_area(de_pts2)
                
                if de_score2 > best_score:
                    best_score = de_score2
                    best_pts = de_pts2.copy()
                    
                    # Polish
                    result2 = minimize(
                        _objective_flat,
                        de_pts2.flatten(),
                        method='Nelder-Mead',
                        options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10}
                    )
                    nm_pts2 = _clamp_to_triangle(result2.x.reshape(N, 2))
                    nm_score2 = _min_triangle_area(nm_pts2)
                    if nm_score2 > best_score:
                        best_score = nm_score2
                        best_pts = nm_pts2.copy()
            except Exception:
                continue
    
    if best_pts is None:
        # Fallback: return the DE result
        best_pts = _clamp_to_triangle(de_result.x.reshape(N, 2))
    
    return best_pts
