"""Heilbronn triangles: maximize minimum triangle area for 11 points in unit equilateral triangle."""
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
import itertools

N = 11
DEADLINE = 130  # seconds, safety margin before 150s kill

# Equilateral triangle vertices
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, np.sqrt(3) / 2])
TRI_AREA = np.sqrt(3) / 4  # area of the unit equilateral triangle


def _project_to_triangle(pts):
    """Project points to stay inside the triangle with small margin."""
    pts = np.array(pts, dtype=float)
    # Clip to triangle boundaries with small margin
    margin = 0.005
    for i in range(len(pts)):
        x, y = pts[i]
        # Check against each edge; if outside, project back
        # Edge AB: y >= 0
        if y < margin:
            y = margin
        # Edge AC: y <= sqrt(3)*x
        if y > np.sqrt(3) * x - margin:
            # Project onto line y = sqrt(3)*x
            t = x / (1 + 3)  # not quite right, let's do proper projection
            # Simpler: just clip
            y = min(y, np.sqrt(3) * x - margin)
        # Edge BC: y <= -sqrt(3)*(x-1) = sqrt(3)*(1-x)
        if y > np.sqrt(3) * (1 - x) - margin:
            y = min(y, np.sqrt(3) * (1 - x) - margin)
        # Also keep x in [0, 1]
        x = max(margin, min(1 - margin, x))
        pts[i] = [x, y]
    return pts


def _min_triangle_area_normalized(pts):
    """Compute min triangle area / TRI_AREA for N points."""
    pts = np.array(pts, dtype=float)
    min_area = float('inf')
    for i, j, k in itertools.combinations(range(N), 3):
        p1, p2, p3 = pts[i], pts[j], pts[k]
        # Cross product magnitude / 2 = area
        cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
        area = 0.5 * abs(cross)
        if area < min_area:
            min_area = area
    return min_area / TRI_AREA


def _objective(x):
    """Objective to minimize: negative of min normalized triangle area."""
    pts = x.reshape(N, 2)
    # Check if all points are inside triangle
    for i in range(N):
        px, py = pts[i]
        if py < -1e-6:
            return 1e10
        if py > np.sqrt(3) * px + 1e-6:
            return 1e10
        if py > np.sqrt(3) * (1 - px) + 1e-6:
            return 1e10
    return -_min_triangle_area_normalized(pts)


def construct():
    start_time = time.time()
    
    # Define bounds for each coordinate
    # x in [0, 1], y in [0, sqrt(3)/2]
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, np.sqrt(3) / 2))
    
    # Differential Evolution for global search
    best_result = None
    best_score = -float('inf')
    
    try:
        de_result = differential_evolution(
            _objective,
            bounds=bounds,
            seed=42,
            maxiter=50,
            popsize=15,
            tol=1e-8,
            mutation=(0.5, 1.0),
            recombination=0.7,
            polish=False,  # We'll do our own local refinement
            init='latinhypercube',
            update_interval=1,
            callback=None
        )
        
        elapsed = time.time() - start_time
        if de_result.success and de_result.fun < 1e9:
            pts = de_result.x.reshape(N, 2)
            pts = _project_to_triangle(pts)
            score = _min_triangle_area_normalized(pts)
            best_score = score
            best_result = pts.copy()
        
        # Local refinement with Nelder-Mead if time permits
        if best_result is not None and (time.time() - start_time) < DEADLINE - 30:
            nm_result = minimize(
                _objective,
                best_result.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            pts = nm_result.x.reshape(N, 2)
            pts = _project_to_triangle(pts)
            score = _min_triangle_area_normalized(pts)
            if score > best_score:
                best_score = score
                best_result = pts.copy()
        
    except Exception:
        pass
    
    # If DE didn't work well, try a fallback with random restarts
    if best_result is None or (time.time() - start_time) < DEADLINE - 20:
        rng = np.random.RandomState(123)
        for restart in range(10):
            if time.time() - start_time > DEADLINE - 15:
                break
            # Random initial points
            pts = np.array([
                [rng.uniform(0.05, 0.95), rng.uniform(0.05, np.sqrt(3)/2 * 0.9)]
                for _ in range(N)
            ])
            # Project to triangle
            pts = _project_to_triangle(pts)
            
            # Local refinement
            try:
                nm_result = minimize(
                    _objective,
                    pts.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12}
                )
                pts2 = nm_result.x.reshape(N, 2)
                pts2 = _project_to_triangle(pts2)
                score = _min_triangle_area_normalized(pts2)
                if score > best_score:
                    best_score = score
                    best_result = pts2.copy()
            except Exception:
                continue
    
    # Final fallback: simple random if nothing worked
    if best_result is None:
        rng = np.random.RandomState(0)
        best_result = np.array([
            [rng.uniform(0.01, 0.99), rng.uniform(0.01, np.sqrt(3)/2 * 0.98)]
            for _ in range(N)
        ])
        best_result = _project_to_triangle(best_result)
    
    return best_result
