"""Optimized program for Heilbronn convex regions n=14."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 14
_DEADLINE = 110.0  # safety margin before 120s kill
_START = None


def _min_triangle_area(points):
    """Compute minimum triangle area over all triples (vectorized)."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                area = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
                if area < min_area:
                    min_area = area
    return min_area


def _convex_hull_area(points):
    """Compute area of convex hull."""
    try:
        hull = ConvexHull(points)
        return hull.volume  # For 2D, volume is area
    except Exception:
        return 0.0


def _score(points):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area <= 1e-12:
        return 0.0
    min_tri = _min_triangle_area(points)
    return min_tri / hull_area


def _objective_flat(flat):
    """Objective for minimization: negative score."""
    points = flat.reshape(N, 2)
    return -_score(points)


def _refine(points):
    """Refine a point configuration using L-BFGS-B."""
    flat = points.flatten()
    try:
        result = minimize(
            _objective_flat,
            flat,
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        refined = result.x.reshape(N, 2)
        return refined
    except Exception:
        return points


def _grid_initialization(spacing=0.1):
    """Create a grid of points and select N of them."""
    points = []
    x = 0
    while x < 1.0:
        y = 0
        while y < 1.0:
            points.append([x, y])
            y += spacing
        x += spacing
    points = np.array(points)
    # Randomly select N points
    idx = np.random.choice(len(points), N, replace=False)
    return points[idx]


def _circle_initialization():
    """Points on a circle with some jitter."""
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    radius = 0.5
    cx, cy = 0.5, 0.5
    points = np.column_stack([
        cx + radius * np.cos(angles),
        cy + radius * np.sin(angles)
    ])
    # Add small jitter
    points += 0.01 * np.random.randn(N, 2)
    return points


def _random_initialization():
    """Random points in unit square."""
    return np.random.rand(N, 2)


def _interleaved_circle():
    """Two interleaved circles."""
    angles1 = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    angles2 = np.linspace(0, 2 * np.pi, 7, endpoint=False) + np.pi / 7
    r1, r2 = 0.35, 0.6
    cx, cy = 0.5, 0.5
    p1 = np.column_stack([cx + r1 * np.cos(angles1), cy + r1 * np.sin(angles1)])
    p2 = np.column_stack([cx + r2 * np.cos(angles2), cy + r2 * np.sin(angles2)])
    points = np.vstack([p1, p2])
    points += 0.02 * np.random.randn(N, 2)
    return points


def construct():
    global _START
    if _START is None:
        _START = time.time()
    
    np.random.seed(42)
    
    best_points = None
    best_score = -np.inf
    
    # Generate diverse initializations
    initializations = []
    
    # Grid-based
    for spacing in [0.08, 0.1, 0.12, 0.15]:
        initializations.append(_grid_initialization(spacing))
    
    # Circle-based
    initializations.append(_circle_initialization())
    initializations.append(_interleaved_circle())
    
    # Random
    for _ in range(5):
        initializations.append(_random_initialization())
    
    # Evaluate initializations
    for init in initializations:
        score = _score(init)
        if score > best_score:
            best_score = score
            best_points = init.copy()
    
    # Refine the best ones with L-BFGS
    # Sort by score and refine top candidates
    scored_inits = []
    for init in initializations:
        scored_inits.append((_score(init), init))
    scored_inits.sort(key=lambda x: -x[0])
    
    # Refine top 5 initializations
    for score, init in scored_inits[:5]:
        if time.time() - _START > _DEADLINE:
            break
        
        refined = _refine(init)
        refined_score = _score(refined)
        if refined_score > best_score:
            best_score = refined_score
            best_points = refined.copy()
    
    # Additional refinement rounds on the best
    for _ in range(3):
        if time.time() - _START > _DEADLINE:
            break
        
        if best_points is not None:
            # Perturb and refine
            perturbed = best_points + 0.05 * np.random.randn(N, 2)
            refined = _refine(perturbed)
            refined_score = _score(refined)
            if refined_score > best_score:
                best_score = refined_score
                best_points = refined.copy()
    
    # Final refinement of the best
    if time.time() - _START < _DEADLINE and best_points is not None:
        final_refined = _refine(best_points)
        final_score = _score(final_refined)
        if final_score > best_score:
            best_score = final_score
            best_points = final_refined.copy()
    
    if best_points is None:
        best_points = _random_initialization()
    
    return best_points
