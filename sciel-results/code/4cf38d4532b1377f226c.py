"""Heilbronn triangles: place 11 points in an equilateral triangle to maximize the minimum triangle area."""
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

N = 11
# Vertices of the equilateral triangle
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
TRI_AREA = np.sqrt(3) / 4


def _min_triangle_area(points):
    """Compute the minimum area among all triangles formed by triples of points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
                if area < min_area:
                    min_area = area
    return min_area


def _is_inside_triangle(p):
    """Check if point p is inside the triangle (0,0), (1,0), (0.5, sqrt(3)/2)."""
    x, y = p
    if y < 0:
        return False
    if x < 0:
        return False
    if x > 1:
        return False
    # Check below the top edges
    # Left edge: from (0,0) to (0.5, sqrt(3)/2): y = sqrt(3)*x
    # Right edge: from (1,0) to (0.5, sqrt(3)/2): y = -sqrt(3)*(x-1)
    if y > np.sqrt(3) * x + 1e-9:
        return False
    if y > -np.sqrt(3) * (x - 1) + 1e-9:
        return False
    return True


def _project_to_triangle(p):
    """Project point p to the nearest point inside the triangle."""
    if _is_inside_triangle(p):
        return p
    # Use barycentric coordinates or simple projection
    # For now, clip to bounding box and check
    x = np.clip(p[0], 0, 1)
    y = np.clip(p[1], 0, np.sqrt(3) / 2)
    if _is_inside_triangle([x, y]):
        return np.array([x, y])
    # Otherwise, find closest point on boundary
    # This is a fallback; for DE bounds, we should stay inside
    return np.array([0.5, np.sqrt(3) / 4])  # centroid


def _objective(flat_params):
    """Objective function for optimization: minimize negative min area."""
    points = flat_params.reshape((N, 2))
    # Project to triangle
    points = np.array([_project_to_triangle(p) for p in points])
    min_area = _min_triangle_area(points)
    return -min_area


def _objective_constrained(flat_params):
    """Objective with penalty for points outside triangle."""
    points = flat_params.reshape((N, 2))
    penalty = 0.0
    for p in points:
        if not _is_inside_triangle(p):
            # Compute distance to triangle boundary
            # Simple penalty
            penalty += 1.0
    min_area = _min_triangle_area(points)
    return -min_area + penalty * 10


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Define bounds for DE
    # x in [0, 1], y in [0, sqrt(3)/2]
    bounds = [(0, 1), (0, np.sqrt(3) / 2)] * N
    
    best_result = None
    best_score = -float('inf')
    
    # Strategy 1: Differential Evolution
    try:
        result = differential_evolution(
            _objective,
            bounds=bounds,
            maxiter=200,
            popsize=30,
            tol=1e-6,
            seed=42,
            polish=True,
            updating='immediate',
            init='sobol'
        )
        if result.success or result.fun < best_score:
            points = result.x.reshape((N, 2))
            points = np.array([_project_to_triangle(p) for p in points])
            score = _min_triangle_area(points)
            if score > best_score:
                best_score = score
                best_result = points
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline:
        if best_result is not None:
            return best_result
        # Fallback to random
        rng = np.random.RandomState(7)
        points = []
        while len(points) < N:
            u, v = rng.random(), rng.random()
            if u + v < 0.98 and u > 0.01 and v > 0.01:
                points.append([u + 0.5 * v, (3 ** 0.5) / 2 * v])
        return np.array(points)
    
    # Strategy 2: Local refinement with SLSQP
    if best_result is not None:
        try:
            # Flatten
            x0 = best_result.flatten()
            result2 = minimize(
                _objective,
                x0=x0,
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-10}
            )
            if result2.fun < best_score:
                points = result2.x.reshape((N, 2))
                points = np.array([_project_to_triangle(p) for p in points])
                score = _min_triangle_area(points)
                if score > best_score:
                    best_score = score
                    best_result = points
        except Exception:
            pass
    
    # Strategy 3: More DE restarts with different seeds
    for seed in [123, 456, 789]:
        if time.time() > deadline:
            break
        try:
            result = differential_evolution(
                _objective,
                bounds=bounds,
                maxiter=100,
                popsize=25,
                tol=1e-6,
                seed=seed,
                polish=True,
                updating='immediate'
            )
            if result.fun < best_score:
                points = result.x.reshape((N, 2))
                points = np.array([_project_to_triangle(p) for p in points])
                score = _min_triangle_area(points)
                if score > best_score:
                    best_score = score
                    best_result = points
        except Exception:
            pass
    
    if best_result is not None:
        return best_result
    
    # Fallback to random
    rng = np.random.RandomState(7)
    points = []
    while len(points) < N:
        u, v = rng.random(), rng.random()
        if u + v < 0.98 and u > 0.01 and v > 0.01:
            points.append([u + 0.5 * v, (3 ** 0.5) / 2 * v])
    return np.array(points)
