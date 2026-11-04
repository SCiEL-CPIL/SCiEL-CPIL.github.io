"""Heilbronn triangles: maximize min triangle area for 11 points in unit equilateral triangle."""
import numpy as np
from scipy.optimize import minimize
import time
import itertools

N = 11
TIME_LIMIT = 130  # seconds, safety margin before 150s kill

# Triangle vertices
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, np.sqrt(3) / 2])

def _inside_triangle(p):
    """Check if point p is inside or on the boundary of the triangle."""
    x, y = p
    if y < -1e-10:
        return False
    if y > np.sqrt(3) * x + 1e-10:
        return False
    if y > -np.sqrt(3) * (x - 1.0) + 1e-10:
        return False
    return True

def _project_to_triangle(p):
    """Project point p onto the triangle (closest point in triangle)."""
    x, y = p
    # Simple projection: if outside, move toward centroid
    if _inside_triangle(p):
        return np.array([x, y])
    # Move toward centroid
    centroid = (A + B + C) / 3.0
    # Try a few steps toward centroid
    for t in np.linspace(0.1, 1.0, 10):
        q = p + t * (centroid - p)
        if _inside_triangle(q):
            return q
    return centroid

def _all_triangle_areas(points):
    """Compute areas of all C(N,3) triangles."""
    n = len(points)
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
                area = 0.5 * abs(cross)
                areas.append(area)
    return np.array(areas)

def _min_area(points):
    """Return the minimum triangle area."""
    return np.min(_all_triangle_areas(points))

def _soft_min_area(points, beta=50.0):
    """Soft approximation of min area using log-sum-exp."""
    areas = _all_triangle_areas(points)
    # Use -log-sum-exp of negative areas to approximate min
    # min(areas) ≈ -log(sum(exp(-beta * areas))) / beta
    # For numerical stability
    shifted = -beta * areas
    max_shifted = np.max(shifted)
    log_sum = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
    return -log_sum / beta

def _objective_neg_soft_min(x, beta=50.0):
    """Objective to minimize: negative soft-min area."""
    points = x.reshape(-1, 2)
    # Project points back to triangle
    points = np.array([_project_to_triangle(p) for p in points])
    x = points.flatten()
    return -_soft_min_area(points, beta)

def _objective_neg_min(x):
    """Objective to minimize: negative exact min area (for final polish)."""
    points = x.reshape(-1, 2)
    points = np.array([_project_to_triangle(p) for p in points])
    return -_min_area(points)

def _refine(points, beta=50.0, max_iter=200, tol=1e-8):
    """Refine points using scipy minimize."""
    x0 = points.flatten()
    
    # First pass with soft-min
    res = minimize(
        _objective_neg_soft_min,
        x0,
        args=(beta,),
        method='L-BFGS-B',
        options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-6}
    )
    
    if res.success or res.fun < _objective_neg_soft_min(x0, beta):
        x = res.x
        points = x.reshape(-1, 2)
        points = np.array([_project_to_triangle(p) for p in points])
    
    # Second pass with exact min (fewer iterations, higher beta)
    res2 = minimize(
        _objective_neg_min,
        points.flatten(),
        method='L-BFGS-B',
        options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
    )
    
    if res2.fun < _objective_neg_min(points.flatten()):
        points = res2.x.reshape(-1, 2)
        points = np.array([_project_to_triangle(p) for p in points])
    
    return points

def _generate_initial_points(rng, strategy='grid'):
    """Generate initial points using different strategies."""
    if strategy == 'grid':
        # Place points on a grid-like pattern
        points = []
        # Vertices
        points.append(A + 0.05 * (B - A))
        points.append(B - 0.05 * (B - A))
        points.append(C + 0.05 * (A - C))
        # Edge midpoints
        points.append((A + B) / 2)
        points.append((B + C) / 2)
        points.append((C + A) / 2)
        # Interior points
        centroid = (A + B + C) / 3
        points.append(centroid)
        # Add more interior points
        for i in range(4):
            u = rng.uniform(0.1, 0.9)
            v = rng.uniform(0.1, 0.9)
            if u + v < 1.0:
                p = A + u * (B - A) + v * (C - A)
                points.append(p)
        # Trim to N points
        return np.array(points[:N])
    elif strategy == 'random':
        points = []
        while len(points) < N:
            u = rng.uniform(0.02, 0.98)
            v = rng.uniform(0.02, 0.98)
            if u + v < 0.96:
                p = A + u * (B - A) + v * (C - A)
                if _inside_triangle(p):
                    points.append(p)
        return np.array(points)
    elif strategy == 'hexagonal':
        # Hexagonal packing-like arrangement
        points = []
        # Create a hexagonal grid
        rows = 4
        cols = 4
        dx = 0.25
        dy = 0.25 * np.sqrt(3)
        for i in range(rows):
            for j in range(cols):
                x = 0.15 + j * dx + (0 if i % 2 == 0 else dx / 2)
                y = 0.15 + i * dy
                p = np.array([x, y])
                if _inside_triangle(p) and len(points) < N:
                    # Check not too close to existing points
                    if all(np.linalg.norm(p - q) > 0.1 for q in points):
                        points.append(p)
        # Fill remaining with random
        while len(points) < N:
            u = rng.uniform(0.05, 0.95)
            v = rng.uniform(0.05, 0.95)
            if u + v < 0.9:
                p = A + u * (B - A) + v * (C - A)
                if _inside_triangle(p):
                    points.append(p)
        return np.array(points[:N])
    else:
        # Default: random
        return _generate_initial_points(rng, 'random')

def construct():
    start_time = time.time()
    rng = np.random.RandomState(42)
    
    best_points = None
    best_area = 0.0
    
    # Try multiple initial strategies and random seeds
    strategies = ['grid', 'hexagonal', 'random']
    seeds = [42, 123, 456, 789, 101112]
    
    attempts = []
    for strategy in strategies:
        for seed in seeds:
            rng_attempt = np.random.RandomState(seed)
            points = _generate_initial_points(rng_attempt, strategy)
            if len(points) == N:
                attempts.append(points)
    
    # Also add some purely random starts
    for seed in seeds:
        rng_attempt = np.random.RandomState(seed + 1000)
        points = _generate_initial_points(rng_attempt, 'random')
        if len(points) == N:
            attempts.append(points)
    
    # Evaluate initial points
    for points in attempts:
        area = _min_area(points)
        if area > best_area:
            best_area = area
            best_points = points.copy()
    
    # Refine top candidates
    # Sort by initial area
    scored = []
    for points in attempts:
        area = _min_area(points)
        scored.append((area, points))
    scored.sort(key=lambda x: -x[0])
    
    # Refine the best few
    max_refine = min(10, len(scored))
    for i in range(max_refine):
        elapsed = time.time() - start_time
        if elapsed > TIME_LIMIT - 10:
            break
        
        points = scored[i][1].copy()
        refined = _refine(points, beta=50.0, max_iter=150)
        area = _min_area(refined)
        
        if area > best_area:
            best_area = area
            best_points = refined.copy()
        
        # Try with different beta
        if time.time() - start_time < TIME_LIMIT - 15:
            refined2 = _refine(points, beta=100.0, max_iter=100)
            area2 = _min_area(refined2)
            if area2 > best_area:
                best_area = area2
                best_points = refined2.copy()
    
    # Final polish on the best solution
    if best_points is not None and time.time() - start_time < TIME_LIMIT - 5:
        polished = _refine(best_points, beta=200.0, max_iter=50)
        area = _min_area(polished)
        if area > best_area:
            best_area = area
            best_points = polished
    
    # Ensure all points are inside the triangle
    if best_points is not None:
        best_points = np.array([_project_to_triangle(p) for p in best_points])
    
    return best_points
