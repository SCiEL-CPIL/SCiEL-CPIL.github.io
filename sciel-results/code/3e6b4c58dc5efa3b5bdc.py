"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from scipy.optimize import differential_evolution

S3H = np.sqrt(3) / 2
TRI_AREA = S3H / 2.0  # area of equilateral triangle with side 1: sqrt(3)/4

DEADLINE = 140  # seconds, safety margin from 150s limit


def _point_in_triangle(p, v1, v2, v3):
    """Check if point p is inside or on the boundary of triangle v1,v2,v3."""
    # Using barycentric coordinates
    x, y = p
    x1, y1 = v1
    x2, y2 = v2
    x3, y3 = v3
    
    denom = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
    if abs(denom) < 1e-12:
        return False
    
    a = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / denom
    b = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / denom
    c = 1.0 - a - b
    
    return (a >= -1e-9) and (b >= -1e-9) and (c >= -1e-9)


def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by 3 of the given points.
    Returns the raw area (not normalized)."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                dx1 = points[j, 0] - points[i, 0]
                dy1 = points[j, 1] - points[i, 1]
                dx2 = points[k, 0] - points[i, 0]
                dy2 = points[k, 1] - points[i, 1]
                area = abs(dx1 * dy2 - dy1 * dx2) * 0.5
                if area < min_area:
                    min_area = area
    return min_area


def _is_valid(points):
    """Check if all points are inside the unit equilateral triangle."""
    v1 = np.array([0.0, 0.0])
    v2 = np.array([1.0, 0.0])
    v3 = np.array([0.5, S3H])
    for p in points:
        if not _point_in_triangle(p, v1, v2, v3):
            return False
    return True


def _score(points):
    """Compute the normalized minimum triangle area (divided by triangle area)."""
    if not _is_valid(points):
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / TRI_AREA


def _random_point_in_triangle(rng):
    """Generate a random point uniformly inside the unit equilateral triangle."""
    u = rng.random()
    v = rng.random()
    if u + v > 1.0:
        u = 1.0 - u
        v = 1.0 - v
    # Map to triangle with vertices (0,0), (1,0), (0.5, sqrt(3)/2)
    x = u + 0.5 * v
    y = S3H * v
    return np.array([x, y])


def _initial_guess():
    """Generate an initial configuration using a grid-like pattern."""
    # Place points in a roughly uniform pattern
    points = []
    # Try a few structured patterns
    # Pattern 1: 3x3 grid plus 2 extra
    for i in range(3):
        for j in range(3):
            # Map grid to triangle
            u = (i + 0.5) / 3.0
            v = (j + 0.5) / 3.0
            if u + v <= 1.0:
                x = u + 0.5 * v
                y = S3H * v
                points.append([x, y])
    
    # Add 2 more points at vertices or near them
    if len(points) < N:
        points.append([0.05, 0.05])
    if len(points) < N:
        points.append([0.95, 0.05])
    if len(points) < N:
        points.append([0.5, S3H - 0.05])
    
    # Take first N points
    points = points[:N]
    
    # Ensure we have exactly N points
    while len(points) < N:
        points.append([0.5, S3H / 2])
    
    return np.array(points[:N])


def _local_search(points, max_iters=500, step_size=0.02, deadline=None):
    """Perform local search to improve the minimum triangle area."""
    points = points.copy()
    current_score = _score(points)
    best_points = points.copy()
    best_score = current_score
    
    rng = np.random.default_rng(42)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Pick a random point to move
        idx = rng.integers(0, N)
        old_p = points[idx].copy()
        
        # Try small random perturbations
        for _ in range(20):
            perturbation = rng.normal(0, step_size, size=2)
            new_p = old_p + perturbation
            
            # Check if still in triangle
            if not _point_in_triangle(new_p, np.array([0.0, 0.0]), 
                                       np.array([1.0, 0.0]), 
                                       np.array([0.5, S3H])):
                continue
            
            points[idx] = new_p
            new_score = _score(points)
            
            if new_score > current_score:
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_points = points.copy()
            else:
                points[idx] = old_p
                break
    
    return best_points, best_score


def construct():
    start_time = time.time()
    
    # Start with a structured initial guess
    points = _initial_guess()
    best_points = points.copy()
    best_score = _score(points)
    
    # Also try a few random starts
    rng = np.random.default_rng(7)
    for _ in range(5):
        if time.time() - start_time > DEADLINE - 20:
            break
        random_points = np.array([_random_point_in_triangle(rng) for _ in range(N)])
        if _is_valid(random_points):
            s = _score(random_points)
            if s > best_score:
                best_score = s
                best_points = random_points.copy()
    
    # Run local search
    deadline = start_time + DEADLINE - 10
    for _ in range(3):
        if time.time() > deadline:
            break
        points, score = _local_search(best_points, max_iters=300, step_size=0.015, 
                                       deadline=deadline)
        if score > best_score:
            best_score = score
            best_points = points
    
    # Try differential evolution if time permits
    if time.time() - start_time < DEADLINE - 30:
        def _objective(flat_params):
            pts = flat_params.reshape(N, 2)
            if not _is_valid(pts):
                return -1.0
            return -_score(pts)
        
        bounds = [(0, 1), (0, S3H)] * N
        de_start = time.time()
        de_deadline = min(deadline, de_start + 60)
        
        # Use a reduced population for speed
        result = differential_evolution(
            _objective,
            bounds,
            maxiter=50,
            popsize=10,
            seed=42,
            tol=1e-4,
            update_interval=1
        )
        
        if result.success and result.fun > -1.0:
            de_points = result.x.reshape(N, 2)
            if _is_valid(de_points):
                de_score = _score(de_points)
                if de_score > best_score:
                    best_score = de_score
                    best_points = de_points
    
    return best_points
# EVOLVE-BLOCK-END
