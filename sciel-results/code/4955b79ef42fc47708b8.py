"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import itertools
import time

V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
TRI_AREA = np.sqrt(3) / 4.0  # area of the equilateral triangle with side 1


def point_in_triangle(p, a=V0, b=V1, c=V2, eps=1e-12):
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    d1 = sign(p, a, b)
    d2 = sign(p, b, c)
    d3 = sign(p, c, a)
    has_neg = (d1 < -eps) or (d2 < -eps) or (d3 < -eps)
    has_pos = (d1 > eps) or (d2 > eps) or (d3 > eps)
    return not (has_neg and has_pos)


def min_triangle_area(points):
    """Return the minimum area of any triangle formed by three of the points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = 0.5 * abs((points[j][0] - points[i][0]) * (points[k][1] - points[i][1]) -
                                 (points[k][0] - points[i][0]) * (points[j][1] - points[i][1]))
                if area < min_area:
                    min_area = area
    return min_area


def objective(x):
    """Objective to minimize: negative of the minimum triangle area."""
    points = x.reshape(-1, 2)
    # Check all points are inside the triangle (with small margin)
    for p in points:
        if not point_in_triangle(p, eps=1e-10):
            return 1.0  # large penalty
    return -min_triangle_area(points)


def project_into_triangle(p):
    """Project point p into the triangle if it's outside."""
    if point_in_triangle(p, eps=-1e-9):
        return p
    # Move toward centroid
    centroid = (V0 + V1 + V2) / 3.0
    t = 0.5
    for _ in range(50):
        new_p = p + t * (centroid - p)
        if point_in_triangle(new_p, eps=1e-10):
            return new_p
        t *= 0.5
    return centroid


def generate_initial_points(seed):
    """Generate diverse initial points using structured grid + jitter."""
    rng = np.random.RandomState(seed)
    # Create a grid of candidate positions
    n_grid = 6
    points = []
    for i in range(n_grid):
        for j in range(n_grid):
            # Barycentric coordinates
            u = (i + 0.5) / n_grid
            v = (j + 0.5) / n_grid
            if u + v < 1.0:
                # Convert barycentric to Cartesian
                p = u * V1 + v * V2
                points.append(p)
    
    # Add jitter to create diversity
    jittered_points = []
    for p in points:
        jitter = rng.normal(0, 0.05, 2)
        new_p = p + jitter
        if point_in_triangle(new_p, eps=1e-10):
            jittered_points.append(new_p)
    
    # Select N points from jittered candidates
    if len(jittered_points) >= N:
        # Use farthest-point sampling for diversity
        selected = [jittered_points[0]]
        for _ in range(N - 1):
            best_idx = 0
            best_min_dist = -1
            for idx, p in enumerate(jittered_points):
                if idx in [jittered_points.index(s) for s in selected]:
                    continue
                min_dist = min(np.linalg.norm(p - s) for s in selected)
                if min_dist > best_min_dist:
                    best_min_dist = min_dist
                    best_idx = idx
            selected.append(jittered_points[best_idx])
        return np.array(selected)
    else:
        # Fallback: random points
        pts = []
        while len(pts) < N:
            u = rng.random()
            v = rng.random()
            if u + v < 1.0:
                p = u * V1 + v * V2
                pts.append(p)
        return np.array(pts)


def nelder_mead_optimize(initial_points, max_iter=500, timeout=20):
    """Run Nelder-Mead optimization starting from initial_points."""
    x0 = initial_points.flatten()
    start_time = time.time()
    
    try:
        result = minimize(
            objective,
            x0,
            method='Nelder-Mead',
            options={
                'maxiter': max_iter,
                'xatol': 1e-8,
                'fatol': 1e-10,
                'adaptive': True
            }
        )
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    return result.x.reshape(-1, 2), elapsed


def construct():
    """Construct 11 points inside the equilateral triangle to maximize minimum triangle area."""
    deadline = time.time() + 140  # Leave 10s safety margin
    
    best_points = None
    best_area = -1
    
    # Multi-start optimization with different seeds
    for seed in range(20):
        if time.time() > deadline - 25:
            break
        
        initial_points = generate_initial_points(seed)
        
        # Run Nelder-Mead
        optimized_points, elapsed = nelder_mead_optimize(initial_points, max_iter=300, timeout=15)
        
        # Ensure all points are inside the triangle
        valid = True
        for p in optimized_points:
            if not point_in_triangle(p, eps=1e-10):
                valid = False
                break
        
        if valid:
            area = min_triangle_area(optimized_points)
            if area > best_area:
                best_area = area
                best_points = optimized_points
        
        # Also check the initial points
        init_area = min_triangle_area(initial_points)
        if init_area > best_area:
            best_area = init_area
            best_points = initial_points
    
    # If no valid points found, use a simple fallback
    if best_points is None:
        best_points = generate_initial_points(0)
    
    return best_points
# EVOLVE-BLOCK-END

