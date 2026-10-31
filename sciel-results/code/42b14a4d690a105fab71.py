"""Heilbronn triangles: place 11 points in a unit-area equilateral triangle to maximize the minimum triangle area."""
import numpy as np
import time
from itertools import combinations
from scipy.optimize import minimize

N = 11

# Pre-compute triangle index combinations
TRI_INDICES = np.array(list(combinations(range(N), 3)))
N_TRIS = TRI_INDICES.shape[0]

# Equilateral triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2.0])
H = np.sqrt(3) / 2.0  # height of equilateral triangle


def _check_inside(points):
    """Check if all points are inside or on the boundary of the equilateral triangle."""
    # points: (N, 2)
    x = points[:, 0]
    y = points[:, 1]
    # y >= 0
    if np.any(y < -1e-10):
        return False
    # y <= sqrt(3)/2
    if np.any(y > H + 1e-10):
        return False
    # y <= sqrt(3) * x  (left edge: from (0,0) to (0.5, sqrt(3)/2))
    if np.any(y > np.sqrt(3) * x + 1e-10):
        return False
    # y <= sqrt(3) * (1 - x)  (right edge: from (1,0) to (0.5, sqrt(3)/2))
    if np.any(y > np.sqrt(3) * (1.0 - x) + 1e-10):
        return False
    return True


def _min_area(points):
    """Compute the minimum area of any triangle formed by 3 of the N points."""
    # points: (N, 2)
    p1 = points[TRI_INDICES[:, 0]]
    p2 = points[TRI_INDICES[:, 1]]
    p3 = points[TRI_INDICES[:, 2]]
    # Area = 0.5 * |cross(p2-p1, p3-p1)|
    dx1 = p2[:, 0] - p1[:, 0]
    dy1 = p2[:, 1] - p1[:, 1]
    dx2 = p3[:, 0] - p1[:, 0]
    dy2 = p3[:, 1] - p1[:, 1]
    cross = np.abs(dx1 * dy2 - dy1 * dx2)
    return 0.5 * np.min(cross)


def _objective(x):
    """Objective: negative of min area (to minimize)."""
    points = x.reshape(N, 2)
    if not _check_inside(points):
        return 1e6
    return -_min_area(points)


def _project_inside(points):
    """Project points to be inside the triangle (simple clipping)."""
    points = points.copy()
    x = points[:, 0]
    y = points[:, 1]
    # Clip y to [0, H]
    y = np.clip(y, 0, H)
    # Clip based on left edge: y <= sqrt(3)*x => x >= y/sqrt(3)
    x = np.maximum(x, y / np.sqrt(3))
    # Clip based on right edge: y <= sqrt(3)*(1-x) => x <= 1 - y/sqrt(3)
    x = np.minimum(x, 1.0 - y / np.sqrt(3))
    # Re-clip y after x changes
    y = np.minimum(y, np.sqrt(3) * x)
    y = np.minimum(y, np.sqrt(3) * (1.0 - x))
    y = np.maximum(y, 0.0)
    points[:, 0] = x
    points[:, 1] = y
    return points


def _initial_grid():
    """Create an initial configuration using a grid-like pattern."""
    # Use a 3x3 grid plus 2 extra points, mapped into the triangle
    # Actually, let's use a more structured approach
    # Place points on a triangular lattice pattern
    
    # For 11 points, use rows: 1, 3, 4, 3 or similar
    # Let's try: row 0: 1 point, row 1: 3 points, row 2: 4 points, row 3: 3 points
    points = []
    
    # Row 0 (bottom): 1 point at center
    points.append([0.5, 0.05])
    
    # Row 1: 3 points
    for i in range(3):
        x = 0.2 + i * 0.3
        y = 0.25
        points.append([x, y])
    
    # Row 2: 4 points
    for i in range(4):
        x = 0.15 + i * 0.25
        y = 0.45
        points.append([x, y])
    
    # Row 3: 3 points
    for i in range(3):
        x = 0.25 + i * 0.25
        y = 0.6
        points.append([x, y])
    
    points = np.array(points)
    points = _project_inside(points)
    return points


def _random_inside(rng):
    """Generate a random point inside the equilateral triangle."""
    u = rng.random()
    v = rng.random()
    while u + v > 1.0:
        u = rng.random()
        v = rng.random()
    # Barycentric coordinates: (1-u-v, u, v)
    x = u * V1[0] + v * V2[0]
    y = u * V1[1] + v * V2[1]
    return [x, y]


def _random_initial(rng):
    """Generate a random initial configuration."""
    points = np.array([_random_inside(rng) for _ in range(N)])
    points = _project_inside(points)
    return points


def _local_optimize(points, max_iter=1000, tol=1e-10):
    """Local optimization using Nelder-Mead on the flattened point array."""
    x0 = points.flatten()
    result = minimize(
        _objective,
        x0,
        method='Nelder-Mead',
        options={'maxiter': max_iter, 'xatol': tol, 'fatol': tol, 'adaptive': True}
    )
    best_points = result.x.reshape(N, 2)
    best_points = _project_inside(best_points)
    return best_points, -result.fun if result.fun < 1e5 else 0.0


def _simulated_annealing(points, rng, n_iter=5000, t_start=0.05, t_end=0.001):
    """Simulated annealing to improve the configuration."""
    best_points = points.copy()
    best_area = _min_area(points)
    current_points = points.copy()
    current_area = best_area
    
    n_total = N * 2
    for i in range(n_iter):
        t = t_start * (t_end / t_start) ** (i / n_iter)
        
        # Random perturbation
        delta = np.random.normal(0, 0.02 * (t_start / t + 0.1), size=(N, 2))
        new_points = current_points + delta
        new_points = _project_inside(new_points)
        
        new_area = _min_area(new_points)
        
        if new_area > current_area or rng.random() < np.exp((new_area - current_area) / t):
            current_points = new_points
            current_area = new_area
            if new_area > best_area:
                best_area = new_area
                best_points = new_points.copy()
    
    return best_points, best_area


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    rng = np.random.RandomState(42)
    random_state = 42
    
    best_points = None
    best_area = 0.0
    
    # Start with a structured initial guess
    init_points = _initial_grid()
    best_points, best_area = _local_optimize(init_points, max_iter=2000)
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        return best_points
    
    # Try multiple random restarts with local optimization
    n_restarts = 0
    while time.time() < deadline - 20.0:  # Leave 20s buffer
        n_restarts += 1
        rand_points = _random_initial(rng)
        
        # Quick local optimization
        opt_points, opt_area = _local_optimize(rand_points, max_iter=500)
        
        if opt_area > best_area:
            best_points = opt_points
            best_area = opt_area
        
        # If time allows, do simulated annealing on the best
        if time.time() < deadline - 30.0 and n_restarts % 3 == 0:
            sa_points, sa_area = _simulated_annealing(
                best_points, rng, 
                n_iter=2000, 
                t_start=0.03 * (1.0 / (1.0 + n_restarts * 0.1)),
                t_end=0.001
            )
            if sa_area > best_area:
                best_points = sa_points
                best_area = sa_area
    
    # Final local optimization on the best
    if time.time() < deadline - 5.0:
        final_points, final_area = _local_optimize(best_points, max_iter=1000)
        if final_area > best_area:
            best_points = final_points
            best_area = final_area
    
    return best_points
