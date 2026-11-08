"""Heilbronn triangles: maximize min triangle area for 11 points in unit-area equilateral triangle."""
import numpy as np
import random
import time
from itertools import combinations
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull

N = 11
S3 = np.sqrt(3.0)
# Equilateral triangle vertices: (0,0), (1,0), (0.5, sqrt(3)/2)
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, S3 / 2.0])
AREA = S3 / 4.0  # area of the equilateral triangle with side 1

# Deadline: must finish within 150s, use 140s as safety margin
DEADLINE = 140.0


def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by 3 of the points."""
    n = len(points)
    min_area = float('inf')
    # Vectorized calculation for speed
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                area = 0.5 * abs((points[j][0] - points[i][0]) * (points[k][1] - points[i][1])
                                 - (points[j][1] - points[i][1]) * (points[k][0] - points[i][0]))
                if area < min_area:
                    min_area = area
    return min_area

def _min_triangle_area_vec(points):
    """Vectorized computation of min triangle area."""
    n = points.shape[0]
    areas = []
    for i in range(n):
        pi = points[i]
        for j in range(i+1, n):
            pj = points[j]
            # Vectorized for k > j
            if j + 1 < n:
                pk = points[j+1:]
                v1 = pj - pi
                v2 = pk - pi
                cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
                tri_areas = 0.5 * np.abs(cross)
                areas.extend(tri_areas)
    if not areas:
        return float('inf')
    return np.min(areas)

def _is_inside_triangle(p):
    """Check if point p is inside or on the boundary of the equilateral triangle."""
    x, y = p
    if y < -1e-9 or x < -1e-9 or x > 1.0 + 1e-9:
        return False
    # Check below the line from V0 to V2: y <= sqrt(3)*x
    if y > S3 * x + 1e-9:
        return False
    # Check below the line from V1 to V2: y <= -sqrt(3)*(x-1)
    if y > S3 * (1.0 - x) + 1e-9:
        return False
    return True


def _random_point_in_triangle(rng):
    """Generate a random point uniformly inside the equilateral triangle."""
    u = rng.random()
    v = rng.random()
    if u + v > 1.0:
        u = 1.0 - u
        v = 1.0 - v
    # Barycentric coordinates: (1-u-v, u, v) mapped to triangle
    p = (1.0 - u - v) * V0 + u * V1 + v * V2
    return p


def _generate_random_points(rng, n=N):
    """Generate n random points inside the triangle."""
    points = []
    for _ in range(n):
        points.append(_random_point_in_triangle(rng))
    return np.array(points)

def _map_params_to_points(params):
    """Map parameters from [0,1] x [0,1] to points in the triangle."""
    # params shape (N, 2)
    points = np.zeros((N, 2))
    for i in range(N):
        u = params[i, 0]
        v = params[i, 1]
        if u + v > 1.0:
            u = 1.0 - u
            v = 1.0 - v
        points[i] = (1.0 - u - v) * V0 + u * V1 + v * V2
    return points

def _objective(params):
    """Objective function for optimization: minimize negative min area."""
    points = _map_params_to_points(params)
    min_area = _min_triangle_area_vec(points)
    return -min_area

def construct():
    """Construct 11 points in the equilateral triangle to maximize the minimum triangle area."""
    start_time = time.time()
    
    # Initial structured configuration
    # Based on known good configurations or symmetry
    # Let's try a grid-like structure first
    rng = random.Random(42)
    
    # Try a few random starts and one structured start
    initial_guesses = []
    
    # Structured guess: vertices and midpoints and centroid
    # This might not be optimal but is a decent start
    p0 = V0 + 0.1 * (V1 + V2 - 2*V0) / 2 # Move inwards
    # Actually, let's just use random points for DE, it's robust.
    
    # Define bounds for DE: each point has 2 coordinates in [0,1] (mapped)
    bounds = [(0, 1)] * (2 * N)
    
    best_points = None
    best_score = -float('inf')
    
    # Run Differential Evolution
    # maxiter and popsize control the budget
    # 150s limit. DE can be slow. Let's limit it.
    try:
        result = differential_evolution(
            _objective, 
            bounds=bounds, 
            seed=42, 
            maxiter=50, 
            popsize=10, 
            tol=0.01,
            polish=True
        )
        
        if result.success or result.fun < 0: # If it found something
            best_params = result.x
            best_points = _map_params_to_points(best_params)
            best_score = -result.fun
            
            if time.time() - start_time < DEADLINE - 10:
                # Polish with local optimizer
                # We need to map back to Cartesian for local opt?
                # Or just stick with param space?
                # Let's do a local polish in Cartesian space for the best point found
                local_result = minimize(
                    lambda p: -_min_triangle_area_vec(_map_params_to_points(p.reshape(-1, 2))),
                    best_params,
                    method='Nelder-Mead',
                    options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6}
                )
                if local_result.fun < result.fun:
                    best_params = local_result.x
                    best_points = _map_params_to_points(best_params)
                    best_score = -local_result.fun

    except Exception as e:
        print(f"DE failed: {e}")
        # Fallback to random search if DE fails
        for _ in range(100):
            if time.time() - start_time > DEADLINE - 5:
                break
            points = _generate_random_points(rng)
            score = _min_triangle_area_vec(points)
            if score > best_score:
                best_score = score
                best_points = points.copy()

    if best_points is None:
        # Fallback to a simple random point if everything failed
        best_points = _generate_random_points(rng)
        
    return best_points
