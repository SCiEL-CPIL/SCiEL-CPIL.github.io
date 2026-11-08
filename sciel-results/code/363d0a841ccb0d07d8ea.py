"""Heilbronn triangles: maximize min triangle area for 11 points in unit-area equilateral triangle."""
import numpy as np
import random
import time
from itertools import combinations

N = 11

# Equilateral triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2.0])
AREA = np.sqrt(3) / 4.0  # area of the equilateral triangle


def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by 3 of the points, normalized by triangle area."""
    n = len(points)
    # Vectorized computation
    # For each triple (i,j,k), area = 0.5 * |cross(pj-pi, pk-pi)|
    # We can compute all pairs of differences
    min_area = float('inf')
    for i in range(n):
        pi = points[i]
        for j in range(i + 1, n):
            pj = points[j]
            v1 = pj - pi
            for k in range(j + 1, n):
                pk = points[k]
                v2 = pk - pi
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if area < min_area:
                    min_area = area
    return min_area / AREA


def _is_inside_triangle(p):
    """Check if point p is inside or on the boundary of the equilateral triangle."""
    x, y = p
    if y < -1e-9:
        return False
    if x < -1e-9 or x > 1.0 + 1e-9:
        return False
    max_y = np.sqrt(3) * x if x <= 0.5 else np.sqrt(3) * (1.0 - x)
    return y <= max_y + 1e-9


def _is_inside_triangle_batch(pts):
    """Vectorized check for multiple points."""
    x = pts[:, 0]
    y = pts[:, 1]
    mask = (y >= -1e-9) & (x >= -1e-9) & (x <= 1.0 + 1e-9)
    max_y = np.where(x <= 0.5, np.sqrt(3) * x, np.sqrt(3) * (1.0 - x))
    mask &= (y <= max_y + 1e-9)
    return mask


def _generate_structured_initial():
    """Generate a structured initial configuration based on known good patterns."""
    # Place points in a pattern that avoids clustering
    # Use a grid-like structure within the triangle
    points = []
    # Vertices
    points.append(V0.copy())
    points.append(V1.copy())
    points.append(V2.copy())
    
    # Midpoints of edges
    points.append((V0 + V1) / 2.0)
    points.append((V1 + V2) / 2.0)
    points.append((V2 + V0) / 2.0)
    
    # Centroid
    points.append((V0 + V1 + V2) / 3.0)
    
    # Additional points - place them in a structured way
    # Use points at 1/3 and 2/3 positions
    for t in [1/3, 2/3]:
        # Along edges
        points.append(V0 * (1-t) + V1 * t)
    
    # We need exactly 11 points, we have 9 so far
    # Add 2 more points
    # Place them in the interior
    h = np.sqrt(3) / 2.0
    points.append(np.array([0.5, h * 0.5]))
    points.append(np.array([0.5, h * 0.25]))
    
    # Adjust to get exactly 11
    points = points[:N]
    return np.array(points)


def _generate_random_points(rng, n=N):
    """Generate n random points inside the triangle using rejection sampling."""
    points = []
    while len(points) < n:
        u = rng.random()
        v = rng.random()
        if u + v <= 1.0:
            p = V0 * (1 - u - v) + V1 * u + V2 * v
            points.append(p)
    return np.array(points)


def _coordinate_descent(points, max_iter=500, step_sizes=None, rng=None):
    """Coordinate descent optimization."""
    if rng is None:
        rng = random.Random()
    if step_sizes is None:
        step_sizes = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002]
    
    current = points.copy()
    current_min = _min_triangle_area(current)
    best_points = current.copy()
    best_min = current_min
    
    for step_size in step_sizes:
        improved = True
        while improved:
            improved = False
            for i in range(N):
                orig = current[i].copy()
                # Try moving in x and y directions
                for dx in [-step_size, -step_size/2, step_size/2, step_size]:
                    for dy in [-step_size, -step_size/2, step_size/2, step_size]:
                        new_p = orig + np.array([dx, dy])
                        if _is_inside_triangle(new_p):
                            candidate = current.copy()
                            candidate[i] = new_p
                            new_min = _min_triangle_area(candidate)
                            if new_min > current_min:
                                current[i] = new_p
                                current_min = new_min
                                improved = True
                                if new_min > best_min:
                                    best_min = new_min
                                    best_points = current.copy()
                                break
                    else:
                        continue
                    break
                # Also try diagonal moves
                for angle in [np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4]:
                    new_p = orig + step_size * np.array([np.cos(angle), np.sin(angle)])
                    if _is_inside_triangle(new_p):
                        candidate = current.copy()
                        candidate[i] = new_p
                        new_min = _min_triangle_area(candidate)
                        if new_min > current_min:
                            current[i] = new_p
                            current_min = new_min
                            improved = True
                            if new_min > best_min:
                                best_min = new_min
                                best_points = current.copy()
                            break
    
    return best_points, best_min


def _random_perturb_optimize(points, max_iter=200, rng=None):
    """Random perturbation local search."""
    if rng is None:
        rng = random.Random()
    
    current = points.copy()
    current_min = _min_triangle_area(current)
    best_points = current.copy()
    best_min = current_min
    step_size = 0.1
    
    for it in range(max_iter):
        improved = False
        for i in range(N):
            orig = current[i].copy()
            angle = rng.random() * 2 * np.pi
            dist = rng.random() * step_size
            new_p = current[i] + np.array([dist * np.cos(angle), dist * np.sin(angle)])
            if _is_inside_triangle(new_p):
                current[i] = new_p
                new_min = _min_triangle_area(current)
                if new_min > best_min:
                    best_min = new_min
                    best_points = current.copy()
                    improved = True
                else:
                    current[i] = orig
        if not improved:
            step_size *= 0.7
            if step_size < 1e-4:
                break
    
    return best_points, best_min


def _simulated_annealing(points, max_iter=1000, t_start=0.1, t_end=0.001, rng=None):
    """Simulated annealing for global optimization."""
    if rng is None:
        rng = random.Random()
    
    current = points.copy()
    current_min = _min_triangle_area(current)
    best_points = current.copy()
    best_min = current_min
    
    n_iter = max_iter
    for it in range(n_iter):
        T = t_start * (t_end / t_start) ** (it / n_iter)
        
        # Random move: pick a random point and perturb it
        i = rng.randint(0, N - 1)
        angle = rng.random() * 2 * np.pi
        dist = rng.random() * 0.1
        orig = current[i].copy()
        new_p = current[i] + np.array([dist * np.cos(angle), dist * np.sin(angle)])
        
        if _is_inside_triangle(new_p):
            current[i] = new_p
            new_min = _min_triangle_area(current)
            
            if new_min > current_min:
                current_min = new_min
                if new_min > best_min:
                    best_min = new_min
                    best_points = current.copy()
            else:
                # Accept with probability exp((new_min - current_min) / T)
                delta = new_min - current_min
                if rng.random() < np.exp(delta / T):
                    current_min = new_min
                else:
                    current[i] = orig
    
    return best_points, best_min


# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s
    
    best_points = None
    best_min = 0.0
    
    rng = random.Random(42)
    
    # Strategy 1: Start with structured initial configuration
    structured = _generate_structured_initial()
    cd_points, cd_min = _coordinate_descent(structured, max_iter=300, rng=rng)
    if cd_min > best_min:
        best_min = cd_min
        best_points = cd_points.copy()
    
    # Strategy 2: Random restarts with coordinate descent
    n_restarts = 40
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        points = _generate_random_points(rng)
        cd_points, cd_min = _coordinate_descent(points, max_iter=200, rng=rng)
        if cd_min > best_min:
            best_min = cd_min
            best_points = cd_points.copy()
    
    # Strategy 3: Simulated annealing from best
    if best_points is not None and time.time() < deadline - 10:
        sa_points, sa_min = _simulated_annealing(best_points.copy(), max_iter=500, rng=rng)
        if sa_min > best_min:
            best_min = sa_min
            best_points = sa_points.copy()
    
    # Strategy 4: More intensive coordinate descent from best
    if best_points is not None and time.time() < deadline - 5:
        for step_sizes in [[0.05, 0.02, 0.01, 0.005], [0.02, 0.01, 0.005, 0.002]]:
            if time.time() > deadline:
                break
            cd_points, cd_min = _coordinate_descent(best_points.copy(), max_iter=300, 
                                                     step_sizes=step_sizes, rng=rng)
            if cd_min > best_min:
                best_min = cd_min
                best_points = cd_points.copy()
    
    # Strategy 5: Random perturbation from best
    if best_points is not None and time.time() < deadline - 3:
        rp_points, rp_min = _random_perturb_optimize(best_points.copy(), max_iter=100, rng=rng)
        if rp_min > best_min:
            best_min = rp_min
            best_points = rp_points.copy()
    
    # Final check
    if best_points is None:
        best_points = _generate_structured_initial()
    
    return best_points
# EVOLVE-BLOCK-END
