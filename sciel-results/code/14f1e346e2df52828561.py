"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations

def _triangle_area(p1, p2, p3):
    """Compute area of triangle given 3 points (each is 2D)."""
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))

def _min_triangle_area(points):
    """Compute the minimum triangle area among all triples of points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                a = _triangle_area(points[i], points[j], points[k])
                if a < min_area:
                    min_area = a
    return min_area

def _bary_to_cart(bary, vertices):
    """Convert barycentric coordinates to Cartesian."""
    a, b, c = bary
    return np.array([a * vertices[0] + b * vertices[1] + c * vertices[2]])

def _cart_to_bary(pt, vertices):
    """Convert Cartesian to barycentric coordinates."""
    v0 = vertices[1] - vertices[0]
    v1 = vertices[2] - vertices[0]
    v2 = pt - vertices[0]
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0)
    d21 = np.dot(v2, v1)
    denom = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return np.array([u, v, w])

def _is_inside_triangle(pt, vertices, eps=1e-9):
    """Check if point is inside triangle."""
    bary = _cart_to_bary(pt, vertices)
    return np.all(bary >= -eps) and np.all(bary <= 1.0 + eps)

def _clamp_bary(bary, eps=1e-6):
    """Clamp barycentric coordinates to valid range."""
    bary = np.clip(bary, eps, None)
    bary = bary / np.sum(bary)
    return bary

def _generate_grid_config(n_rows, vertices):
    """Generate a grid-like configuration in barycentric coordinates."""
    # For 11 points, use rows of 3, 3, 3, 2
    rows = [3, 3, 3, 2]
    points_bary = []
    for i, count in enumerate(rows):
        # Height in barycentric: c coordinate (opposite to base)
        # Distribute rows evenly
        c_val = (i + 1) / (len(rows) + 1)  # between 0 and 1
        for j in range(count):
            if count == 1:
                a_val = 0.5
            else:
                a_val = (j + 1) / (count + 1)
            b_val = 1.0 - a_val - c_val
            if b_val < 0:
                b_val = 0.0
                a_val = 1.0 - c_val
            bary = _clamp_bary(np.array([a_val, b_val, c_val]))
            points_bary.append(bary)
    return points_bary

def _generate_random_config(rng, vertices):
    """Generate a random configuration in barycentric coordinates."""
    points_bary = []
    for _ in range(N):
        u = rng.random()
        v = rng.random()
        if u + v > 1.0:
            u = 1.0 - u
            v = 1.0 - v
        bary = np.array([1.0 - u - v, u, v])
        points_bary.append(bary)
    return points_bary

def _local_search(points_bary, vertices, rng, max_iters=500, step_size=0.05):
    """Perform local hill-climbing search on barycentric coordinates."""
    current = np.array(points_bary)
    current_score = _min_triangle_area([_bary_to_cart(b, vertices) for b in current])
    best = current.copy()
    best_score = current_score
    
    for it in range(max_iters):
        improved = False
        for i in range(N):
            # Try perturbing point i
            for _ in range(5):  # multiple perturbations per point
                new_bary = current[i].copy()
                # Random perturbation
                new_bary += np.array([rng.gauss(0, step_size) for _ in range(3)])
                new_bary = _clamp_bary(new_bary)
                
                # Compute new score
                new_points = [current[j].copy() for j in range(N)]
                new_points[i] = new_bary
                new_cart = [_bary_to_cart(b, vertices) for b in new_points]
                new_score = _min_triangle_area(new_cart)
                
                if new_score > current_score:
                    current[i] = new_bary
                    current_score = new_score
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best = current.copy()
        
        if not improved:
            # Reduce step size
            step_size *= 0.9
            if step_size < 1e-6:
                break
    
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Triangle vertices
    vertices = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, np.sqrt(3) / 2]
    ])
    
    rng = np.random.RandomState(42)
    
    # Generate multiple initial configurations
    initial_configs = []
    
    # Grid configurations with different row distributions
    row_distributions = [
        [3, 3, 3, 2],
        [2, 3, 3, 3],
        [3, 2, 3, 3],
        [4, 3, 2, 2],
        [2, 4, 2, 3],
        [5, 3, 2, 1],
        [1, 2, 3, 5],
    ]
    for dist in row_distributions:
        if sum(dist) == N:
            initial_configs.append(_generate_grid_config(dist, vertices))
    
    # Random configurations
    for seed in range(20):
        rng2 = np.random.RandomState(seed)
        initial_configs.append(_generate_random_config(rng2, vertices))
    
    best_points = None
    best_score = 0.0
    
    for config in initial_configs:
        if time.time() > deadline:
            break
        
        # Convert to Cartesian for initial evaluation
        cart_points = [_bary_to_cart(b, vertices) for b in config]
        score = _min_triangle_area(cart_points)
        
        if score > best_score:
            best_score = score
            best_points = np.array(cart_points)
        
        # Run local search
        if time.time() > deadline:
            break
        
        improved_points, improved_score = _local_search(config, vertices, rng, max_iters=300, step_size=0.05)
        
        if improved_score > best_score:
            best_score = improved_score
            best_points = np.array([_bary_to_cart(b, vertices) for b in improved_points])
    
    # If no valid points found, return a default
    if best_points is None:
        default_config = _generate_grid_config([3, 3, 3, 2], vertices)
        best_points = np.array([_bary_to_cart(b, vertices) for b in default_config])
    
    return best_points

# EVOLVE-BLOCK-END
