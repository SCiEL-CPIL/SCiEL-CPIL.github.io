"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations

def _precompute_indices(n):
    """Precompute all (i, j, k) index triples for i < j < k."""
    triples = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                triples.append((i, j, k))
    return np.array(triples, dtype=np.int32)

_TRIPLES = _precompute_indices(N)

def _min_triangle_area_vec(points):
    """
    Vectorized computation of minimum triangle area.
    points: array of shape (n, 2)
    Returns the minimum area over all C(n,3) triangles.
    """
    i, j, k = _TRIPLES[:, 0], _TRIPLES[:, 1], _TRIPLES[:, 2]
    pi, pj, pk = points[i], points[j], points[k]
    # Area = 0.5 * |cross(pj - pi, pk - pi)|
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pk[:, 0] - pi[:, 0]) * (pj[:, 1] - pi[:, 1])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _clamp_to_triangle(p):
    """Clamp a point to be inside the equilateral triangle (0,0), (1,0), (0.5, sqrt(3)/2)."""
    x, y = p
    h = np.sqrt(3) / 2
    eps = 1e-10
    
    y = max(eps, y)
    y = min(y, h - eps)
    
    x_min_left = y / np.sqrt(3)
    x_max_right = 1 - y / np.sqrt(3)
    
    x = max(x_min_left + eps, x)
    x = min(x_max_right - eps, x)
    
    return np.array([x, y])

def _clamp_all(points):
    """Clamp all points to be inside the triangle."""
    return np.array([_clamp_to_triangle(p) for p in points])

def _evaluate(points):
    """Evaluate the minimum triangle area for a set of points."""
    return _min_triangle_area_vec(points)

def _simulated_annealing(points, T0=0.1, T_end=0.001, cooling=0.9995, max_iter=50000, deadline=None):
    """
    Perform simulated annealing to maximize the minimum triangle area.
    Returns (optimized_points, score).
    """
    rng = np.random.RandomState()
    current_points = np.array(points).copy()
    current_score = _evaluate(current_points)
    
    best_points = current_points.copy()
    best_score = current_score
    
    h = np.sqrt(3) / 2
    
    for iteration in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        T = T0 * (cooling ** iteration)
        if T < T_end:
            break
        
        # Generate a perturbation: move one or two random points
        num_moved = rng.choice([1, 1, 2])  # Mostly move 1 point
        indices = rng.choice(N, size=num_moved, replace=False)
        
        new_points = current_points.copy()
        for idx in indices:
            # Random step in [-step_size, step_size] for each coordinate
            step_size = 0.05 * (1 + T / T0)  # Larger steps at higher T
            dx = rng.uniform(-step_size, step_size)
            dy = rng.uniform(-step_size, step_size)
            new_points[idx, 0] += dx
            new_points[idx, 1] += dy
        
        # Clamp to triangle
        new_points = _clamp_all(new_points)
        
        # Evaluate
        new_score = _evaluate(new_points)
        
        # Accept or reject
        if new_score >= current_score:
            current_points = new_points
            current_score = new_score
            if current_score > best_score:
                best_points = new_points.copy()
                best_score = current_score
        else:
            # Accept with probability exp((new_score - current_score) / T)
            delta = new_score - current_score
            if delta > 0 or rng.random() < np.exp(delta / T):
                current_points = new_points
                current_score = new_score
    
    return best_points, best_score

def _generate_initial_configs(seed=42):
    """Generate diverse initial point configurations."""
    np.random.seed(seed)
    h = np.sqrt(3) / 2
    configs = []
    
    # Config 1: Symmetric hexagonal-like arrangement
    config1 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, h],
        [0.5, h/3],
        [0.25, h/6],
        [0.75, h/6],
        [0.5, 2*h/3],
        [0.15, h/4],
        [0.85, h/4],
        [0.35, 3*h/8],
        [0.65, 3*h/8],
    ])
    configs.append(config1)
    
    # Config 2: Evenly spaced on a triangular grid
    config2 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, h],
        [0.5, 0.0],
        [0.25, h/2],
        [0.75, h/2],
        [0.5, h/3],
        [0.25, h/6],
        [0.75, h/6],
        [0.5, 2*h/3],
        [0.5, h/2],
    ])
    configs.append(config2)
    
    # Config 3: Perturbed symmetric
    base3 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, h],
        [0.5, h/3],
        [0.25, h/6],
        [0.75, h/6],
        [0.5, 2*h/3],
        [0.15, h/4],
        [0.85, h/4],
        [0.35, 3*h/8],
        [0.65, 3*h/8],
    ])
    config3 = base3 + 0.05 * np.random.randn(11, 2)
    configs.append(config3)
    
    # Config 4: Random
    config4 = []
    for _ in range(N):
        u = np.random.random()
        v = np.random.random()
        if u + v > 1:
            u = 1 - u
            v = 1 - v
        x = u + 0.5 * v
        y = h * v
        config4.append([x, y])
    configs.append(np.array(config4))
    
    # Config 5: Another symmetric pattern
    config5 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, h],
        [0.33, 0.0],
        [0.67, 0.0],
        [0.25, h/2],
        [0.75, h/2],
        [0.5, h/3],
        [0.5, 2*h/3],
        [0.33, h/3],
        [0.67, h/3],
    ])
    configs.append(config5)
    
    # Config 6: Centroid-based
    config6 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, h],
        [0.5, h/3],
        [0.5, 2*h/3],
        [0.2, h/4],
        [0.8, h/4],
        [0.2, 3*h/4],
        [0.8, 3*h/4],
        [0.35, h/2],
        [0.65, h/2],
    ])
    configs.append(config6)
    
    return configs

def construct():
    """
    Place 11 points on or inside a unit-area equilateral triangle to maximize
    the smallest area of a triangle formed by three points.
    """
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    h = np.sqrt(3) / 2
    
    # Generate initial configurations
    initial_configs = _generate_initial_configs(seed=42)
    
    best_points = None
    best_score = -np.inf
    
    # Run simulated annealing from each initial configuration
    for i, config in enumerate(initial_configs):
        if time.time() > deadline - 10:
            break
        
        points = _clamp_all(config)
        current_score = _evaluate(points)
        
        if current_score > best_score:
            best_points = points.copy()
            best_score = current_score
        
        # Run simulated annealing
        if time.time() < deadline - 5:
            sa_points, sa_score = _simulated_annealing(
                points,
                T0=0.08,
                T_end=0.0005,
                cooling=0.9997,
                max_iter=80000,
                deadline=deadline - 2
            )
            if sa_score > best_score:
                best_points = sa_points
                best_score = sa_score
    
    # Try additional random restarts if time permits
    rng = np.random.RandomState(123)
    num_restarts = 20
    for restart in range(num_restarts):
        if time.time() > deadline - 8:
            break
        
        # Generate random initial points
        if restart % 2 == 0:
            # Symmetric base with perturbation
            base = np.array([
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, h],
                [0.5, h/3],
                [0.25, h/6],
                [0.75, h/6],
                [0.5, 2*h/3],
                [0.15, h/4],
                [0.85, h/4],
                [0.35, 3*h/8],
                [0.65, 3*h/8],
            ])
            points = base + 0.06 * rng.randn(11, 2)
        else:
            # Pure random
            points = []
            for _ in range(N):
                u = rng.random()
                v = rng.random()
                if u + v > 1:
                    u = 1 - u
                    v = 1 - v
                x = u + 0.5 * v
                y = h * v
                points.append([x, y])
            points = np.array(points)
        
        points = _clamp_all(points)
        current_score = _evaluate(points)
        
        if current_score > best_score:
            best_points = points.copy()
            best_score = current_score
        
        if time.time() < deadline - 3:
            sa_points, sa_score = _simulated_annealing(
                points,
                T0=0.05,
                T_end=0.0003,
                cooling=0.9998,
                max_iter=50000,
                deadline=deadline - 1
            )
            if sa_score > best_score:
                best_points = sa_points
                best_score = sa_score
    
    # Final check: ensure all points are inside the triangle
    if best_points is None:
        best_points = _clamp_all(initial_configs[0])
    else:
        best_points = _clamp_all(best_points)
    
    return best_points

# EVOLVE-BLOCK-END
