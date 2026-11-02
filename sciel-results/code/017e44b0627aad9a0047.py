"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13

# EVOLVE-BLOCK-START

def _compute_score(points):
    """Compute the minimum triangle area divided by the convex hull area."""
    n = len(points)
    
    # Compute convex hull area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume  # In 2D, volume is area
    except Exception:
        # Fallback: use shoelace formula on sorted points
        # This shouldn't happen with 13 points in general position
        return 0.0
    
    if hull_area <= 1e-12:
        return 0.0
    
    # Compute minimum triangle area
    min_tri_area = float('inf')
    for i, j, k in combinations(range(n), 3):
        p1 = points[i]
        p2 = points[j]
        p3 = points[k]
        # Area of triangle using cross product
        area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - 
                         (p3[0] - p1[0]) * (p2[1] - p1[1]))
        if area < min_tri_area:
            min_tri_area = area
    
    return min_tri_area / hull_area


def _fermat_spiral(n, seed=0):
    """Generate points using Fermat's spiral (Vogel's model)."""
    rng = np.random.RandomState(seed)
    i = np.arange(n)
    # Add small random perturbation for diversity
    i_perturbed = i + rng.uniform(-0.1, 0.1, n)
    r = np.sqrt(i_perturbed / n)
    theta = i_perturbed * 2.399963229728653  # Golden angle
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    points = np.column_stack((x, y))
    # Center
    points -= np.mean(points, axis=0)
    return points


def _circle_initialization(n, seed=0):
    """Generate points on a circle with small perturbations."""
    rng = np.random.RandomState(seed)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    angles += rng.uniform(-0.1, 0.1, n)
    radii = 1.0 + rng.uniform(-0.05, 0.05, n)
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    points = np.column_stack((x, y))
    points -= np.mean(points, axis=0)
    return points


def _grid_initialization(n, seed=0):
    """Generate points in a grid pattern with perturbations."""
    rng = np.random.RandomState(seed)
    side = int(np.ceil(np.sqrt(n)))
    points = []
    count = 0
    for i in range(side):
        for j in range(side):
            if count >= n:
                break
            x = (i - side / 2 + 0.5) / side
            y = (j - side / 2 + 0.5) / side
            points.append([x + rng.uniform(-0.05, 0.05), 
                         y + rng.uniform(-0.05, 0.05)])
            count += 1
    return np.array(points[:n])


def _random_initialization(n, seed=0):
    """Generate random points in a unit square."""
    rng = np.random.RandomState(seed)
    points = rng.uniform(-0.5, 0.5, (n, 2))
    return points


def _simulated_annealing(points, max_iter=50000, t_start=0.1, t_end=0.001, 
                         cooling_rate=0.9999, rng_seed=42):
    """Simulated annealing with 2-point moves."""
    rng = np.random.RandomState(rng_seed)
    current = points.copy()
    current_score = _compute_score(current)
    best = current.copy()
    best_score = current_score
    
    for i in range(max_iter):
        # Cool temperature
        t = t_start * (cooling_rate ** i)
        if t < t_end:
            break
        
        # Pick 2 random points to move
        idx = rng.choice(len(current), size=2, replace=False)
        
        # Generate new positions with small perturbation
        new_points = current.copy()
        for idx_i in idx:
            new_points[idx_i] = current[idx_i] + rng.normal(0, 0.05, 2)
        
        new_score = _compute_score(new_points)
        
        # Accept if better or with probability exp((new - current) / t)
        if new_score > current_score:
            current = new_points
            current_score = new_score
            if current_score > best_score:
                best = current.copy()
                best_score = current_score
        else:
            delta = new_score - current_score
            if delta > 0:
                prob = np.exp(delta / t)
                if rng.random() < prob:
                    current = new_points
                    current_score = new_score
    
    return best, best_score


def _local_search(points, max_iter=1000, step_size=0.01, rng_seed=42):
    """Local search with single-point moves."""
    rng = np.random.RandomState(rng_seed)
    current = points.copy()
    current_score = _compute_score(current)
    best = current.copy()
    best_score = current_score
    
    for i in range(max_iter):
        # Pick a random point to move
        idx = rng.randint(0, len(current) - 1)
        
        # Try small perturbations
        improved = False
        for _ in range(20):
            new_points = current.copy()
            new_points[idx] = current[idx] + rng.normal(0, step_size, 2)
            new_score = _compute_score(new_points)
            
            if new_score > current_score:
                current = new_points
                current_score = new_score
                improved = True
                if current_score > best_score:
                    best = current.copy()
                    best_score = current_score
                break
        
        if not improved:
            # Reduce step size
            step_size *= 0.9
            if step_size < 1e-6:
                break
    
    return best, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Generate diverse initializations
    initializations = [
        _fermat_spiral(N, seed=0),
        _fermat_spiral(N, seed=1),
        _circle_initialization(N, seed=0),
        _circle_initialization(N, seed=1),
        _grid_initialization(N, seed=0),
        _random_initialization(N, seed=0),
        _random_initialization(N, seed=1),
    ]
    
    best_points = None
    best_score = -float('inf')
    
    # First pass: evaluate all initializations
    for init in initializations:
        score = _compute_score(init)
        if score > best_score:
            best_score = score
            best_points = init.copy()
    
    # Second pass: run SA on the best initializations
    top_inits = sorted(initializations, key=lambda p: _compute_score(p), reverse=True)[:4]
    
    for i, init in enumerate(top_inits):
        if time.time() > deadline - 20:
            break
        
        # Run SA
        sa_points, sa_score = _simulated_annealing(
            init, max_iter=30000, t_start=0.05, t_end=0.0005,
            cooling_rate=0.99995, rng_seed=100 + i
        )
        
        if sa_score > best_score:
            best_score = sa_score
            best_points = sa_points.copy()
        
        # Run local search on SA result
        if time.time() > deadline - 10:
            break
        
        ls_points, ls_score = _local_search(
            sa_points, max_iter=500, step_size=0.005, rng_seed=200 + i
        )
        
        if ls_score > best_score:
            best_score = ls_score
            best_points = ls_points.copy()
    
    # Final local search on the best point found
    if time.time() > deadline - 5:
        if best_points is not None:
            return best_points
        return initializations[0]
    
    final_points, final_score = _local_search(
        best_points, max_iter=1000, step_size=0.003, rng_seed=300
    )
    
    if final_score > best_score:
        best_points = final_points
        best_score = final_score
    
    return best_points

# EVOLVE-BLOCK-END
