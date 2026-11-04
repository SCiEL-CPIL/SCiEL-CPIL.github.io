"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from itertools import combinations
import time

def _min_triangle_area_fast(points):
    """Compute the minimum area of any triangle formed by 3 points using vectorized ops."""
    n = len(points)
    # Compute all pairwise cross products
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            # Vectorized: compute area of triangles (i, j, k) for all k
            p1 = points[i]
            p2 = points[j]
            # Cross product for all k
            cross = (p2[0] - p1[0]) * (points[:, 1] - p1[1]) - (p2[1] - p1[1]) * (points[:, 0] - p1[0])
            areas = 0.5 * np.abs(cross)
            # Exclude i and j themselves (area 0)
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            mask[j] = False
            if np.any(mask):
                local_min = np.min(areas[mask])
                if local_min < min_area:
                    min_area = local_min
    return min_area

def _convex_hull_area_fast(points):
    """Compute the area of the convex hull of the given points."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """Compute the score: min triangle area / convex hull area."""
    min_area = _min_triangle_area_fast(points)
    hull_area = _convex_hull_area_fast(points)
    if hull_area < 1e-10:
        return 0.0
    return min_area / hull_area

def _regular_polygon(n, radius=1.0):
    """Generate a regular n-gon."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def _jittered_grid(n, jitter=0.1):
    """Generate a jittered grid configuration."""
    # Place points in a roughly square grid
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    points = []
    for i in range(rows):
        for j in range(cols):
            if len(points) < n:
                x = (j + 0.5) / cols
                y = (i + 0.5) / rows
                points.append([x, y])
    points = np.array(points[:n])
    # Add jitter
    np.random.seed(42)
    points += np.random.randn(n, 2) * jitter
    return points

def _circular_with_inner(n, n_inner=5):
    """Generate points with some on a circle and some inside."""
    n_outer = n - n_inner
    # Outer ring
    outer_angles = np.linspace(0, 2 * np.pi, n_outer, endpoint=False)
    outer = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
    # Inner points
    inner = np.random.randn(n_inner, 2) * 0.3
    return np.vstack([outer, inner])

def _local_refine(points, max_iter=100, step=0.05):
    """Local refinement using coordinate descent."""
    points = points.copy()
    n = len(points)
    best_score = _score(points)
    
    for _ in range(max_iter):
        improved = False
        for i in range(n):
            for coord in range(2):
                # Try moving point i in each direction
                for direction in [-1, 1]:
                    for step_size in [step, step * 0.5, step * 0.25]:
                        new_points = points.copy()
                        new_points[i, coord] += direction * step_size
                        new_score = _score(new_points)
                        if new_score > best_score:
                            best_score = new_score
                            points = new_points
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        if not improved:
            break
    return points

def _genetic_algorithm(pop_size=20, generations=30, mutation_rate=0.1, elite_size=5):
    """Genetic algorithm to optimize point configuration."""
    # Initialize population with diverse configurations
    population = []
    
    # Add regular polygon
    population.append(_regular_polygon(N))
    
    # Add jittered grid
    population.append(_jittered_grid(N))
    
    # Add circular with inner
    np.random.seed(123)
    population.append(_circular_with_inner(N))
    
    # Add random configurations
    for i in range(pop_size - 3):
        np.random.seed(1000 + i)
        pts = np.random.randn(N, 2)
        # Normalize to unit area
        hull_area = _convex_hull_area_fast(pts)
        if hull_area > 1e-10:
            pts = pts / np.sqrt(hull_area)
        population.append(pts)
    
    # Evaluate initial population
    scores = [_score(pts) for pts in population]
    
    for gen in range(generations):
        # Sort by score
        order = np.argsort(scores)[::-1]
        population = [population[i] for i in order]
        scores = [scores[i] for i in order]
        
        # Elitism: keep best
        new_population = population[:elite_size]
        
        # Generate offspring
        while len(new_population) < pop_size:
            # Tournament selection
            idx1 = np.random.randint(0, pop_size)
            idx2 = np.random.randint(0, pop_size)
            if scores[idx1] >= scores[idx2]:
                parent1 = population[idx1]
            else:
                parent1 = population[idx2]
            
            idx1 = np.random.randint(0, pop_size)
            idx2 = np.random.randint(0, pop_size)
            if scores[idx1] >= scores[idx2]:
                parent2 = population[idx1]
            else:
                parent2 = population[idx2]
            
            # Crossover: mix points from parents
            child = parent1.copy()
            for i in range(N):
                if np.random.random() < 0.5:
                    child[i] = parent2[i]
            
            # Mutation
            for i in range(N):
                if np.random.random() < mutation_rate:
                    child[i] += np.random.randn(2) * 0.1
            
            # Normalize to unit area
            hull_area = _convex_hull_area_fast(child)
            if hull_area > 1e-10:
                child = child / np.sqrt(hull_area)
            
            new_population.append(child)
        
        population = new_population
        scores = [_score(pts) for pts in population]
        
        # Local refine the best few
        for i in range(min(3, len(population))):
            refined = _local_refine(population[i], max_iter=50, step=0.03)
            refined_score = _score(refined)
            if refined_score > scores[i]:
                population[i] = refined
                scores[i] = refined_score
    
    # Return best
    best_idx = np.argmax(scores)
    return population[best_idx], scores[best_idx]

def construct():
    """Construct the optimal point configuration."""
    start_time = time.time()
    time_limit = 110  # Leave 10s buffer
    
    # Run genetic algorithm
    best_points, best_score = _genetic_algorithm(pop_size=15, generations=20, mutation_rate=0.15, elite_size=5)
    
    # Check time
    if time.time() - start_time > time_limit:
        return best_points
    
    # Local refine the best solution
    refined_points = _local_refine(best_points, max_iter=200, step=0.05)
    refined_score = _score(refined_points)
    
    if refined_score > best_score:
        best_points = refined_points
        best_score = refined_score
    
    # Try a few more local refinements from different starting points
    np.random.seed(999)
    for trial in range(3):
        if time.time() - start_time > time_limit:
            break
        # Start from perturbed best
        trial_points = best_points + np.random.randn(N, 2) * 0.1
        trial_points = _local_refine(trial_points, max_iter=100, step=0.03)
        trial_score = _score(trial_points)
        if trial_score > best_score:
            best_points = trial_points
            best_score = trial_score
    
    return best_points
# EVOLVE-BLOCK-END
