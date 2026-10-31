"""Heilbronn convex regions: place 13 points to maximize min triangle area / convex hull area."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations
import time

def _min_triangle_area(points):
    """Compute the minimum triangle area over all triples of points."""
    n = len(points)
    min_area = np.inf
    for i, j, k in combinations(range(n), 3):
        v1 = points[j] - points[i]
        v2 = points[k] - points[i]
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        area = 0.5 * abs(cross)
        if area < min_area:
            min_area = area
    return min_area

def _convex_hull_area(points):
    """Compute the area of the convex hull of the points."""
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except Exception:
        # Fallback: compute area using shoelace formula on sorted points
        # This is a rough fallback
        return 1.0

def _score(points):
    """Compute the score: min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area <= 1e-12:
        return 0.0
    min_tri = _min_triangle_area(points)
    return min_tri / hull_area

def _regular_polygon(n, radius=1.0):
    """Generate points on a regular polygon."""
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    return radius * np.column_stack([np.cos(angles), np.sin(angles)])

def _random_points(n, seed=None):
    """Generate random points in unit square."""
    rng = np.random.RandomState(seed)
    return rng.rand(n, 2)

def _refine_local(points, max_iter=1000, step_size=0.1, deadline=None):
    """Local refinement using coordinate descent on the worst triangle."""
    points = points.copy()
    current_score = _score(points)
    best_score = current_score
    best_points = points.copy()
    
    n = len(points)
    all_triples = list(combinations(range(n), 3))
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Find the minimum area triangle
        min_area = np.inf
        min_idx = None
        for i, j, k in all_triples:
            v1 = points[j] - points[i]
            v2 = points[k] - points[i]
            cross = v1[0] * v2[1] - v1[1] * v2[0]
            area = 0.5 * abs(cross)
            if area < min_area:
                min_area = area
                min_idx = (i, j, k)
        
        if min_idx is None:
            break
        
        i, j, k = min_idx
        
        # Try to move points i, j, k to increase the minimum area
        # Use a simple gradient-like approach
        moved = False
        for idx in [i, j, k]:
            for dx in [-step_size, 0, step_size]:
                for dy in [-step_size, 0, step_size]:
                    if dx == 0 and dy == 0:
                        continue
                    new_points = points.copy()
                    new_points[idx, 0] += dx
                    new_points[idx, 1] += dy
                    
                    # Check if all points are still in reasonable bounds
                    if np.any(new_points < -10) or np.any(new_points > 10):
                        continue
                    
                    new_score = _score(new_points)
                    if new_score > best_score + 1e-10:
                        best_score = new_score
                        best_points = new_points.copy()
                        points = new_points.copy()
                        moved = True
                        break
                if moved:
                    break
            if moved:
                break
        
        if not moved:
            # Reduce step size
            step_size *= 0.8
            if step_size < 1e-6:
                break
    
    return best_points, best_score

def _anneal(points, start_temp=1.0, end_temp=0.01, max_iter=5000, deadline=None):
    """Simulated annealing to improve the configuration."""
    points = points.copy()
    current_score = _score(points)
    best_score = current_score
    best_points = points.copy()
    
    n = len(points)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        temp = start_temp * (end_temp / start_temp) ** (it / max_iter)
        
        # Random perturbation
        new_points = points + np.random.randn(n, 2) * 0.1
        
        new_score = _score(new_points)
        
        if new_score >= current_score or np.random.random() < np.exp((new_score - current_score) / max(temp, 1e-10)):
            points = new_points
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = points.copy()
    
    return best_points, best_score

def construct():
    deadline = time.time() + 110  # Leave 10s margin
    
    # Start with a regular polygon as initial guess
    best_points = _regular_polygon(N)
    best_score = _score(best_points)
    
    # Try several random starts
    for seed in range(10):
        if time.time() > deadline:
            break
        points = _random_points(N, seed=seed)
        score = _score(points)
        if score > best_score:
            best_score = score
            best_points = points.copy()
    
    # Local refinement from the best point
    refined_points, refined_score = _refine_local(best_points, max_iter=500, step_size=0.05, deadline=deadline)
    if refined_score > best_score:
        best_score = refined_score
        best_points = refined_points
    
    # Simulated annealing from the best point
    annealed_points, annealed_score = _anneal(best_points, start_temp=0.5, end_temp=0.01, max_iter=2000, deadline=deadline)
    if annealed_score > best_score:
        best_score = annealed_score
        best_points = annealed_points
    
    # Final local refinement
    final_points, final_score = _refine_local(best_points, max_iter=300, step_size=0.02, deadline=deadline)
    if final_score > best_score:
        best_score = final_score
        best_points = final_points
    
    return best_points
# EVOLVE-BLOCK-END
