# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time
import itertools
import sys

def compute_score(points):
    """
    Compute the Heilbronn score: A_min / A_hull.
    Returns 0 if invalid.
    """
    n = len(points)
    if n < 3:
        return 0.0
    
    # Check for validity: finite, non-collinear
    if not np.all(np.isfinite(points)):
        return 0.0
        
    try:
        hull = ConvexHull(points)
        A_hull = hull.volume
    except Exception:
        return 0.0
        
    if A_hull <= 1e-12:
        return 0.0
        
    # Vectorized area calculation for all triangles
    # points shape (n, 2)
    # We need to compute area for all combinations of 3 points.
    # Number of triangles is n*(n-1)*(n-2)/6. For n=13, this is 286.
    
    # Generate indices for all triples
    idx = np.array(list(itertools.combinations(range(n), 3)))
    p1 = points[idx[:, 0]]
    p2 = points[idx[:, 1]]
    p3 = points[idx[:, 2]]
    
    # Area = 0.5 * |x1(y2-y3) + x2(y3-y1) + x3(y1-y2)|
    areas = 0.5 * np.abs(p1[:, 0]*(p2[:, 1]-p3[:, 1]) + p2[:, 0]*(p3[:, 1]-p1[:, 1]) + p3[:, 0]*(p1[:, 1]-p2[:, 1]))
    
    A_min = np.min(areas)
    
    if A_min <= 0:
        return 0.0
        
    return A_min / A_hull


def negative_score(points):
    """
    Objective function for minimization (we minimize -score).
    """
    pts = np.array(points).reshape(13, 2)
    score = compute_score(pts)
    if score <= 0:
        return 1e6  # Large penalty for invalid
    return -score


def generate_initializations():
    """
    Generate multiple diverse initial point sets.
    Returns list of (points, initial_score) tuples.
    """
    initializations = []
    
    # 1. Concentric polygons (ring + center)
    # Outer ring: 7 points on a circle
    angles_outer = np.linspace(0, 2*np.pi, 7, endpoint=False)
    outer = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)]) * 0.8
    
    # Inner ring: 5 points on a smaller circle, rotated
    angles_inner = np.linspace(0, 2*np.pi, 5, endpoint=False) + np.pi/5
    inner = np.column_stack([np.cos(angles_inner), np.sin(angles_inner)]) * 0.4
    
    # Center point
    center = np.array([[0.0, 0.0]])
    
    pts1 = np.vstack([outer, inner, center])
    score1 = compute_score(pts1)
    initializations.append((pts1, score1))
    
    # 2. Fibonacci spiral
    n = 13
    golden_angle = np.pi * (3 - np.sqrt(5))
    fib_pts = []
    for i in range(n):
        r = np.sqrt(i / n) * 0.9
        theta = i * golden_angle
        fib_pts.append([r * np.cos(theta), r * np.sin(theta)])
    pts2 = np.array(fib_pts)
    score2 = compute_score(pts2)
    initializations.append((pts2, score2))
    
    # 3. Grid-like with perturbation
    # 4x4 grid = 16 points, take 13
    xs = np.linspace(-0.8, 0.8, 4)
    ys = np.linspace(-0.8, 0.8, 4)
    grid = np.array([[x, y] for x in xs for y in ys])
    # Shuffle and take 13
    rng = np.random.default_rng(42)
    perm = rng.permutation(16)
    pts3 = grid[perm[:13]]
    # Add small random perturbation
    pts3 = pts3 + rng.random((13, 2)) * 0.05
    score3 = compute_score(pts3)
    initializations.append((pts3, score3))
    
    # 4. Regular 13-gon
    angles = np.linspace(0, 2*np.pi, 13, endpoint=False)
    pts4 = np.column_stack([np.cos(angles), np.sin(angles)]) * 0.8
    score4 = compute_score(pts4)
    initializations.append((pts4, score4))
    
    # 5. Two concentric polygons: 6 outer, 6 inner, 1 center
    angles_outer = np.linspace(0, 2*np.pi, 6, endpoint=False)
    outer = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)]) * 0.75
    
    angles_inner = np.linspace(0, 2*np.pi, 6, endpoint=False) + np.pi/6
    inner = np.column_stack([np.cos(angles_inner), np.sin(angles_inner)]) * 0.35
    
    center = np.array([[0.0, 0.0]])
    
    pts5 = np.vstack([outer, inner, center])
    score5 = compute_score(pts5)
    initializations.append((pts5, score5))
    
    # 6. Random with seed
    rng = np.random.default_rng(123)
    pts6 = rng.random((13, 2)) * 1.6 - 0.8
    score6 = compute_score(pts6)
    initializations.append((pts6, score6))
    
    return initializations


def local_search(points, max_iter=200, step_size=0.01):
    """
    Perform local search to improve the score.
    Uses coordinate descent with adaptive step size.
    """
    best_points = points.copy()
    best_score = compute_score(best_points)
    
    n = 13
    current_step = step_size
    no_improve_count = 0
    
    for iteration in range(max_iter):
        improved = False
        for i in range(n):
            for dim in range(2):
                # Try moving point i in dimension dim
                for direction in [1, -1]:
                    new_points = best_points.copy()
                    new_points[i, dim] += direction * current_step
                    
                    new_score = compute_score(new_points)
                    
                    if new_score > best_score + 1e-10:
                        best_points = new_points
                        best_score = new_score
                        improved = True
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
        
        if not improved:
            # Reduce step size
            current_step *= 0.5
            if current_step < 1e-6:
                break
            no_improve_count = 0
        elif no_improve_count > 100:
            # Stagnation detected, reduce step size
            current_step *= 0.7
            no_improve_count = 0
    
    return best_points, best_score


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points.
    
    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    start_time = time.time()
    deadline = start_time + 600  # 600 second limit
    
    # Generate initializations
    initializations = generate_initializations()
    
    # Sort by initial score (best first)
    initializations.sort(key=lambda x: x[1], reverse=True)
    
    best_points = initializations[0][0]
    best_score = initializations[0][1]
    
    # Perform local search on each initialization, prioritizing better initial scores
    for init_points, init_score in initializations:
        if time.time() > deadline - 30:  # Safety margin
            break
        
        # Local search
        improved_points, improved_score = local_search(init_points, max_iter=150, step_size=0.02)
        
        if improved_score > best_score:
            best_points = improved_points
            best_score = improved_score
    
    # If we have time, do more refinement on the best solution
    if time.time() < deadline - 10:
        refined_points, refined_score = local_search(best_points, max_iter=300, step_size=0.01)
        if refined_score > best_score:
            best_points = refined_points
            best_score = refined_score
    
    return best_points


# EVOLVE-BLOCK-END
