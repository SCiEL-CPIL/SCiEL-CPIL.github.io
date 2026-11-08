"""Heilbronn triangles: maximize min triangle area for 11 points in unit-area equilateral triangle."""
import numpy as np
import random
import time
from itertools import combinations

N = 11
AREA = np.sqrt(3) / 4.0  # area of the equilateral triangle with side 1


def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by 3 of the points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = 0.5 * abs((points[j][0] - points[i][0]) * (points[k][1] - points[i][1])
                                 - (points[j][1] - points[i][1]) * (points[k][0] - points[i][0]))
                if area < min_area:
                    min_area = area
    return min_area


def _is_inside_triangle(p):
    """Check if point p is inside the equilateral triangle (0,0), (1,0), (0.5, sqrt(3)/2)."""
    x, y = p
    if y < 0:
        return False
    if x < 0:
        return False
    if x > 1:
        return False
    # Check below the line from (0,0) to (0.5, sqrt(3)/2): y <= sqrt(3)*x
    if y > np.sqrt(3) * x:
        return False
    # Check below the line from (1,0) to (0.5, sqrt(3)/2): y <= -sqrt(3)*(x-1)
    if y > -np.sqrt(3) * (x - 1):
        return False
    return True


def _project_to_triangle(p):
    """Project point p onto the equilateral triangle if outside."""
    x, y = p
    # Simple projection: clip to bounding box and check constraints
    # This is approximate but good enough for optimization
    if y < 0:
        y = 0
    if x < 0:
        x = 0
    if x > 1:
        x = 1
    # Check and adjust for the two slanted edges
    if y > np.sqrt(3) * x:
        # Project onto line y = sqrt(3)*x
        t = x / (1 + 3)  # not exact, just a heuristic
        # Better: find closest point on line y = sqrt(3)*x from (x,y)
        # Line: sqrt(3)*x - y = 0
        dist = (np.sqrt(3) * x - y) / np.sqrt(3 + 1)
        x_new = x - dist * np.sqrt(3)
        y_new = y + dist
        x, y = x_new, y_new
    if y > -np.sqrt(3) * (x - 1):
        # Project onto line y = -sqrt(3)*(x-1)
        dist = (y + np.sqrt(3) * (x - 1)) / np.sqrt(3 + 1)
        x_new = x + dist * np.sqrt(3)
        y_new = y - dist
        x, y = x_new, y_new
    return np.array([x, y])


def _local_optimize(points, max_iters=500, step_size=0.01):
    """Local optimization: move points to increase the minimum triangle area."""
    points = np.array(points, dtype=float)
    best_area = _min_triangle_area(points)
    
    for it in range(max_iters):
        # Find the smallest triangle
        n = len(points)
        min_area = float('inf')
        min_idx = (0, 1, 2)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    area = 0.5 * abs((points[j][0] - points[i][0]) * (points[k][1] - points[i][1])
                                     - (points[j][1] - points[i][1]) * (points[k][0] - points[i][0]))
                    if area < min_area:
                        min_area = area
                        min_idx = (i, j, k)
        
        if min_area >= best_area:
            break
        
        # Move the three vertices of the smallest triangle apart
        i, j, k = min_idx
        # Compute centroid of the triangle
        centroid = (points[i] + points[j] + points[k]) / 3.0
        
        # Move each vertex away from centroid
        for idx in [i, j, k]:
            direction = points[idx] - centroid
            norm = np.linalg.norm(direction)
            if norm > 1e-10:
                direction = direction / norm
            else:
                direction = np.random.randn(2)
                direction = direction / np.linalg.norm(direction)
            
            # Try moving in the direction away from centroid
            new_point = points[idx] + step_size * direction
            if _is_inside_triangle(new_point):
                points[idx] = new_point
            else:
                # Try smaller step or different direction
                new_point = points[idx] + step_size * 0.5 * direction
                if _is_inside_triangle(new_point):
                    points[idx] = new_point
                else:
                    # Try random direction
                    for _ in range(5):
                        rand_dir = np.random.randn(2)
                        rand_dir = rand_dir / np.linalg.norm(rand_dir)
                        new_point = points[idx] + step_size * rand_dir
                        if _is_inside_triangle(new_point):
                            points[idx] = new_point
                            break
        
        # Also try moving other points slightly randomly to escape local minima
        if it % 10 == 0:
            for idx in range(n):
                if random.random() < 0.3:
                    new_point = points[idx] + step_size * 0.5 * np.random.randn(2)
                    if _is_inside_triangle(new_point):
                        points[idx] = new_point
        
        # Update best
        current_area = _min_triangle_area(points)
        if current_area > best_area:
            best_area = current_area
    
    return points, best_area


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 140s deadline with 10s safety margin
    
    best_points = None
    best_score = 0.0
    
    # Use multiple random restarts
    rng = random.Random(42)
    
    for restart in range(50):
        if time.time() > deadline:
            break
        
        # Generate random points inside the triangle
        points = []
        while len(points) < N:
            u, v = rng.random(), rng.random()
            if u + v < 1.0 and u > 0.005 and v > 0.005:
                x = u + 0.5 * v
                y = (np.sqrt(3) / 2.0) * v
                points.append([x, y])
        
        points = np.array(points, dtype=float)
        
        # Run local optimization
        opt_points, opt_score = _local_optimize(points, max_iters=300, step_size=0.005)
        
        if opt_score > best_score:
            best_score = opt_score
            best_points = opt_points.copy()
        
        # Check time
        if time.time() > deadline:
            break
    
    if best_points is None:
        # Fallback: return random points
        points = []
        rng2 = random.Random(7)
        while len(points) < N:
            u, v = rng2.random(), rng2.random()
            if u + v < 0.98 and u > 0.01 and v > 0.01:
                points.append([u + 0.5 * v, (3 ** 0.5) / 2 * v])
        return np.array(points)
    
    return best_points
# EVOLVE-BLOCK-END
