"""Heilbronn triangles: maximize min triangle area for 11 points in unit-area equilateral triangle."""
import numpy as np
import random
import time
import math

N = 11
AREA = math.sqrt(3) / 4.0  # area of equilateral triangle with side 1

# Vertices of the equilateral triangle
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, math.sqrt(3) / 2.0])


def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by 3 of the points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                ax, ay = points[i]
                bx, by = points[j]
                cx, cy = points[k]
                area = 0.5 * abs((bx - ax) * (cy - ay) - (by - ay) * (cx - ax))
                if area < min_area:
                    min_area = area
    return min_area


def _is_inside_triangle(p):
    """Check if point p is inside or on the boundary of the equilateral triangle."""
    x, y = p
    if y < 0:
        return False
    if x < 0:
        return False
    if x > 1:
        return False
    # Check below the line from V0 to V2: y <= sqrt(3)*x
    if y > math.sqrt(3) * x + 1e-12:
        return False
    # Check below the line from V1 to V2: y <= sqrt(3)*(1-x)
    if y > math.sqrt(3) * (1.0 - x) + 1e-12:
        return False
    return True


def _clamp_to_triangle(p):
    """Clamp a point to be inside the triangle."""
    x, y = p
    # Clamp to the three half-planes
    if y < 0:
        y = 0.0
    if x < 0:
        x = 0.0
    if x > 1:
        x = 1.0
    # y <= sqrt(3)*x
    if y > math.sqrt(3) * x:
        # Project onto line y = sqrt(3)*x
        # The line is sqrt(3)*x - y = 0
        # Normal vector: (sqrt(3), -1), norm = 2
        d = (math.sqrt(3) * x - y) / 2.0
        x += d * math.sqrt(3)
        y -= d
    # y <= sqrt(3)*(1-x)
    if y > math.sqrt(3) * (1.0 - x):
        d = (math.sqrt(3) * (1.0 - x) - y) / 2.0
        x -= d * math.sqrt(3)
        y -= d
    # Re-clamp for numerical safety
    x = max(0.0, min(1.0, x))
    y = max(0.0, y)
    return np.array([x, y])


def _generate_structured_initial():
    """Generate a structured initial configuration based on known good patterns."""
    points = []
    # Vertices (slightly inset to avoid boundary issues)
    eps = 0.02
    points.append(V0 + eps * (V1 - V0) + eps * (V2 - V0))
    points.append(V1 + eps * (V0 - V1) + eps * (V2 - V1))
    points.append(V2 + eps * (V0 - V2) + eps * (V1 - V2))
    
    # Midpoints of edges
    points.append((V0 + V1) / 2.0)
    points.append((V1 + V2) / 2.0)
    points.append((V2 + V0) / 2.0)
    
    # Centroid
    points.append((V0 + V1 + V2) / 3.0)
    
    # Additional points: trisection points on edges
    points.append(V0 + (2.0/3.0) * (V1 - V0))
    points.append(V0 + (1.0/3.0) * (V2 - V0))
    points.append(V1 + (1.0/3.0) * (V2 - V1))
    
    # Clamp all points
    points = [_clamp_to_triangle(p) for p in points]
    return np.array(points)


def _generate_random_points(rng):
    """Generate random points inside the triangle."""
    points = []
    for _ in range(N):
        while True:
            u = rng.random()
            v = rng.random()
            if u + v < 1.0:
                p = u * V1 + v * V2
                points.append(p)
                break
    return np.array(points)


def _local_optimize(points, max_iter=2000, step_size=0.05, rng=None):
    """Coordinate descent to improve the minimum triangle area."""
    if rng is None:
        rng = random.Random()
    points = points.copy()
    current_val = _min_triangle_area(points)
    
    for it in range(max_iter):
        improved = False
        # Randomize order of points
        order = list(range(N))
        rng.shuffle(order)
        for i in order:
            best_p = points[i]
            best_val = current_val
            # Try multiple step sizes
            for ss in [step_size, step_size * 0.5, step_size * 0.1, step_size * 2.0]:
                for dx in [-ss, 0.0, ss]:
                    for dy in [-ss, 0.0, ss]:
                        if dx == 0.0 and dy == 0.0:
                            continue
                        new_p = _clamp_to_triangle(points[i] + np.array([dx, dy]))
                        points[i] = new_p
                        val = _min_triangle_area(points)
                        if val > best_val:
                            best_val = val
                            best_p = new_p
                        else:
                            points[i] = best_p
            if best_p is not points[i]:
                points[i] = best_p
                current_val = best_val
                improved = True
        if not improved:
            break
    return points, current_val


def _simulated_annealing(points, start_temp=0.1, end_temp=0.001, max_iter=5000, rng=None):
    """Simulated annealing to escape local optima."""
    if rng is None:
        rng = random.Random()
    points = points.copy()
    current_val = _min_triangle_area(points)
    best_points = points.copy()
    best_val = current_val
    
    for it in range(max_iter):
        temp = start_temp * (end_temp / start_temp) ** (it / max_iter)
        i = rng.randrange(N)
        new_p = _clamp_to_triangle(points[i] + np.array([rng.gauss(0, 0.05), rng.gauss(0, 0.05)]))
        points[i] = new_p
        new_val = _min_triangle_area(points)
        
        if new_val > current_val or rng.random() < math.exp((new_val - current_val) / max(temp, 1e-10)):
            current_val = new_val
            if new_val > best_val:
                best_val = new_val
                best_points = points.copy()
        else:
            points[i] = _clamp_to_triangle(best_points[i])  # revert
            # Actually, we should revert to the state before the move
            # Let me fix this: we need to keep the old point
            # Re-implement properly below
    
    return best_points, best_val


def _simulated_annealing_fixed(points, start_temp=0.05, end_temp=0.0005, max_iter=3000, rng=None):
    """Simulated annealing with proper state management."""
    if rng is None:
        rng = random.Random()
    points = points.copy()
    current_val = _min_triangle_area(points)
    best_points = points.copy()
    best_val = current_val
    
    for it in range(max_iter):
        temp = start_temp * (end_temp / start_temp) ** (it / max_iter)
        i = rng.randrange(N)
        old_p = points[i].copy()
        new_p = _clamp_to_triangle(old_p + np.array([rng.gauss(0, 0.04), rng.gauss(0, 0.04)]))
        points[i] = new_p
        new_val = _min_triangle_area(points)
        
        accept = False
        if new_val >= current_val:
            accept = True
        else:
            delta = new_val - current_val
            if temp > 1e-12:
                accept = rng.random() < math.exp(delta / temp)
        
        if accept:
            current_val = new_val
            if new_val > best_val:
                best_val = new_val
                best_points = points.copy()
        else:
            points[i] = old_p
    
    return best_points, best_val


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    rng = random.Random(42)
    
    best_points = None
    best_val = -float('inf')
    
    # Start with structured configuration
    points = _generate_structured_initial()
    val = _min_triangle_area(points)
    if val > best_val:
        best_val = val
        best_points = points.copy()
    
    # Coordinate descent on structured start
    points, val = _local_optimize(points, max_iter=1000, step_size=0.03, rng=rng)
    if val > best_val:
        best_val = val
        best_points = points.copy()
    
    # Simulated annealing
    if time.time() < deadline:
        points, val = _simulated_annealing_fixed(best_points, start_temp=0.03, end_temp=0.0003, max_iter=2000, rng=rng)
        if val > best_val:
            best_val = val
            best_points = points.copy()
    
    # More coordinate descent
    if time.time() < deadline:
        points, val = _local_optimize(best_points, max_iter=1500, step_size=0.02, rng=rng)
        if val > best_val:
            best_val = val
            best_points = points.copy()
    
    # Random restarts with optimization
    while time.time() < deadline - 10.0:
        points = _generate_random_points(rng)
        points, val = _local_optimize(points, max_iter=800, step_size=0.04, rng=rng)
        if val > best_val:
            best_val = val
            best_points = points.copy()
        
        if time.time() < deadline - 5.0:
            points, val = _simulated_annealing_fixed(points, start_temp=0.02, end_temp=0.0002, max_iter=1500, rng=rng)
            if val > best_val:
                best_val = val
                best_points = points.copy()
    
    # Final polish
    if time.time() < deadline:
        points, val = _local_optimize(best_points, max_iter=1000, step_size=0.01, rng=rng)
        if val > best_val:
            best_val = val
            best_points = points.copy()
    
    return best_points
