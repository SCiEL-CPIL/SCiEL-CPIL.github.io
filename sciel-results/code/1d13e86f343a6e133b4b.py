"""Heilbronn triangles: maximize the minimum area of any triangle formed by 11 points in an equilateral triangle."""
import numpy as np
import time

N = 11
SQRT3 = np.sqrt(3)

# Vertices of the unit equilateral triangle
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, SQRT3 / 2.0])

def project_to_triangle(p):
    """Project point p onto the equilateral triangle defined by V0, V1, V2."""
    x, y = p[0], p[1]
    for _ in range(5):
        changed = False
        if y < 0:
            y = 0.0
            changed = True
        if y > SQRT3 * x:
            d = (SQRT3 * x - y) / 4.0
            x = x - SQRT3 * d
            y = y + d
            changed = True
        if SQRT3 * x + y > SQRT3:
            d = (SQRT3 * x + y - SQRT3) / 4.0
            x = x - SQRT3 * d
            y = y - d
            changed = True
        if not changed:
            break
    return np.array([x, y])

def min_triangle_area(points):
    """Compute the minimum area of any triangle formed by three points."""
    n = len(points)
    min_area = np.inf
    for i in range(n):
        pi = points[i]
        for j in range(i + 1, n):
            pj = points[j]
            v1x = pj[0] - pi[0]
            v1y = pj[1] - pi[1]
            for k in range(j + 1, n):
                pk = points[k]
                v2x = pk[0] - pi[0]
                v2y = pk[1] - pi[1]
                area = 0.5 * abs(v1x * v2y - v1y * v2x)
                if area < min_area:
                    min_area = area
    return min_area

def construct():
    """Construct 11 points inside the equilateral triangle to maximize the minimum triangle area."""
    
    best_points = None
    best_area = -np.inf
    
    start_time = time.time()
    time_limit = 115.0 # Leave 5s buffer for Python overhead
    
    rng = np.random.default_rng(42)
    
    # 3 restarts
    for restart in range(3):
        # Initialize points
        points = np.zeros((N, 2))
        if restart == 0:
            # Stratified random init
            for i in range(N):
                u = rng.random()
                v = rng.random()
                if u + v > 1:
                    u = 1 - u
                    v = 1 - v
                x = 0.5 * v + 0.5 - 0.5 * u
                y = (SQRT3 / 2.0) * (1.0 - u - v)
                points[i] = [x, y]
        else:
            # Random init
            for i in range(N):
                while True:
                    x = rng.random()
                    y = rng.random() * (SQRT3 / 2.0)
                    if y >= 0 and y <= SQRT3 * x and y <= SQRT3 * (1 - x):
                        points[i] = [x, y]
                        break
        
        current_points = points.copy()
        current_area = min_triangle_area(current_points)
        
        if current_area > best_area:
            best_area = current_area
            best_points = current_points.copy()
        
        # SA Parameters
        T = 0.01
        T_min = 1e-6
        steps = 40000
        alpha = (T_min / T) ** (1.0 / steps)
        
        for step in range(steps):
            if time.time() - start_time > time_limit:
                return best_points
            
            idx = rng.integers(0, N)
            step_size = 0.05 * T + 0.001
            dx = rng.normal(0, step_size)
            dy = rng.normal(0, step_size)
            
            new_points = current_points.copy()
            new_p = new_points[idx] + np.array([dx, dy])
            new_p = project_to_triangle(new_p)
            new_points[idx] = new_p
            
            new_area = min_triangle_area(new_points)
            
            if new_area > current_area:
                current_points = new_points
                current_area = new_area
                if current_area > best_area:
                    best_area = current_area
                    best_points = current_points.copy()
            else:
                delta = new_area - current_area
                if delta < 0:
                    if rng.random() < np.exp(delta / T):
                        current_points = new_points
                        current_area = new_area
            
            T *= alpha
        
        # End of restart
    
    return best_points
