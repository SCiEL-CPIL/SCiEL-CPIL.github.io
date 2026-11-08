"""Heilbronn triangles: maximize min triangle area for 11 points in unit-area equilateral triangle."""
import numpy as np
import time
from itertools import combinations

N = 11
SQRT3 = np.sqrt(3.0)
H = SQRT3 / 2.0  # height of the equilateral triangle with side 1

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, H])

# Precompute triangle indices for vectorized area computation
TRI_IDX = np.array(list(combinations(range(N), 3)))


def min_triangle_area(points):
    """Compute the minimum area over all C(n,3) triangles using vectorization."""
    pi = points[TRI_IDX[:, 0]]
    pj = points[TRI_IDX[:, 1]]
    pk = points[TRI_IDX[:, 2]]
    
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return float(np.min(areas))


def is_inside_triangle(p, eps=1e-9):
    """Check if point p is inside the equilateral triangle."""
    x, y = p
    if y < -eps or x < -eps or x > 1.0 + eps:
        return False
    if y > SQRT3 * x + eps:
        return False
    if y > SQRT3 * (1.0 - x) + eps:
        return False
    return True


def project_to_triangle(p):
    """Project a point onto the triangle if it's outside."""
    x, y = p
    # Clip y
    if y < 0:
        y = 0.0
    if y > H:
        y = H
    # Clip x based on y
    x_min = y / SQRT3
    x_max = 1.0 - y / SQRT3
    if x < x_min:
        x = x_min
    if x > x_max:
        x = x_max
    return np.array([x, y])


def generate_structured_initial():
    """Generate a structured initial guess with good symmetry."""
    points = []
    # Vertices
    points.append([0.0, 0.0])
    points.append([1.0, 0.0])
    points.append([0.5, H])
    
    # Edge midpoints
    points.append([0.5, 0.0])
    points.append([0.25, H/2])
    points.append([0.75, H/2])
    
    # Center
    points.append([0.5, H/3])
    
    # Additional points
    points.append([0.25, 0.0])
    points.append([0.75, 0.0])
    points.append([0.35, H/4])
    points.append([0.65, H/4])
    
    return np.array(points[:N])


def generate_random_points(rng):
    """Generate random points inside the triangle."""
    points = []
    for _ in range(N):
        u = rng.random()
        v = rng.random()
        if u + v > 1.0:
            u = 1.0 - u
            v = 1.0 - v
        p = (1.0 - u - v) * V0 + u * V1 + v * V2
        points.append(p)
    return np.array(points)


def local_optimize(points, rng, max_iter=500, step_size=0.05, deadline=None):
    """Local optimization using coordinate descent / hill climbing."""
    points = points.copy()
    best_val = min_triangle_area(points)
    
    for iteration in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
            
        improved = False
        for i in range(N):
            # Try moving point i in different directions
            best_local = points[i].copy()
            best_local_val = best_val
            
            for _ in range(30):
                if deadline is not None and time.time() > deadline:
                    break
                    
                # Random perturbation
                dx = rng.uniform(-step_size, step_size, 2)
                new_p = best_local + dx
                new_p = project_to_triangle(new_p)
                
                if not is_inside_triangle(new_p, eps=1e-6):
                    continue
                    
                new_points = points.copy()
                new_points[i] = new_p
                
                val = min_triangle_area(new_points)
                if val > best_local_val + 1e-10:
                    best_local_val = val
                    best_local = new_p
                    improved = True
            
            points[i] = best_local
        
        # Update best value
        current_val = min_triangle_area(points)
        if current_val > best_val:
            best_val = current_val
        
        # If no improvement in a full pass, reduce step size
        if not improved:
            step_size *= 0.85
            if step_size < 1e-7:
                break
    
    return points, best_val


def simulated_annealing(points, rng, deadline, initial_temp=0.1, cooling=0.9995, max_iter=10000):
    """Simulated annealing to escape local optima."""
    points = points.copy()
    best_points = points.copy()
    best_val = min_triangle_area(points)
    current_val = best_val
    temp = initial_temp
    
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
            
        # Pick a random point to move
        i = rng.randint(N)
        dx = rng.uniform(-0.05, 0.05, 2)
        new_p = points[i] + dx
        new_p = project_to_triangle(new_p)
        
        if not is_inside_triangle(new_p, eps=1e-6):
            continue
        
        new_points = points.copy()
        new_points[i] = new_p
        
        new_val = min_triangle_area(new_points)
        delta = new_val - current_val
        
        if delta > 0 or rng.random() < np.exp(delta / max(temp, 1e-10)):
            points = new_points
            current_val = new_val
            if current_val > best_val:
                best_val = current_val
                best_points = points.copy()
        
        temp *= cooling
    
    return best_points, best_val


def construct():
    """Construct 11 points in the equilateral triangle to maximize the minimum triangle area."""
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    rng = np.random.RandomState(42)
    
    best_points = None
    best_value = -1
    
    # Try multiple strategies
    strategies = [
        ("structured", generate_structured_initial()),
    ]
    
    # Add random initializations
    for seed in range(10):
        strategies.append((f"random_{seed}", generate_random_points(np.random.RandomState(seed * 42 + 7))))
    
    for name, init_points in strategies:
        if time.time() > deadline - 20:
            break
        
        # First, do local optimization
        opt_points, opt_val = local_optimize(init_points, rng, max_iter=300, step_size=0.08, deadline=deadline - 5)
        
        if opt_val > best_value:
            best_value = opt_val
            best_points = opt_points.copy()
        
        # If we have time, try simulated annealing from the best point
        if time.time() < deadline - 25:
            sa_points, sa_val = simulated_annealing(opt_points, rng, deadline, initial_temp=0.05, cooling=0.9998, max_iter=5000)
            
            if sa_val > best_value:
                best_value = sa_val
                best_points = sa_points.copy()
    
    # Final local optimization pass on the best found
    if best_points is not None and time.time() < deadline - 10:
        final_points, final_val = local_optimize(best_points, rng, max_iter=500, step_size=0.03, deadline=deadline - 2)
        if final_val > best_value:
            best_value = final_val
            best_points = final_points.copy()
    
    # Verify all points are inside the triangle
    if best_points is not None:
        for i, p in enumerate(best_points):
            if not is_inside_triangle(p, eps=1e-6):
                best_points[i] = project_to_triangle(p)
    
    if best_points is None:
        # Fallback
        best_points = generate_structured_initial()
    
    return best_points
