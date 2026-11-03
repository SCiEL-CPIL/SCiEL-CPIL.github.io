"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize


def _min_triangle_area(points):
    """Calculate the minimum area of any triangle formed by the points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            dx12 = x2 - x1
            dy12 = y2 - y1
            for k in range(j + 1, n):
                x3, y3 = points[k]
                cross = dx12 * (y3 - y1) - dy12 * (x3 - x1)
                area = 0.5 * abs(cross)
                if area < min_area:
                    min_area = area
    return min_area


def _barycentric_to_cartesian(bary):
    """Convert barycentric coordinates to Cartesian in the equilateral triangle."""
    a, b, c = bary
    x = a + 0.5 * b
    y = (np.sqrt(3) / 2) * b
    return x, y


def _in_triangle(x, y, eps=1e-6):
    """Check if point is inside the equilateral triangle."""
    if y < -eps or y > np.sqrt(3) / 2 + eps:
        return False
    if x < -eps or x > 1 + eps:
        return False
    # Check below the top edge
    if x < 0.5:
        # Left side: y <= sqrt(3) * x
        if y > np.sqrt(3) * x + eps:
            return False
    else:
        # Right side: y <= sqrt(3) * (1 - x)
        if y > np.sqrt(3) * (1 - x) + eps:
            return False
    return True


def _clamp_to_triangle(x, y):
    """Clamp a point to be inside the equilateral triangle."""
    # First clamp to bounding box
    y = max(0.0, min(y, np.sqrt(3) / 2))
    x = max(0.0, min(x, 1.0))
    
    # Then clamp to triangle edges
    if x < 0.5:
        max_y = np.sqrt(3) * x
    else:
        max_y = np.sqrt(3) * (1 - x)
    
    if y > max_y:
        y = max_y
    
    # Final safety check
    if not _in_triangle(x, y, eps=-1e-9):
        # Project to nearest edge
        if y < 0:
            y = 0.0
        if y > np.sqrt(3) / 2:
            y = np.sqrt(3) / 2
            x = 0.5
        if x < 0:
            x = 0.0
        if x > 1:
            x = 1.0
    
    return x, y


def _objective_flat(flat_params):
    """Objective function for differential evolution (minimize negative min area)."""
    points = []
    for i in range(N):
        u = flat_params[2 * i]
        v = flat_params[2 * i + 1]
        x = u + 0.5 * v
        y = (np.sqrt(3) / 2) * v
        if not _in_triangle(x, y, eps=0.01):
            return -1e-10  # Heavy penalty for out-of-bounds
        points.append([x, y])
    
    min_area = _min_triangle_area(np.array(points))
    return -min_area


def _refine_with_l_bfgs(points, time_limit):
    """Refine points using L-BFGS-B with smooth objective."""
    start = time.time()
    
    # Convert to barycentric-like parameters (u, v)
    flat = []
    for p in points:
        v = 2 * p[1] / np.sqrt(3)
        u = p[0] - 0.5 * v
        flat.extend([u, v])
    
    def obj_smooth(flat_params):
        if time.time() - start > time_limit:
            return -1e10
        points = []
        for i in range(N):
            u = flat_params[2 * i]
            v = flat_params[2 * i + 1]
            x = u + 0.5 * v
            y = (np.sqrt(3) / 2) * v
            # Soft penalty for being outside
            if not _in_triangle(x, y, eps=0.005):
                return -1e-10
            points.append([x, y])
        
        min_area = _min_triangle_area(np.array(points))
        return -min_area
    
    bounds = [(0, 1), (0, 1)] * N
    
    try:
        result = minimize(
            obj_smooth,
            flat,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        if result.fun < 0:
            best_flat = result.x
            refined_points = []
            for i in range(N):
                u = best_flat[2 * i]
                v = best_flat[2 * i + 1]
                x = u + 0.5 * v
                y = (np.sqrt(3) / 2) * v
                x, y = _clamp_to_triangle(x, y)
                refined_points.append([x, y])
            refined_points = np.array(refined_points)
            return refined_points
    except Exception:
        pass
    
    return points


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Initial good starting point: distribute points roughly evenly
    barycentric_init = [
        [0.1, 0.1, 0.8],   # Near bottom-left
        [0.1, 0.8, 0.1],   # Near bottom-right  
        [0.8, 0.1, 0.1],   # Near top
        [0.15, 0.15, 0.7], # Bottom edge left
        [0.15, 0.7, 0.15], # Bottom edge right
        [0.7, 0.15, 0.15], # Top edge left
        [0.7, 0.15, 0.15], # Top edge right
        [0.333, 0.333, 0.334], # Center
        [0.2, 0.4, 0.4],
        [0.4, 0.2, 0.4],
        [0.4, 0.4, 0.2],
    ]
    
    initial_points = []
    for b in barycentric_init:
        x, y = _barycentric_to_cartesian(b)
        initial_points.append([x, y])
    
    initial_flat = []
    for p in initial_points:
        v = 2 * p[1] / np.sqrt(3)
        u = p[0] - 0.5 * v
        initial_flat.extend([u, v])
    
    # Define bounds for u and v
    bounds = [(0, 1), (0, 1)] * N
    
    # Track best solution found
    best_points = np.array(initial_points)
    best_score = _min_triangle_area(best_points)
    
    # Run differential evolution with time monitoring
    elapsed = time.time() - start_time
    de_time_budget = min(100, deadline - start_time - 30)  # Leave time for refinement
    
    if de_time_budget > 10:
        # Use callback to monitor time
        class TimeMonitor:
            def __init__(self):
                self.elapsed = 0
                self.last_time = time.time()
                self.timeout = False
            
            def __call__(self, x, convergence):
                current_time = time.time()
                self.elapsed += current_time - self.last_time
                self.last_time = current_time
                if self.elapsed > de_time_budget:
                    self.timeout = True
                return self.timeout
        
        monitor = TimeMonitor()
        
        try:
            result = differential_evolution(
                _objective_flat,
                bounds,
                x0=initial_flat,
                maxiter=300,
                popsize=15,
                tol=1e-8,
                mutation=(0.3, 1.0),
                recombination=0.8,
                polish=True,
                seed=42,
                callback=monitor
            )
            
            if result.fun < 0:
                best_flat = result.x
                points = []
                for i in range(N):
                    u = best_flat[2 * i]
                    v = best_flat[2 * i + 1]
                    x = u + 0.5 * v
                    y = (np.sqrt(3) / 2) * v
                    x, y = _clamp_to_triangle(x, y)
                    points.append([x, y])
                points = np.array(points)
                score = _min_triangle_area(points)
                if score > best_score:
                    best_score = score
                    best_points = points
        except Exception:
            pass
    
    # Refine with L-BFGS-B if time permits
    remaining_time = deadline - start_time
    if remaining_time > 5:
        refined = _refine_with_l_bfgs(best_points, min(remaining_time - 5, 40))
        refined_score = _min_triangle_area(refined)
        if refined_score > best_score:
            best_score = refined_score
            best_points = refined
    
    # Final safety check
    if time.time() > deadline - 2:
        return best_points
    
    return best_points
# EVOLVE-BLOCK-END
