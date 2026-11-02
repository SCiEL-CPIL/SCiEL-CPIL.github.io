"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

S3H = np.sqrt(3) / 2
TRI_AREA = S3H / 2.0
DEADLINE = 140  # seconds, safety margin from 150s limit


def _min_area_normalized(points):
    """Compute minimum triangle area normalized by the unit equilateral triangle area."""
    n = len(points)
    if n < 3:
        return 0.0
    
    min_a = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                dx1 = points[j, 0] - points[i, 0]
                dy1 = points[j, 1] - points[i, 1]
                dx2 = points[k, 0] - points[i, 0]
                dy2 = points[k, 1] - points[i, 1]
                cross = abs(dx1 * dy2 - dy1 * dx2) * 0.5
                if cross < min_a:
                    min_a = cross
    return min_a / TRI_AREA


def _is_inside_triangle(p):
    """Check if point p is inside the equilateral triangle (with small tolerance)."""
    x, y = p
    eps = 1e-10
    if y < -eps or y > S3H + eps:
        return False
    if x < -eps or x > 1 + eps:
        return False
    if y > S3H * (1 - x) + eps:
        return False
    if y > S3H * x + eps:
        return False
    return True


def _penalized_min_area(x):
    """Objective for optimization: minimize negative min area plus penalty for out-of-bounds."""
    points = x.reshape(-1, 2)
    
    # Penalty for points outside triangle
    penalty = 0.0
    for i in range(N):
        px, py = points[i]
        # Distances to boundaries (positive if outside)
        d_bottom = -py
        d_top = py - S3H
        d_left = -px
        d_right = px - 1
        d_slant_right = py - S3H * (1 - px)
        d_slant_left = py - S3H * px
        
        dist = max(0, d_bottom, d_top, d_left, d_right, d_slant_right, d_slant_left)
        penalty += dist ** 2
    
    min_a = _min_area_normalized(points)
    return -min_a + 100.0 * penalty


def _local_polish(points, start_time, deadline_margin=5):
    """Polish a set of points using Nelder-Mead."""
    if time.time() - start_time > DEADLINE - deadline_margin:
        return points, _min_area_normalized(points)
    
    best_points = points.copy()
    best_score = _min_area_normalized(points)
    
    try:
        res = minimize(
            _penalized_min_area,
            best_points.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
        )
        if time.time() - start_time > DEADLINE - deadline_margin:
            return best_points, best_score
        
        pts = res.x.reshape(-1, 2)
        valid = all(_is_inside_triangle(p) for p in pts)
        if valid:
            score = _min_area_normalized(pts)
            if score > best_score:
                best_points = pts
                best_score = score
    except Exception:
        pass
    
    return best_points, best_score


def construct():
    start_time = time.time()
    
    # Define bounds for differential evolution
    # The triangle has vertices (0,0), (1,0), (0.5, sqrt(3)/2)
    # x ranges from 0 to 1
    # y ranges from 0 to sqrt(3)/2
    bounds = []
    for i in range(N):
        bounds.append((0, 1))  # x
        bounds.append((0, S3H))  # y
    
    best_points = None
    best_score = -float('inf')
    
    # Run differential evolution with time limit
    try:
        # Use a callback to check time
        def callback(xk, convergence=None):
            return time.time() - start_time > DEADLINE - 20
        
        result = differential_evolution(
            _penalized_min_area,
            bounds,
            maxiter=100,
            popsize=15,
            tol=1e-6,
            mutation=(0.5, 1.0),
            recombination=0.7,
            seed=42,
            callback=callback,
            polish=True,
            init='latinhypercube'
        )
        
        if time.time() - start_time < DEADLINE - 5:
            pts = result.x.reshape(-1, 2)
            valid = all(_is_inside_triangle(p) for p in pts)
            if valid:
                score = _min_area_normalized(pts)
                if score > best_score:
                    best_score = score
                    best_points = pts.copy()
    except Exception:
        pass
    
    # If we didn't get a good result from DE, try some random restarts with local polish
    if best_points is None or time.time() - start_time < DEADLINE - 15:
        rng = np.random.RandomState(42)
        
        def random_point_in_triangle():
            # Rejection sampling
            while True:
                u, v = rng.random(), rng.random()
                if u + v < 1:
                    x = u + 0.5 * v
                    y = S3H * v
                    if 0.01 < x < 0.99 and 0.01 < y < S3H * 0.99:
                        if y < S3H * (1 - x) * 0.99 and y < S3H * x * 0.99:
                            return [x, y]
        
        n_restarts = 20
        for r in range(n_restarts):
            if time.time() - start_time > DEADLINE - 10:
                break
            
            x0 = np.array([random_point_in_triangle() for _ in range(N)]).flatten()
            pts, score = _local_polish(x0.reshape(-1, 2), start_time)
            if score > best_score:
                best_score = score
                best_points = pts
    
    # Final polish on the best candidate if time permits
    if best_points is not None and time.time() - start_time < DEADLINE - 3:
        best_points, best_score = _local_polish(best_points, start_time, deadline_margin=3)
    
    if best_points is None:
        # Fallback: return a valid random configuration
        rng = np.random.RandomState(0)
        def random_point_in_triangle_fallback():
            while True:
                u, v = rng.random(), rng.random()
                if u + v < 1:
                    x = u + 0.5 * v
                    y = S3H * v
                    if 0.001 < x < 0.999 and 0.001 < y < S3H * 0.999:
                        if y < S3H * (1 - x) * 0.999 and y < S3H * x * 0.999:
                            return [x, y]
        best_points = np.array([random_point_in_triangle_fallback() for _ in range(N)])
    
    return best_points
# EVOLVE-BLOCK-END
