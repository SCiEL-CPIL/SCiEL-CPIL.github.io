import numpy as np
import time
import random
from scipy.optimize import minimize
from itertools import combinations

N = 11
H = np.sqrt(3) / 2.0

# Pre-compute all C(11,3) = 165 triangle index combinations
combos = list(combinations(range(N), 3))
idx_i = np.array([c[0] for c in combos])
idx_j = np.array([c[1] for c in combos])
idx_k = np.array([c[2] for c in combos])

def _min_triangle_area(points):
    """Calculate the minimum area of all triangles formed by the points."""
    pi = points[idx_i]
    pj = points[idx_j]
    pk = points[idx_k]
    v1 = pj - pi
    v2 = pk - pi
    # Cross product in 2D: v1[0]*v2[1] - v1[1]*v2[0]
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _project_point(p):
    """Project a single point into the equilateral triangle."""
    x, y = p
    # Barycentric coordinates for triangle with vertices (0,0), (1,0), (0.5, H)
    # v0=(0,0), v1=(1,0), v2=(0.5, H)
    # Solve: p = a*v0 + b*v1 + c*v2, a+b+c=1
    # p[0] = b + 0.5*c
    # p[1] = H*c
    # So c = p[1]/H, b = p[0] - 0.5*c, a = 1 - b - c
    
    c = y / H
    b = x - 0.5 * c
    a = 1.0 - b - c
    
    # Clip to [0, 1] and renormalize
    a = max(0.0, a)
    b = max(0.0, b)
    c = max(0.0, c)
    s = a + b + c
    if s < 1e-12:
        return np.array([0.5, H / 3.0])  # Centroid as fallback
    a /= s
    b /= s
    c /= s
    
    # Reconstruct point
    px = b * 1.0 + c * 0.5
    py = c * H
    return np.array([px, py])

def _project_points(points):
    """Project all points into the triangle."""
    return np.array([_project_point(p) for p in points])

def _objective_flat(flat):
    """Objective function: negative of min triangle area (to minimize)."""
    points = flat.reshape(N, 2)
    points = _project_points(points)
    return -_min_triangle_area(points)

def _smooth_objective_flat(flat, sigma=0.01):
    """Smoothed objective using log-sum-exp approximation."""
    points = flat.reshape(N, 2)
    points = _project_points(points)
    pi = points[idx_i]
    pj = points[idx_j]
    pk = points[idx_k]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    # Use log-sum-exp to approximate min smoothly
    min_area = np.min(areas)
    # Soft min: -log-sum-exp(-areas/sigma) * sigma
    scaled = -areas / sigma
    max_scaled = np.max(scaled)
    log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min = -log_sum_exp * sigma
    return -soft_min

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_points = None
    best_score = -1e10
    
    # Initial configurations
    def make_initial_config(seed):
        rng = random.Random(seed)
        points = []
        # Place some points near vertices and edges for better spread
        # Vertex 1
        points.append([0.05 + rng.random() * 0.05, 0.02 + rng.random() * 0.05])
        # Vertex 2
        points.append([0.95 - rng.random() * 0.05, 0.02 + rng.random() * 0.05])
        # Vertex 3
        points.append([0.5 + (rng.random() - 0.5) * 0.05, H - 0.05 - rng.random() * 0.05])
        
        # Fill remaining with random points in triangle
        while len(points) < N:
            u, v = rng.random(), rng.random()
            if u + v < 1.0:
                px = u + 0.5 * v
                py = H * v
                points.append([px, py])
        
        # Ensure all points are in triangle
        points = _project_points(np.array(points))
        return np.array(points)
    
    # Try multiple random restarts
    n_restarts = 20
    for r in range(n_restarts):
        if time.time() > deadline - 5.0:
            break
        
        # Different initial strategies
        if r < 5:
            # Strategy 1: Random with vertex bias
            points = make_initial_config(42 + r)
        elif r < 10:
            # Strategy 2: Grid-like pattern
            points = []
            # Use a triangular grid
            rows = [1, 2, 3, 5]  # 1+2+3+5 = 11
            for row_idx, count in enumerate(rows):
                y = H * (row_idx + 1) / (len(rows) + 1)
                # x range at this y: from x_min to x_max
                # At height y, the triangle width is (1 - y/H) * 1
                # Left edge: x = y / (2*H) * 1 ... actually let me compute properly
                # The left edge goes from (0,0) to (0.5, H): x = 0.5 * y / H
                # The right edge goes from (1,0) to (0.5, H): x = 1 - 0.5 * y / H
                x_left = 0.5 * y / H
                x_right = 1.0 - 0.5 * y / H
                for i in range(count):
                    if count == 1:
                        x = 0.5 * (x_left + x_right)
                    else:
                        x = x_left + (x_right - x_left) * (i + 0.5) / count
                    points.append([x, y])
            points = _project_points(np.array(points))
        else:
            # Strategy 3: Pure random
            points = make_initial_config(100 + r)
        
        # Optimize using scipy
        flat = points.flatten()
        
        # Use L-BFGS-B for optimization
        try:
            result = minimize(
                _objective_flat,
                flat,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-10}
            )
            if result.success or result.fun < _objective_flat(flat):
                opt_points = result.x.reshape(N, 2)
                opt_points = _project_points(opt_points)
                score = _min_triangle_area(opt_points)
                if score > best_score:
                    best_score = score
                    best_points = opt_points
        except Exception:
            pass
        
        # Also try with smoothed objective
        if time.time() > deadline - 5.0:
            break
        
        try:
            result2 = minimize(
                _smooth_objective_flat,
                flat,
                args=(0.005,),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-10}
            )
            if result2.fun < _objective_flat(flat):
                opt_points = result2.x.reshape(N, 2)
                opt_points = _project_points(opt_points)
                score = _min_triangle_area(opt_points)
                if score > best_score:
                    best_score = score
                    best_points = opt_points
        except Exception:
            pass
    
    # If no good result, use the random one
    if best_points is None:
        best_points = make_initial_config(7)
    
    return best_points
