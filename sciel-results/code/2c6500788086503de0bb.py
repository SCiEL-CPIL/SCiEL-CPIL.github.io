import numpy as np
import time
from scipy.optimize import minimize
from itertools import combinations

N = 11
H = np.sqrt(3) / 2
S3 = np.sqrt(3)

# Pre-compute indices for all 165 triangles
tri_indices = list(combinations(range(N), 3))
tri_indices = np.array(tri_indices)
I = tri_indices[:, 0]
J = tri_indices[:, 1]
K = tri_indices[:, 2]

def _min_area(points):
    """
    points: (N, 2)
    Returns the minimum area of any triangle formed by 3 points.
    """
    Pi = points[I]
    Pj = points[J]
    Pk = points[K]
    
    v1 = Pj - Pi
    v2 = Pk - Pi
    
    # Cross product z-component
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def _penalty(points):
    """
    Calculate penalty for points outside the unit equilateral triangle.
    Triangle vertices: (0,0), (1,0), (0.5, H)
    """
    x = points[:, 0]
    y = points[:, 1]
    
    # Distance to boundary violations
    v1 = np.maximum(0, -y)
    v2 = np.maximum(0, y - S3 * x)
    v3 = np.maximum(0, y - S3 * (1 - x))
    
    return np.sum(v1**2 + v2**2 + v3**2)

def _objective(points):
    """
    Objective to maximize: min_area - penalty
    """
    area = _min_area(points)
    pen = _penalty(points)
    return area - 1000.0 * pen

def _smooth_objective(points):
    """
    Smooth approximation of min area using log-sum-exp.
    This provides gradient information for L-BFGS-B.
    """
    Pi = points[I]
    Pj = points[J]
    Pk = points[K]
    
    v1 = Pj - Pi
    v2 = Pk - Pi
    
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # Log-sum-exp approximation of min
    # min(x) ≈ -logsumexp(-x)
    beta = 50.0  # Sharpness parameter
    min_approx = -np.log(np.sum(np.exp(-beta * areas))) / beta
    
    pen = _penalty(points)
    return min_approx - 1000.0 * pen

def _random_valid_point(rng):
    """
    Generate a random point inside the triangle.
    """
    while True:
        u = rng.random()
        v = rng.random()
        if u + v < 1:
            x = u + 0.5 * v
            y = H * v
            return np.array([x, y])

def _initial_population(rng, count=1):
    """
    Generate initial valid configurations.
    """
    configs = []
    for _ in range(count):
        pts = []
        for _ in range(N):
            pts.append(_random_valid_point(rng))
        configs.append(np.array(pts))
    return configs

def _simulated_annealing(initial_points, time_limit, rng):
    """
    Run Simulated Annealing with adaptive step size.
    """
    current_points = initial_points.copy()
    current_score = _objective(current_points)
    
    best_points = current_points.copy()
    best_score = current_score
    
    T = 0.05  # Initial temperature
    cooling_rate = 0.9995
    min_T = 1e-6
    
    # Adaptive step size: start large, decrease
    step_scale = 0.05
    
    start_time = time.time()
    iter_count = 0
    no_improve_count = 0
    
    while T > min_T:
        if time.time() - start_time > time_limit:
            break
        
        iter_count += 1
        
        # Propose a move
        # Pick a random point to perturb
        idx = rng.integers(N)
        # Gaussian noise with adaptive scale
        noise = rng.normal(0, step_scale, size=2)
        
        new_points = current_points.copy()
        new_points[idx] += noise
        
        new_score = _objective(new_points)
        
        delta = new_score - current_score
        
        if delta > 0 or (T > min_T and rng.random() < np.exp(delta / T)):
            current_points = new_points
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1
        
        # Adaptive step size: decrease if improving, increase if stuck
        if no_improve_count > 100:
            step_scale = max(0.001, step_scale * 0.9)
            no_improve_count = 0
        elif no_improve_count == 0:
            step_scale = min(0.1, step_scale * 1.05)
        
        T *= cooling_rate
        
        # Occasionally, if stuck, do a larger jump
        if rng.random() < 0.005:
             idx = rng.integers(N)
             new_points = current_points.copy()
             new_points[idx] = _random_valid_point(rng)
             new_score = _objective(new_points)
             delta = new_score - current_score
             if delta > 0 or (T > min_T and rng.random() < np.exp(delta / T)):
                 current_points = new_points
                 current_score = new_score
                 if current_score > best_score:
                     best_score = current_score
                     best_points = current_points.copy()
    
    return best_points, best_score

def construct():
    start_time = time.time()
    timeout = 120.0  # Safety margin: evaluator kills at 150s, leave 30s buffer
    
    rng = np.random.default_rng(42)
    
    best_points = None
    best_score = -np.inf
    
    # Multi-start SA
    num_starts = 5
    time_per_start = (timeout - 25) / num_starts  # Leave 25s for polishing
    
    for i in range(num_starts):
        if time.time() - start_time > timeout - 25:
            break
            
        # Generate initial config
        init_pts = _initial_population(rng, 1)[0]
        
        # Run SA
        sa_pts, sa_score = _simulated_annealing(init_pts, time_per_start, rng)
        
        if sa_score > best_score:
            best_score = sa_score
            best_points = sa_pts
    
    # Polish the best result found using L-BFGS-B with smooth objective
    if best_points is not None and time.time() - start_time < timeout - 10:
        def _min_smooth_obj(flat_pts):
            pts = flat_pts.reshape(-1, 2)
            return -_smooth_objective(pts)
        
        try:
            res = minimize(
                _min_smooth_obj, 
                best_points.flatten(), 
                method='L-BFGS-B',
                options={'maxiter': 500, 'gtol': 1e-6}
            )
            polished_pts = res.x.reshape(-1, 2)
            polished_score = _objective(polished_pts)
            
            if polished_score > best_score:
                best_score = polished_score
                best_points = polished_pts
        except Exception:
            pass
        
        # Second polish with tighter tolerance if time remains
        if time.time() - start_time < timeout - 5:
            try:
                res2 = minimize(
                    _min_smooth_obj, 
                    best_points.flatten(), 
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'gtol': 1e-8}
                )
                polished_pts2 = res2.x.reshape(-1, 2)
                polished_score2 = _objective(polished_pts2)
                
                if polished_score2 > best_score:
                    best_score = polished_score2
                    best_points = polished_pts2
            except Exception:
                pass
    
    # Final validity check: clip points to triangle if slightly outside
    if best_points is None:
        # Fallback
        best_points = _initial_population(rng, 1)[0]
    
    # Simple clipping to ensure points are inside triangle
    x = best_points[:, 0]
    y = best_points[:, 1]
    
    # Clip y >= 0
    y = np.maximum(y, 0)
    
    # Clip y <= sqrt(3)*x
    y = np.minimum(y, S3 * x)
    
    # Clip y <= sqrt(3)*(1-x)
    y = np.minimum(y, S3 * (1 - x))
    
    best_points[:, 1] = y
    
    return best_points
