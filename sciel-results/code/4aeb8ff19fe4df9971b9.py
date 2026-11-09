import numpy as np
import time
from scipy.optimize import differential_evolution

N = 11
H = np.sqrt(3) / 2

# Precompute all C(11,3) triangle index triplets
_idx_i, _idx_j, _idx_k = [], [], []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            _idx_i.append(i)
            _idx_j.append(j)
            _idx_k.append(k)
_idx_i = np.array(_idx_i)
_idx_j = np.array(_idx_j)
_idx_k = np.array(_idx_k)
_n_triangles = len(_idx_i)

def _min_area_vec(points):
    """
    points: (N, 2) array
    Returns the minimum area of any triangle formed by 3 points.
    """
    p1 = points[_idx_i]
    p2 = points[_idx_j]
    p3 = points[_idx_k]
    
    # Area = 0.5 * | (p2-p1) x (p3-p1) |
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _params_to_points_symmetric(params):
    """
    Maps 11 parameters to 11 points in the triangle, enforcing reflection symmetry.
    params[0]: y-coordinate of the central point (x=0.5)
    params[1:]: x, t for 5 pairs. x in [0, 0.5], t in [0, 1]. y = t * sqrt(3) * x.
    """
    points = np.zeros((N, 2))
    
    # Central point
    points[0, 0] = 0.5
    points[0, 1] = params[0]
    
    # 5 pairs
    for i in range(5):
        x = params[1 + 2*i]
        t = params[2 + 2*i]
        y = t * np.sqrt(3) * x
        
        # Pair 1: (x, y) - Left side
        points[1 + 2*i, 0] = x
        points[1 + 2*i, 1] = y
        
        # Pair 2: (1-x, y) - Right side
        points[2 + 2*i, 0] = 1.0 - x
        points[2 + 2*i, 1] = y
        
    return points

def _objective_symmetric(params):
    """
    Objective function to minimize (negative of min triangle area) for symmetric configuration.
    """
    points = _params_to_points_symmetric(params)
    return -_min_area_vec(points)

def _local_polish_cartesian(points, deadline, rng, max_iters=20000):
    """
    Local polish on Cartesian coordinates.
    Tries to improve the min area by moving individual points.
    """
    best_points = points.copy()
    best_score = _min_area_vec(best_points)
    current_points = points.copy()
    current_score = best_score
    
    step_size = 0.01
    
    for it in range(max_iters):
        if time.time() > deadline - 2.0:
            break
            
        # Pick a random point
        idx = rng.randint(N)
        
        # Try a few random directions
        found_improvement = False
        for _ in range(10):
            direction = rng.randn(2)
            direction /= (np.linalg.norm(direction) + 1e-10)
            
            # Try a few step sizes
            for s in [step_size, step_size * 0.5, step_size * 0.1]:
                new_pt = current_points[idx] + direction * s
                # Check if inside triangle
                x, y = new_pt
                if y < -1e-9 or x < -1e-9 or x > 1 + 1e-9:
                    continue
                if y > np.sqrt(3) * x + 1e-9:
                    continue
                if y > np.sqrt(3) * (1 - x) + 1e-9:
                    continue
                
                old_pt = current_points[idx].copy()
                current_points[idx] = new_pt
                new_score = _min_area_vec(current_points)
                
                if new_score > current_score:
                    current_score = new_score
                    if new_score > best_score:
                        best_score = new_score
                        best_points = current_points.copy()
                    found_improvement = True
                    break
                else:
                    current_points[idx] = old_pt
            
            if found_improvement:
                break
        
        if found_improvement:
            step_size = max(0.0001, step_size * 0.95)
        else:
            step_size = max(0.0001, step_size * 0.8)
            
    return best_points, best_score

def construct():
    t_start = time.time()
    deadline = t_start + 145.0  # Safety margin

    # Phase 1: Global Search with Differential Evolution on Symmetric Parameters
    # Bounds:
    # y_0 in [0, H]
    # x_i in [0, 0.5]
    # t_i in [0, 1]
    bounds = [(0.0, H)]
    for _ in range(5):
        bounds.append((0.0, 0.5))
        bounds.append((0.0, 1.0))
        
    best_params = None
    best_score = -np.inf
    
    # Use DE to find a good symmetric configuration
    try:
        result = differential_evolution(
            _objective_symmetric, 
            bounds, 
            maxiter=100, 
            popsize=20, 
            seed=42, 
            tol=1e-6,
            polish=False,
            workers=1
        )
        best_params = result.x
        best_score = -result.fun
        
    except Exception as e:
        # Fallback: Random search in symmetric space
        rng = np.random.default_rng(42)
        best_params = np.array([rng.uniform(b[0], b[1]) for b in bounds])
        best_score = -_objective_symmetric(best_params)

    # Convert best symmetric params to Cartesian points
    points = _params_to_points_symmetric(best_params)
    current_score = best_score

    # Phase 2: Local Polish on Cartesian Coordinates
    # This allows breaking symmetry slightly if it helps, or just fine-tuning
    if time.time() < deadline - 10.0:
        rng = np.random.default_rng(123)
        polished_points, polished_score = _local_polish_cartesian(
            points, deadline, rng, max_iters=5000
        )
        if polished_score > current_score:
            points = polished_points
            current_score = polished_score

    # Phase 3: Second round of DE with different seed if time permits
    if time.time() < deadline - 20.0:
        try:
            result2 = differential_evolution(
                _objective_symmetric, 
                bounds, 
                maxiter=50, 
                popsize=15, 
                seed=999, 
                tol=1e-6,
                polish=False,
                workers=1
            )
            params2 = result2.x
            score2 = -result2.fun
            if score2 > current_score:
                points = _params_to_points_symmetric(params2)
                current_score = score2
                
                # Quick polish
                if time.time() < deadline - 5.0:
                    rng2 = np.random.default_rng(456)
                    polished_points2, polished_score2 = _local_polish_cartesian(
                        points, deadline, rng2, max_iters=2000
                    )
                    if polished_score2 > current_score:
                        points = polished_points2
                        current_score = polished_score2
        except Exception:
            pass

    # Ensure all points are inside
    for i in range(N):
        x, y = points[i]
        if y < 0: points[i, 1] = 0
        if x < 0: points[i, 0] = 0
        if x > 1: points[i, 0] = 1
        if y > np.sqrt(3) * x:
            # Project to edge
            points[i, 1] = np.sqrt(3) * x
        if y > np.sqrt(3) * (1 - x):
            points[i, 1] = np.sqrt(3) * (1 - x)

    return points
