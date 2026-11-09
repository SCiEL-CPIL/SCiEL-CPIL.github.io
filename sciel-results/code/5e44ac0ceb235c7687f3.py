"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time

def _min_triangle_area(points):
    """
    Calculate the minimum area of any triangle formed by 3 points in the set.
    Points: (N, 2) array.
    """
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        pi = points[i]
        for j in range(i + 1, n):
            pj = points[j]
            if j + 1 < n:
                pk = points[j + 1:]
                v1 = pj - pi
                v2 = pk - pi
                cross = v1[0] * v2[:, 1] - v1[1] * v2[:, 0]
                areas = 0.5 * np.abs(cross)
                local_min = np.min(areas)
                if local_min < min_area:
                    min_area = local_min
    return min_area

def _hull_area(points):
    """
    Calculate the area of the convex hull of the points.
    """
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    """
    min_area = _min_triangle_area(points)
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    return min_area / hull_area

def _soft_min_triangle_area(points, beta=50.0):
    """
    Smooth surrogate for min triangle area using log-sum-exp.
    """
    n = len(points)
    all_areas = []
    for i in range(n):
        pi = points[i]
        for j in range(i + 1, n):
            pj = points[j]
            if j + 1 < n:
                pk = points[j + 1:]
                v1 = pj - pi
                v2 = pk - pi
                cross = v1[0] * v2[:, 1] - v1[1] * v2[:, 0]
                areas = 0.5 * np.abs(cross)
                all_areas.append(areas)
    all_areas = np.concatenate(all_areas)
    if len(all_areas) == 0:
        return 0.0
    # Soft min via log-sum-exp trick
    max_val = np.max(all_areas)
    soft_min = max_val - np.log(np.sum(np.exp(all_areas - max_val))) / beta
    return soft_min

def _soft_score(points, beta=50.0):
    """
    Smooth score using soft-min for triangle areas.
    """
    soft_min_area = _soft_min_triangle_area(points, beta)
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    return soft_min_area / hull_area

def _objective_full(x):
    """
    Objective function for full coordinate optimization.
    x: (2*N,) array.
    """
    points = x.reshape(N, 2)
    return -_score(points)

def _objective_soft(x, beta=50.0):
    """
    Smooth objective for gradient-based optimization.
    x: (2*N,) array.
    """
    points = x.reshape(N, 2)
    return -_soft_score(points, beta)

def _polish_with_l_bfgs(points, start_time, deadline):
    """
    High-precision polish using L-BFGS-B with smooth surrogate.
    """
    elapsed = time.time() - start_time
    remaining = deadline - start_time - elapsed
    
    if remaining < 10:
        return points, _score(points)
    
    current_score = _score(points)
    
    # Use smooth surrogate for gradient-based optimization
    def obj(x):
        return _objective_soft(x, beta=100.0)
    
    try:
        res = minimize(obj, points.flatten(), method='L-BFGS-B',
                      options={'maxiter': 500, 'maxfev': 2000, 'ftol': 1e-12, 'gtol': 1e-10})
        new_points = res.x.reshape(N, 2)
        new_score = _score(new_points)
        
        if new_score > current_score:
            return new_points, new_score
    except Exception:
        pass
    
    # Try Nelder-Mead with tighter tolerances as fallback
    if time.time() < deadline - 5:
        try:
            res_nm = minimize(_objective_full, points.flatten(),
                             method='Nelder-Mead',
                             options={'maxiter': 5000, 'maxfev': 10000,
                                     'xatol': 1e-10, 'fatol': 1e-12,
                                     'adaptive': True})
            new_points = res_nm.x.reshape(N, 2)
            new_score = _score(new_points)
            
            if new_score > current_score:
                return new_points, new_score
        except Exception:
            pass
    
    return points, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_score = 0.0
    best_points = None
    
    # Regular polygon initialization
    angles = 2 * np.pi * np.arange(N) / N
    init_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Try multiple diverse starting points
    starting_points = []
    
    # 1. Regular polygon
    starting_points.append(init_points.copy())
    
    # 2. Slightly perturbed regular polygons
    rng = np.random.RandomState(42)
    for scale in [0.05, 0.1, 0.15, 0.2, 0.3]:
        perturbed = init_points + rng.normal(0, scale, init_points.shape)
        starting_points.append(perturbed)
    
    # 3. Two concentric rings (7 points each) with different ratios
    for ratio in [0.4, 0.5, 0.6, 0.7]:
        angles_outer = 2 * np.pi * np.arange(7) / 7
        angles_inner = 2 * np.pi * np.arange(7) / 7 + np.pi / 7
        outer = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
        inner = ratio * np.column_stack([np.cos(angles_inner), np.sin(angles_inner)])
        starting_points.append(np.vstack([outer, inner]))
    
    # 4. Three-ring configuration (5+5+4)
    angles1 = 2 * np.pi * np.arange(5) / 5
    angles2 = 2 * np.pi * np.arange(5) / 5 + np.pi / 5
    angles3 = 2 * np.pi * np.arange(4) / 4 + np.pi / 4
    ring1 = np.column_stack([np.cos(angles1), np.sin(angles1)])
    ring2 = 0.6 * np.column_stack([np.cos(angles2), np.sin(angles2)])
    ring3 = 0.3 * np.column_stack([np.cos(angles3), np.sin(angles3)])
    starting_points.append(np.vstack([ring1, ring2, ring3]))
    
    # 5. Random points in unit disk
    for _ in range(4):
        r = rng.uniform(0.3, 1.0, N)
        theta = rng.uniform(0, 2 * np.pi, N)
        random_pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        starting_points.append(random_pts)
    
    # 6. Near-regular with slight radial perturbation
    for _ in range(3):
        r_pert = 1.0 + rng.normal(0, 0.1, N)
        theta_pert = 2 * np.pi * np.arange(N) / N + rng.normal(0, 0.05, N)
        perturbed = np.column_stack([r_pert * np.cos(theta_pert), r_pert * np.sin(theta_pert)])
        starting_points.append(perturbed)
    
    # Optimize each starting point with Powell
    for idx, start_pts in enumerate(starting_points):
        if time.time() > deadline - 15:
            break
        
        try:
            res = minimize(_objective_full, start_pts.flatten(), method='Powell',
                          options={'maxiter': 1500, 'maxfev': 8000})
            score = -res.fun
            points = res.x.reshape(N, 2)
            
            if score > best_score:
                best_score = score
                best_points = points.copy()
        except Exception:
            pass
    
    # If we have a good solution, try to refine it further
    if best_points is not None:
        best_points, best_score = _polish_with_l_bfgs(best_points, start_time, deadline)
    
    # Final check: if still no good solution, try radial optimization
    if best_score < 0.01 and time.time() < deadline - 5:
        def _objective_radial(r):
            angles = 2 * np.pi * np.arange(N) / N
            points = np.column_stack([r * np.cos(angles), r * np.sin(angles)])
            return -_score(points)
        
        init_r = np.ones(N)
        try:
            res_r = minimize(_objective_radial, init_r, method='Nelder-Mead',
                           options={'maxiter': 5000, 'maxfev': 5000})
            score_r = -res_r.fun
            points_r = np.column_stack([res_r.x * np.cos(angles), res_r.x * np.sin(angles)])
            
            if score_r > best_score:
                best_score = score_r
                best_points = points_r
        except Exception:
            pass
    
    if best_points is None:
        return init_points
        
    return best_points
# EVOLVE-BLOCK-END
