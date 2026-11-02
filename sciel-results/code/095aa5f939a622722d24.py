import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time
from itertools import combinations

N = 13

def _triangle_area(p1, p2, p3):
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))

def _min_triangle_area(points):
    # points is (N, 2)
    min_area = float('inf')
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                area = _triangle_area(points[i], points[j], points[k])
                if area < min_area:
                    min_area = area
    return min_area

def _hull_area(points):
    # points is (N, 2)
    if len(points) < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        return hull.volume # In 2D, volume is area
    except:
        # Fallback if points are collinear or degenerate
        return 0.0

def _score(points):
    hull_area = _hull_area(points)
    if hull_area <= 1e-10:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area

def _neg_score(points_flat):
    points = points_flat.reshape(-1, 2)
    return -_score(points)

def _all_triangle_areas(points):
    """Compute all triangle areas for a point set."""
    n = len(points)
    areas = np.array([
        _triangle_area(points[i], points[j], points[k])
        for i in range(n) for j in range(i+1, n) for k in range(j+1, n)
    ])
    return areas

def _logsumexp_surrogate(points, temperature):
    """
    Smooth surrogate using log-sum-exp of negative areas.
    As temperature -> 0, this approaches -min_area.
    """
    areas = _all_triangle_areas(points)
    if len(areas) == 0:
        return 0.0
    # Use log-sum-exp for numerical stability
    # We want to maximize min area, so we minimize -log(sum(exp(-area/T)))
    # This is approximately -min_area for small T
    scaled = -areas / temperature
    max_scaled = np.max(scaled)
    if not np.isfinite(max_scaled):
        return 0.0
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    return -log_sum * temperature  # This approximates -min_area

def _surrogate_score(points, temperature):
    """
    Score based on log-sum-exp surrogate normalized by hull area.
    """
    hull_area = _hull_area(points)
    if hull_area <= 1e-10:
        return 0.0
    surrogate_min = _logsumexp_surrogate(points, temperature)
    # surrogate_min approximates -min_area, so min_area_approx = -surrogate_min
    min_area_approx = -surrogate_min
    if min_area_approx <= 0:
        return 0.0
    return min_area_approx / hull_area

def _neg_surrogate_score(points_flat, temperature):
    points = points_flat.reshape(-1, 2)
    return -_surrogate_score(points, temperature)

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s safety margin before 150s kill
    
    # Initial guess: Regular 13-gon
    theta = np.linspace(0, 2*np.pi, N, endpoint=False)
    x = np.cos(theta)
    y = np.sin(theta)
    initial_points = np.column_stack((x, y))
    initial_flat = initial_points.flatten()
    
    best_score = _score(initial_points)
    best_points = initial_points.copy()
    
    # Phase 1: Quick local optimization to get a good starting point
    try:
        res = minimize(_neg_score, initial_flat, method='Nelder-Mead', 
                       options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-6})
        if res.fun < -best_score:
            candidate_points = res.x.reshape(-1, 2)
            cand_score = _score(candidate_points)
            if cand_score > best_score:
                best_score = cand_score
                best_points = candidate_points.copy()
    except:
        pass
    
    if time.time() > deadline - 20:
        return best_points
    
    # Phase 2: Simulated Annealing with log-sum-exp surrogate
    rng = np.random.default_rng(42)
    
    current_points = best_points.copy()
    current_score = best_score
    
    # SA parameters
    T_initial = 0.05
    T_final = 0.001
    n_iterations = 5000
    step_size_initial = 0.15
    step_size_final = 0.01
    
    for it in range(n_iterations):
        if time.time() > deadline - 5:
            break
        
        # Annealing schedule
        progress = it / n_iterations
        T = T_initial * (T_final / T_initial) ** progress
        step_size = step_size_initial * (step_size_final / step_size_initial) ** progress
        
        # Generate neighbor
        perturbation = rng.normal(0, step_size, N*2)
        neighbor_flat = current_flat = current_points.flatten() + perturbation
        neighbor_points = neighbor_flat.reshape(-1, 2)
        
        # Evaluate neighbor using surrogate (faster) for acceptance, exact for tracking
        neighbor_surrogate = _surrogate_score(neighbor_points, T)
        
        # Acceptance criterion
        if neighbor_surrogate >= current_score or rng.random() < np.exp((neighbor_surrogate - current_score) / T):
            current_points = neighbor_points
            # Track with exact score periodically or when improving
            if it % 10 == 0 or neighbor_surrogate > current_score:
                exact_score = _score(current_points)
                current_score = exact_score
                if exact_score > best_score:
                    best_score = exact_score
                    best_points = current_points.copy()
    
    # Phase 3: Local polish with Nelder-Mead from best SA result
    if time.time() < deadline - 10:
        try:
            res = minimize(_neg_score, best_points.flatten(), method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8})
            candidate_points = res.x.reshape(-1, 2)
            cand_score = _score(candidate_points)
            if cand_score > best_score:
                best_score = cand_score
                best_points = candidate_points
        except:
            pass
    
    # Phase 4: A few more random restarts with SA if time permits
    if time.time() < deadline - 15:
        for restart in range(5):
            if time.time() > deadline - 5:
                break
            
            # Start from perturbed best
            start_points = best_points + rng.normal(0, 0.05, (N, 2))
            start_score = _score(start_points)
            
            # Short SA
            T_start = 0.03
            T_end = 0.001
            n_iter = 2000
            ss_start = 0.1
            ss_end = 0.005
            
            cur_pts = start_points.copy()
            cur_score = start_score
            
            for it in range(n_iter):
                if time.time() > deadline - 3:
                    break
                
                progress = it / n_iter
                T = T_start * (T_end / T_start) ** progress
                ss = ss_start * (ss_end / ss_start) ** progress
                
                pert = rng.normal(0, ss, N*2)
                nb_pts = cur_pts.flatten() + pert
                nb_pts = nb_pts.reshape(N, 2)
                
                nb_surr = _surrogate_score(nb_pts, T)
                
                if nb_surr >= cur_score or rng.random() < np.exp((nb_surr - cur_score) / T):
                    cur_pts = nb_pts
                    if it % 10 == 0 or nb_surr > cur_score:
                        exact = _score(cur_pts)
                        cur_score = exact
                        if exact > best_score:
                            best_score = exact
                            best_points = cur_pts.copy()
            
            # Polish
            if time.time() < deadline - 5:
                try:
                    res = minimize(_neg_score, best_points.flatten(), method='Nelder-Mead',
                                   options={'maxiter': 200, 'xatol': 1e-7, 'fatol': 1e-7})
                    cand = res.x.reshape(N, 2)
                    cs = _score(cand)
                    if cs > best_score:
                        best_score = cs
                        best_points = cand
                except:
                    pass
    
    return best_points
