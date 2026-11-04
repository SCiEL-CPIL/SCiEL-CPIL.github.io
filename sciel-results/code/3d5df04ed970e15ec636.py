import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, differential_evolution
import itertools

N = 14
START_TIME = time.time()
DEADLINE = START_TIME + 145.0  # Safety margin for 150s limit

def _get_triple_indices(n):
    indices = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                indices.append((i, j, k))
    return np.array(indices, dtype=int)

TRIPLE_INDICES = _get_triple_indices(N)
I_IDX = TRIPLE_INDICES[:, 0]
J_IDX = TRIPLE_INDICES[:, 1]
K_IDX = TRIPLE_INDICES[:, 2]

def _score(points):
    """Compute min triangle area / convex hull area."""
    if len(points) < 3:
        return 0.0
    
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 0.0
        
    if hull_area < 1e-12:
        return 0.0
        
    p_i = points[I_IDX]
    p_j = points[J_IDX]
    p_k = points[K_IDX]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    return min_area / hull_area

def _soft_min_score(points, beta=100.0):
    """
    Compute a smooth approximation of the minimum triangle area divided by hull area.
    Uses log-sum-exp trick for numerical stability.
    """
    if len(points) < 3:
        return 0.0
    
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 0.0
        
    if hull_area < 1e-12:
        return 0.0
        
    p_i = points[I_IDX]
    p_j = points[J_IDX]
    p_k = points[K_IDX]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # Soft min: -1/beta * log(sum(exp(-beta * areas)))
    # To avoid overflow/underflow, we shift by min
    min_a = np.min(areas)
    if min_a < 1e-15:
        return 0.0
        
    # areas_shifted = areas - min_a
    # soft_min = min_a - (1/beta) * log(sum(exp(-beta * (areas - min_a))))
    # Note: exp(-beta * (areas - min_a)) <= 1, so no overflow.
    # sum(exp(-beta * (areas - min_a))) >= 1, so log is defined.
    
    exp_terms = np.exp(-beta * (areas - min_a))
    sum_exp = np.sum(exp_terms)
    
    if sum_exp == 0:
        soft_min_area = min_a
    else:
        soft_min_area = min_a - (1.0 / beta) * np.log(sum_exp)
        
    return soft_min_area / hull_area

def _neg_soft_score(flat_points, beta=100.0):
    points = flat_points.reshape((N, 2))
    return -_soft_min_score(points, beta)

def _make_regular_polygon(n, radius=1.0, offset=0.0):
    angles = np.linspace(offset, 2*np.pi + offset, n, endpoint=False)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def _generate_ring_points(n_out, n_in, r, theta):
    """
    Generate points for a two-ring configuration.
    Outer ring: n_out points on unit circle.
    Inner ring: n_in points on circle of radius r, rotated by theta.
    """
    if n_out + n_in != N:
        return None
        
    pts_out = _make_regular_polygon(n_out, 1.0, 0.0)
    pts_in = _make_regular_polygon(n_in, r, theta)
    
    return np.vstack([pts_out, pts_in])

def _neg_ring_score(params, n_out, n_in):
    """
    Objective function for differential evolution.
    params: [r, theta]
    """
    r, theta = params
    if r < 0.01 or r > 0.99:
        return 0.0
        
    points = _generate_ring_points(n_out, n_in, r, theta)
    if points is None:
        return 0.0
        
    s = _score(points)
    return -s

def _coordinate_polish(points, time_budget, initial_beta=50.0, final_beta=1000.0, max_iters_per_stage=100):
    """
    Polish points using coordinate-wise minimization with soft-min objective.
    Starts with low beta (smoother) and increases to high beta (sharper).
    """
    start_time = time.time()
    deadline = start_time + time_budget
    
    current_points = points.copy()
    current_score = _score(current_points)
    
    # Stage 1: Low beta (smooth landscape)
    beta = initial_beta
    while time.time() < deadline - 10:
        improved = False
        for i in range(N):
            if time.time() > deadline - 10:
                break
            
            # Optimize x coordinate of point i
            def obj_x(x_val):
                pts = current_points.copy()
                pts[i, 0] = x_val
                return _neg_soft_score(pts.flatten(), beta)
            
            try:
                res = minimize(obj_x, current_points[i, 0], method='Nelder-Mead', 
                               options={'maxiter': 50, 'xatol': 1e-8, 'fatol': 1e-10})
                new_x = res.x[0]
                new_pts = current_points.copy()
                new_pts[i, 0] = new_x
                new_score = _score(new_pts)
                
                if new_score > current_score:
                    current_points = new_pts
                    current_score = new_score
                    improved = True
            except Exception:
                pass
            
            # Optimize y coordinate of point i
            def obj_y(y_val):
                pts = current_points.copy()
                pts[i, 1] = y_val
                return _neg_soft_score(pts.flatten(), beta)
            
            try:
                res = minimize(obj_y, current_points[i, 1], method='Nelder-Mead',
                               options={'maxiter': 50, 'xatol': 1e-8, 'fatol': 1e-10})
                new_y = res.x[0]
                new_pts = current_points.copy()
                new_pts[i, 1] = new_y
                new_score = _score(new_pts)
                
                if new_score > current_score:
                    current_points = new_pts
                    current_score = new_score
                    improved = True
            except Exception:
                pass
        
        if not improved:
            break
        
        # Increase beta slightly
        beta *= 1.5
        if beta > final_beta:
            beta = final_beta

    return current_score, current_points

def _optimize_ring_split(n_out, n_in, time_budget):
    """
    Optimize a specific split (n_out, n_in) using DE + Local Polish.
    Returns (score, points)
    """
    start_time = time.time()
    deadline = start_time + time_budget
    
    g = np.gcd(n_out, n_in)
    theta_max = 2 * np.pi / g
    
    bounds = [(0.05, 0.95), (0.0, theta_max)]
    
    best_score = -np.inf
    best_points = None
    
    popsize = 15
    maxiter = 50
    
    try:
        res = differential_evolution(
            lambda p: _neg_ring_score(p, n_out, n_in),
            bounds,
            seed=42,
            maxiter=maxiter,
            popsize=popsize,
            tol=1e-6,
            polish=True,
        )
        
        if res.fun < 0: # Valid score
            best_score = -res.fun
            best_params = res.x
            best_points = _generate_ring_points(n_out, n_in, best_params[0], best_params[1])
        else:
            best_params = None
            
    except Exception:
        best_params = None
        best_points = None

    if best_points is None or best_score < 0.5:
        heuristic_starts = []
        for r in [0.3, 0.5, 0.7, 0.9]:
            for theta in [0, theta_max/2, theta_max/3]:
                heuristic_starts.append([r, theta])
        
        for start in heuristic_starts:
            if time.time() > deadline - 10:
                break
            s = -_neg_ring_score(start, n_out, n_in)
            if s > best_score:
                best_score = s
                best_points = _generate_ring_points(n_out, n_in, start[0], start[1])
                best_params = start

    if best_points is not None and time.time() < deadline - 5:
        # Use coordinate polish instead of global Nelder-Mead which might be slow/unstable
        opt_score, opt_points = _coordinate_polish(best_points, time.time() - start_time + time_budget - 5)
        
        if opt_score > best_score:
            best_score = opt_score
            best_points = opt_points

    return best_score, best_points

def _optimize_three_ring(time_budget):
    """
    Optimize a three-ring configuration.
    Points distributed on 3 concentric circles.
    """
    start_time = time.time()
    deadline = start_time + time_budget
    
    # Try splits like (6, 4, 4) or (5, 5, 4)
    splits = [(6, 4, 4), (5, 5, 4), (7, 4, 3)]
    
    best_score = -np.inf
    best_points = None
    
    for split in splits:
        if time.time() > deadline - 20:
            break
            
        n1, n2, n3 = split
        if n1 + n2 + n3 != N:
            continue
            
        # Parameters: r1, r2, theta1, theta2, theta3
        # r1=1 fixed.
        # r2, r3 in (0, 1)
        # theta1, theta2, theta3 in (0, 2pi)
        
        def _neg_three_ring_score(params):
            r2, r3, t1, t2, t3 = params
            if r2 < 0.1 or r2 > 0.99 or r3 < 0.1 or r3 > 0.99:
                return 0.0
            if r2 > r3: # Ensure ordering for uniqueness, or just let it be
                pass # Actually order doesn't matter for geometry, but for DE bounds it helps
            
            pts1 = _make_regular_polygon(n1, 1.0, 0.0)
            pts2 = _make_regular_polygon(n2, r2, t1)
            pts3 = _make_regular_polygon(n3, r3, t2)
            
            points = np.vstack([pts1, pts2, pts3])
            s = _score(points)
            return -s
        
        bounds = [
            (0.1, 0.99), # r2
            (0.1, 0.99), # r3
            (0.0, 2*np.pi), # t1
            (0.0, 2*np.pi), # t2
            (0.0, 2*np.pi)  # t3
        ]
        
        try:
            res = differential_evolution(
                _neg_three_ring_score,
                bounds,
                seed=42,
                maxiter=30,
                popsize=10,
                tol=1e-5,
                polish=False # Polish is too slow for high dim
            )
            
            if res.fun < 0:
                s = -res.fun
                r2, r3, t1, t2, t3 = res.x
                pts1 = _make_regular_polygon(n1, 1.0, 0.0)
                pts2 = _make_regular_polygon(n2, r2, t1)
                pts3 = _make_regular_polygon(n3, r3, t2)
                points = np.vstack([pts1, pts2, pts3])
                
                if s > best_score:
                    best_score = s
                    best_points = points
                    
        except Exception:
            pass

    # Polish the best three-ring result
    if best_points is not None and time.time() < deadline - 5:
        opt_score, opt_points = _coordinate_polish(best_points, time.time() - start_time + time_budget - 5)
        if opt_score > best_score:
            best_score = opt_score
            best_points = opt_points

    return best_score, best_points

def construct():
    global START_TIME, DEADLINE
    START_TIME = time.time()
    DEADLINE = START_TIME + 145.0
    
    np.random.seed(42)
    
    best_global_score = -np.inf
    best_global_points = None
    
    # Baseline: Regular polygon
    reg_poly = _make_regular_polygon(N)
    s_reg = _score(reg_poly)
    best_global_score = s_reg
    best_global_points = reg_poly.copy()
    
    # 1. Try Two-Ring Optimization (7,7)
    if time.time() < DEADLINE - 60:
        s_ring, pts_ring = _optimize_ring_split(7, 7, 50.0)
        if pts_ring is not None and s_ring > best_global_score:
            best_global_score = s_ring
            best_global_points = pts_ring
            
    # 2. Try Three-Ring Optimization
    if time.time() < DEADLINE - 40:
        s_three, pts_three = _optimize_three_ring(40.0)
        if pts_three is not None and s_three > best_global_score:
            best_global_score = s_three
            best_global_points = pts_three
            
    # 3. Final Polish on the absolute best using soft-min coordinate descent
    if best_global_points is not None and time.time() < DEADLINE - 5:
        remaining_time = DEADLINE - 5 - time.time()
        final_score, final_points = _coordinate_polish(best_global_points, remaining_time)
        
        if final_score > best_global_score:
            best_global_score = final_score
            best_global_points = final_points

    return best_global_points
