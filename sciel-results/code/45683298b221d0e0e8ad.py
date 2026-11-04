import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution, minimize, minimize_scalar
import time

N = 14
SOFT_BETA = 50.0  # Temperature for soft-min; higher = sharper min approximation

# Precompute triple indices for N=14
TRIPLE_INDICES = np.array([
    (i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)
])

# Precompute for each point index, which triples it participates in
_TRIPLE_POINT_TO_IDX = [np.where((TRIPLE_INDICES == i).any(axis=1))[0] for i in range(N)]
# Precompute for each point p, the list of (triple_row, other1, other2)
_POINT_OTHERS = []
for p in range(N):
    rows = _TRIPLE_POINT_TO_IDX[p]
    others = []
    for r in rows:
        a, b, c = TRIPLE_INDICES[r]
        if p == a:
            others.append((r, b, c))
        elif p == b:
            others.append((r, a, c))
        else:
            others.append((r, a, b))
    _POINT_OTHERS.append(others)

def _calculate_score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    points: (N, 2) array
    """
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-9:
            return 0.0
    except Exception:
        return 0.0
    
    p_i = points[TRIPLE_INDICES[:, 0]]
    p_j = points[TRIPLE_INDICES[:, 1]]
    p_k = points[TRIPLE_INDICES[:, 2]]
    
    cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - \
            (p_j[:, 1] - p_i[:, 1]) * (p_k[:, 0] - p_i[:, 0])
    
    min_area = 0.5 * np.min(np.abs(cross))
    
    return min_area / hull_area

def _soft_min_score(points, beta=SOFT_BETA):
    """
    Smooth surrogate: -soft_min of triangle areas / hull area.
    Returns a smooth value that approximates -min_area/hull_area.
    """
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-9:
            return 0.0
    except Exception:
        return 0.0
    
    p_i = points[TRIPLE_INDICES[:, 0]]
    p_j = points[TRIPLE_INDICES[:, 1]]
    p_k = points[TRIPLE_INDICES[:, 2]]
    
    cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - \
            (p_j[:, 1] - p_i[:, 1]) * (p_k[:, 0] - p_i[:, 0])
    
    areas = 0.5 * np.abs(cross)
    
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    soft_min_val = (1.0 / beta) * (max_scaled + np.log(np.sum(np.exp(scaled - max_scaled))))
    
    return soft_min_val / hull_area

def _get_points_2p(n_out, n_in, r, theta):
    """
    Generate points for the 2-parameter ring configuration.
    """
    angles_out = np.linspace(0, 2*np.pi, n_out, endpoint=False)
    x_out = np.cos(angles_out)
    y_out = np.sin(angles_out)
    
    angles_in = np.linspace(0, 2*np.pi, n_in, endpoint=False) + theta
    x_in = r * np.cos(angles_in)
    y_in = r * np.sin(angles_in)
    
    return np.vstack([
        np.column_stack((x_out, y_out)),
        np.column_stack((x_in, y_in))
    ])

def _score_ring_params(n_out, n_in, params):
    """
    Objective function for differential evolution.
    params: [r, theta]
    """
    r, theta = params
    if r < 0 or r > 0.999:
        return 0.0
        
    points = _get_points_2p(n_out, n_in, r, theta)
    return _calculate_score(points)

def _soft_min_ring_score(n_out, n_in, params):
    """
    Smooth surrogate objective for ring parameters using soft-min.
    """
    r, theta = params
    if r < 0 or r > 0.999:
        return 0.0
        
    points = _get_points_2p(n_out, n_in, r, theta)
    return _soft_min_score(points)

def _local_update_polish(points, deadline):
    """
    Local-update coordinate descent: move one point at a time, only recomputing
    affected triple areas. Much faster than full recompute per evaluation.
    """
    pts = points.copy()
    score = _calculate_score(pts)
    if score <= 0:
        return pts, score
    
    # Precompute all triple areas
    p_i = pts[TRIPLE_INDICES[:, 0]]
    p_j = pts[TRIPLE_INDICES[:, 1]]
    p_k = pts[TRIPLE_INDICES[:, 2]]
    cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - \
            (p_j[:, 1] - p_i[:, 1]) * (p_k[:, 0] - p_i[:, 0])
    areas = 0.5 * np.abs(cross)
    
    # Compute hull area
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
    except Exception:
        return pts, score
    
    if hull_area <= 1e-9:
        return pts, score
    
    current_min = np.min(areas)
    current_ratio = current_min / hull_area
    
    n_iter = 0
    max_iter = 200
    step_size = 0.01
    
    for iteration in range(max_iter):
        if time.time() > deadline - 2.0:
            break
        
        improved = False
        for p in range(N):
            if time.time() > deadline - 2.0:
                break
            
            # Get affected triples
            affected_rows = _TRIPLE_POINT_TO_IDX[p]
            
            # Try 8 directions + 4 step sizes
            best_local_pts = pts[p].copy()
            best_local_ratio = current_ratio
            
            for dx in [-step_size, 0, step_size]:
                for dy in [-step_size, 0, step_size]:
                    if dx == 0 and dy == 0:
                        continue
                    
                    new_px = pts[p, 0] + dx
                    new_py = pts[p, 1] + dy
                    
                    # Recompute only affected areas
                    new_areas = areas.copy()
                    for r, other1, other2 in _POINT_OTHERS[p]:
                        pi = pts[TRIPLE_INDICES[r, 0]]
                        pj = pts[TRIPLE_INDICES[r, 1]]
                        pk = pts[TRIPLE_INDICES[r, 2]]
                        
                        if p == TRIPLE_INDICES[r, 0]:
                            c = (pj[0] - new_px) * (pk[1] - new_py) - (pj[1] - new_py) * (pk[0] - new_px)
                        elif p == TRIPLE_INDICES[r, 1]:
                            c = (pi[0] - new_px) * (pk[1] - new_py) - (pi[1] - new_py) * (pk[0] - new_px)
                        else:
                            c = (pi[0] - new_px) * (pj[1] - new_py) - (pi[1] - new_py) * (pj[0] - new_px)
                        new_areas[r] = 0.5 * abs(c)
                    
                    new_min = np.min(new_areas)
                    new_ratio = new_min / hull_area
                    
                    if new_ratio > best_local_ratio + 1e-10:
                        best_local_ratio = new_ratio
                        best_local_pts[0] = new_px
                        best_local_pts[1] = new_py
            
            if not np.allclose(best_local_pts, pts[p]):
                pts[p] = best_local_pts
                # Update affected areas
                for r, other1, other2 in _POINT_OTHERS[p]:
                    pi = pts[TRIPLE_INDICES[r, 0]]
                    pj = pts[TRIPLE_INDICES[r, 1]]
                    pk = pts[TRIPLE_INDICES[r, 2]]
                    c = (pj[0] - pi[0]) * (pk[1] - pi[1]) - (pj[1] - pi[1]) * (pk[0] - pi[0])
                    areas[r] = 0.5 * abs(c)
                
                current_min = np.min(areas)
                current_ratio = current_min / hull_area
                improved = True
        
        if not improved:
            step_size *= 0.5
            if step_size < 1e-5:
                break
    
    final_score = _calculate_score(pts)
    return pts, final_score

def _coordinate_polish(points, deadline):
    """
    Polish all 28 coordinates using multi-start L-BFGS-B with soft-min surrogate,
    plus local-update coordinate descent for fine-tuning.
    """
    if time.time() > deadline - 3.0:
        return points, _calculate_score(points)
    
    best_points = points.copy()
    best_score = _calculate_score(points)
    
    # Phase 1: L-BFGS-B with soft-min from original point
    def neg_soft_score(x):
        pts = x.reshape(-1, 2)
        return _soft_min_score(pts, beta=SOFT_BETA)
    
    try:
        res = minimize(
            neg_soft_score,
            points.flatten(),
            method='L-BFGS-B',
            options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        
        polished = res.x.reshape(-1, 2)
        polished_score = _calculate_score(polished)
        
        if polished_score > best_score:
            best_points = polished
            best_score = polished_score
            
    except Exception:
        pass
    
    # Phase 2: L-BFGS-B with higher beta
    if time.time() < deadline - 5.0:
        def neg_soft_score_high(x):
            pts = x.reshape(-1, 2)
            return _soft_min_score(pts, beta=SOFT_BETA * 5)
        
        try:
            res2 = minimize(
                neg_soft_score_high,
                best_points.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            
            polished2 = res2.reshape(-1, 2) if hasattr(res2, 'x') else res2.x.reshape(-1, 2)
            polished2_score = _calculate_score(polished2)
            
            if polished2_score > best_score:
                best_points = polished2
                best_score = polished2_score
                
        except Exception:
            pass
    
    # Phase 3: Multi-start L-BFGS-B with small random perturbations
    if time.time() < deadline - 8.0:
        np.random.seed(42)
        for trial in range(5):
            if time.time() > deadline - 8.0:
                break
            
            perturbation = np.random.normal(0, 0.02, (N, 2))
            start_pts = best_points + perturbation
            
            def neg_soft_trial(x):
                pts = x.reshape(-1, 2)
                return _soft_min_score(pts, beta=SOFT_BETA * 2)
            
            try:
                res_t = minimize(
                    neg_soft_trial,
                    start_pts.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 150, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                
                polished_t = res_t.x.reshape(-1, 2)
                polished_t_score = _calculate_score(polished_t)
                
                if polished_t_score > best_score:
                    best_points = polished_t
                    best_score = polished_t_score
                    
            except Exception:
                pass
    
    # Phase 4: Local-update coordinate descent for fine-tuning
    if time.time() < deadline - 3.0:
        local_deadline = deadline - 3.0
        best_points, best_score = _local_update_polish(best_points, local_deadline)
    
    return best_points, best_score

def _optimize_2p(n_out, n_in, time_limit, start_time, deadline):
    """
    Optimize the 2-parameter ring configuration (r, theta).
    """
    g = np.gcd(n_out, n_in)
    theta_max = 2 * np.pi / g
    
    bounds = [(0.0, 0.999), (0.0, theta_max)]
    
    best_score = 0.0
    best_params = None
    best_points = None
    
    # Phase 1: Differential Evolution with soft-min surrogate for smoothness
    popsize = 10
    maxiter = 25
    
    if time_limit > 30:
        popsize = 15
        maxiter = 40
    elif time_limit > 15:
        popsize = 12
        maxiter = 35

    try:
        res = differential_evolution(
            lambda p: _soft_min_ring_score(n_out, n_in, p),
            bounds,
            seed=42,
            maxiter=maxiter,
            popsize=popsize,
            tol=1e-5,
            polish=True
        )
        
        if res.success:
            best_params = res.x
            r, theta = best_params
            best_points = _get_points_2p(n_out, n_in, r, theta)
            best_score = _calculate_score(best_points)
    except Exception as e:
        pass
    
    # Phase 2: Local polish with multiple starts using exact score
    if time.time() < deadline - 8.0:
        starts = []
        for r_start in [0.3, 0.5, 0.7, 0.9, 0.95]:
            for theta_start in [0.0, theta_max/2, theta_max/3, 2*theta_max/3]:
                starts.append([r_start, theta_start])
                
        for start in starts:
            if time.time() > deadline - 8.0:
                break
            try:
                res_nm = minimize(
                    lambda p: -_score_ring_params(n_out, n_in, p),
                    start,
                    method='Nelder-Mead',
                    options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 150}
                )
                if -res_nm.fun > best_score:
                    best_score = -res_nm.fun
                    best_params = res_nm.x
                    r, theta = best_params
                    best_points = _get_points_2p(n_out, n_in, r, theta)
            except Exception:
                pass
    
    # Phase 3: Coordinate-level polish to escape ring parameterization
    if best_points is not None and time.time() < deadline - 5.0:
        coord_deadline = deadline - 5.0
        best_points, best_score = _coordinate_polish(best_points, coord_deadline)
        
    if best_points is None:
        return 0.0, None
        
    return best_score, best_points.flatten()

def _generate_jittered_points(params):
    """
    Generate points from jittered regular polygon model.
    params: [M(6), delta(14)]
    """
    M = params[:6].reshape(2, 3)
    delta = params[6:20]
    
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    x_base = np.cos(angles)
    y_base = np.sin(angles)
    base_points = np.column_stack((x_base, y_base))
    
    radial_scale = 1.0 + delta
    jittered_points = base_points * radial_scale[:, np.newaxis]
    
    points = jittered_points @ M[:2].T + M[:, 2]
    
    return points

def _soft_min_jittered_score(params, beta=SOFT_BETA):
    """
    Soft-min score for jittered model.
    """
    points = _generate_jittered_points(params)
    return _soft_min_score(points, beta=beta)

def _optimize_jittered(time_limit, deadline):
    """
    Optimize the jittered regular polygon model.
    """
    start_time = time.time()
    
    best_params = np.zeros(20)
    best_params[0] = 1.0
    best_params[4] = 1.0
    
    best_points = _generate_jittered_points(best_params)
    best_score = _calculate_score(best_points)
    
    def neg_soft_obj(p):
        return _soft_min_jittered_score(p, beta=SOFT_BETA)
    
    try:
        np.random.seed(123)
        restarts = 5
        for i in range(restarts):
            if time.time() > deadline - 10.0:
                break
            
            p_init = np.zeros(20)
            p_init[0] = 1.0
            p_init[4] = 1.0
            
            if i > 0:
                p_init[1] = np.random.uniform(-0.2, 0.2)
                p_init[3] = np.random.uniform(-0.2, 0.2)
                p_init[6:] = np.random.uniform(-0.1, 0.1, 14)
            
            res = minimize(
                neg_soft_obj,
                p_init,
                method='Nelder-Mead',
                options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 300}
            )
            
            if -res.fun > best_score:
                best_score = -res.fun
                best_params = res.x
                best_points = _generate_jittered_points(best_params)
                
    except Exception:
        pass
    
    # Polish the best result with exact score and higher beta soft-min
    if time.time() < deadline - 15.0:
        best_points = _generate_jittered_points(best_params)
        best_score = _calculate_score(best_points)
        
        def neg_soft_obj_high(p):
            return _soft_min_jittered_score(p, beta=SOFT_BETA * 10)
            
        try:
            res_polish = minimize(
                neg_soft_obj_high,
                best_params,
                method='Nelder-Mead',
                options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 100}
            )
            if -res_polish.fun > best_score:
                best_score = -res_polish.fun
                best_params = res_polish.x
                best_points = _generate_jittered_points(best_params)
                best_score = _calculate_score(best_points)
        except Exception:
            pass
    
    # Final coordinate-level polish on the best points
    if time.time() < deadline - 3.0:
        coord_deadline = deadline - 3.0
        best_points, best_score = _coordinate_polish(best_points, coord_deadline)
            
    return best_score, best_points

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin for 150s limit
    
    candidates = [
        (7, 7),
        (8, 6),
        (6, 8),
        (9, 5),
        (5, 9),
        (10, 4),
        (4, 10),
        (11, 3),
        (3, 11),
        (12, 2),
        (2, 12),
        (13, 1),
        (1, 13)
    ]
    
    best_overall_score = 0.0
    best_overall_points = None
    
    # Reserve time for jittered strategy
    jitter_time = 40.0
    jitter_deadline = start_time + jitter_time
    
    # Run jittered strategy first
    try:
        j_score, j_points = _optimize_jittered(jitter_time, jitter_deadline)
        if j_score > best_overall_score:
            best_overall_score = j_score
            best_overall_points = j_points.copy()
    except Exception:
        pass
    
    # Remaining time for ring strategies
    ring_deadline = deadline
    current_time = time.time()
    if current_time > ring_deadline - 10:
        if best_overall_points is None:
             n_out, n_in = 7, 7
             r, theta = 0.5, 0.0
             best_overall_points = _get_points_2p(n_out, n_in, r, theta)
             return best_overall_points
        return best_overall_points

    for n_out, n_in in candidates:
        if time.time() > ring_deadline - 10:
            break
            
        if (n_out, n_in) == (7, 7):
            t_limit = 30.0
        elif (n_out, n_in) in [(8, 6), (6, 8)]:
            t_limit = 15.0
        elif (n_out, n_in) in [(9, 5), (5, 9)]:
            t_limit = 10.0
        else:
            t_limit = 5.0
            
        score, params = _optimize_2p(n_out, n_in, t_limit, start_time, ring_deadline)
        
        if score > best_overall_score:
            best_overall_score = score
            if params is not None:
                best_overall_points = params.reshape(-1, 2).copy()
                actual_score = _calculate_score(best_overall_points)
                best_overall_score = actual_score
                
    if best_overall_points is None:
        n_out, n_in = 7, 7
        r, theta = 0.5, 0.0
        best_overall_points = _get_points_2p(n_out, n_in, r, theta)
        
    return best_overall_points
