import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, minimize_scalar

N = 14

def _compute_areas(points):
    """
    Compute areas of all triangles formed by triples of points.
    Returns a 1D array of length C(N, 3).
    """
    n = len(points)
    # Pre-compute all pair differences? No, just vectorize the triple loop.
    # For N=14, C(14,3)=364. 
    # We can use broadcasting.
    # Shape (N, 1, 1, 2) * (1, N, 1, 2) ... too complex.
    # Let's stick to the efficient semi-vectorized or simple loop if called frequently.
    # Actually, for optimization, we call this many times.
    # Let's use a fully vectorized approach with indices.
    
    # Indices for i < j < k
    idx = np.array([list(range(i+1, k+1)) for i in range(n) for k in range(i+2, n)])
    # This creates a list of lists of pairs (j,k) for each i? No.
    
    # Simpler vectorized way:
    i, j, k = np.indices((n, n, n))
    mask = (i < j) & (j < k)
    
    pi = points[i[mask]]
    pj = points[j[mask]]
    pk = points[k[mask]]
    
    # Cross product
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return areas

def _hull_area(points):
    """Compute area of convex hull."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _exact_score(points):
    """
    Score: Min triangle area / Convex Hull Area.
    """
    hull_area = _hull_area(points)
    if hull_area < 1e-6:
        return 0.0
    areas = _compute_areas(points)
    if len(areas) == 0:
        return 0.0
    min_area = np.min(areas)
    return min_area / hull_area

def _soft_min_score(points, beta=20.0):
    """
    Smooth approximation of the score.
    We want to maximize min(areas) / hull_area.
    min(areas) approx = -1/beta * log(sum(exp(-areas/beta)))
    Since hull_area is also in the denominator, and we want to maximize,
    let's just use soft-min of areas as the objective to maximize,
    and normalize by hull_area separately or just optimize the soft-min area
    while keeping points in a reasonable scale.
    
    Actually, the score is scale invariant.
    Let's define the objective to maximize:
    S = soft_min(areas) / hull_area
    
    To use scipy.minimize (which minimizes), we return -S.
    """
    hull_area = _hull_area(points)
    if hull_area < 1e-6:
        return 0.0
        
    areas = _compute_areas(points)
    if len(areas) == 0:
        return 0.0
        
    # Soft min of areas
    # To avoid overflow/underflow, shift by max area?
    # log-sum-exp trick:
    # m = max(areas)
    # sum(exp((areas - m)/beta))
    # soft_min = m - beta * log(sum(exp((areas-m)/beta)))
    
    m = np.max(areas)
    # If m is 0, score is 0
    if m < 1e-9:
        return 0.0
        
    exp_terms = np.exp((areas - m) / beta)
    soft_min_val = m - beta * np.log(np.sum(exp_terms))
    
    return soft_min_val / hull_area

def _neg_soft_score(points, beta=20.0):
    return -_soft_min_score(points, beta)

def _generate_ring_points(n_inner, n_outer, r_in, r_out, phase_in=0, phase_out=0):
    points = []
    if n_inner > 0:
        angles_in = 2 * np.pi * np.arange(n_inner) / n_inner + phase_in
        for a in angles_in:
            points.append([r_in * np.cos(a), r_in * np.sin(a)])
    if n_outer > 0:
        angles_out = 2 * np.pi * np.arange(n_outer) / n_outer + phase_out
        for a in angles_out:
            points.append([r_out * np.cos(a), r_out * np.sin(a)])
    return np.array(points)

def _optimize_ring_params(params, n_inner, n_outer, beta=20.0):
    """
    params: [r_in, r_out, phase_in, phase_out]
    """
    r_in = abs(params[0])
    r_out = abs(params[1])
    # Ensure r_out > r_in to maintain ring structure? 
    # Or just allow any radii.
    # If r_in > r_out, the "inner" points are actually outer.
    # Let's just let the optimizer decide.
    
    phase_in = params[2]
    phase_out = params[3]
    
    pts = _generate_ring_points(n_inner, n_outer, r_in, r_out, phase_in, phase_out)
    return _neg_soft_score(pts, beta)

def _polish_points(points, max_iter=20, time_limit=10.0):
    """
    Coordinate descent polish using exact score.
    """
    start_time = time.time()
    current_score = _exact_score(points)
    points = points.copy()
    
    for iter in range(max_iter):
        if time.time() - start_time > time_limit:
            break
            
        improved = False
        for i in range(N):
            p_orig = points[i].copy()
            
            # Optimize x
            def obj_x(x):
                points[i, 0] = x
                return -_exact_score(points)
            
            x_min = np.min(points[:, 0]) - 1.0
            x_max = np.max(points[:, 0]) + 1.0
            
            try:
                res_x = minimize_scalar(obj_x, bounds=(x_min, x_max), method='bounded', options={'maxiter': 10})
                if res_x.success:
                    points[i, 0] = res_x.x
            except Exception:
                points[i, 0] = p_orig[0]
            
            # Optimize y
            def obj_y(y):
                points[i, 1] = y
                return -_exact_score(points)
            
            y_min = np.min(points[:, 1]) - 1.0
            y_max = np.max(points[:, 1]) + 1.0
            
            try:
                res_y = minimize_scalar(obj_y, bounds=(y_min, y_max), method='bounded', options={'maxiter': 10})
                if res_y.success:
                    points[i, 1] = res_y.y
            except Exception:
                points[i, 1] = p_orig[1]
                
            new_score = _exact_score(points)
            if new_score > current_score:
                current_score = new_score
                improved = True
            else:
                points[i] = p_orig
                
        if not improved:
            break
            
    return points, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_score = 0.0
    best_points = None
    
    # Strategy:
    # 1. Try various ring splits (n_inner, n_outer)
    # 2. For each split, optimize ring parameters (r_in, r_out, phases) using soft-min score
    # 3. Polish the best candidates with coordinate descent
    
    splits = [(1, 13), (2, 12), (3, 11), (4, 10), (5, 9), (6, 8), (7, 7)]
    
    for n_in, n_out in splits:
        if time.time() > deadline - 20.0:
            break
            
        # Initial guess for radii
        # For regular polygon, all points on R.
        # For two rings, maybe r_out = 1.0, r_in = 0.6?
        # Let's try a few initial configurations for the optimizer
        initial_params_list = [
            [0.5, 1.0, 0.0, 0.0],
            [0.6, 1.0, 0.0, 0.0],
            [0.7, 1.0, 0.0, 0.0],
            [0.5, 1.0, np.pi/n_in, 0.0], # Shift inner
            [0.6, 1.0, np.pi/n_in, 0.0],
            [0.7, 1.0, np.pi/n_in, 0.0],
            [0.5, 1.0, 0.0, np.pi/n_out], # Shift outer
            [0.6, 1.0, 0.0, np.pi/n_out],
            [0.7, 1.0, 0.0, np.pi/n_out],
        ]
        
        for init_params in initial_params_list:
            if time.time() > deadline - 10.0:
                break
                
            try:
                # Optimize ring parameters
                res = minimize(_optimize_ring_params, init_params, args=(n_in, n_out, 20.0), 
                               method='Nelder-Mead', options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-6})
                
                if res.success:
                    best_params = res.x
                    # Ensure radii are positive
                    r_in = abs(best_params[0])
                    r_out = abs(best_params[1])
                    # If r_in > r_out, swap or just use as is? 
                    # The function _generate_ring_points doesn't care about labels, just radii.
                    # But if r_in is huge and r_out is small, it's just two rings.
                    
                    pts = _generate_ring_points(n_in, n_out, r_in, r_out, best_params[2], best_params[3])
                    
                    # Normalize to unit hull area for consistent scale?
                    # The score is scale invariant, so it doesn't matter for the value,
                    # but helps the coordinate descent bounds.
                    hull_a = _hull_area(pts)
                    if hull_a > 1e-6:
                        pts = pts / np.sqrt(hull_a)
                    
                    # Polish
                    polished_pts, score = _polish_points(pts, max_iter=10, time_limit=5.0)
                    
                    if score > best_score:
                        best_score = score
                        best_points = polished_pts
                        
            except Exception as e:
                pass
    
    # Fallback: Regular 14-gon
    if best_points is None or time.time() > deadline - 5.0:
         angles = 2 * np.pi * np.arange(N) / N
         R = 1.0
         init_regular = np.column_stack([R * np.cos(angles), R * np.sin(angles)])
         hull_a = _hull_area(init_regular)
         if hull_a > 0:
             init_regular = init_regular / np.sqrt(hull_a)
         if best_points is None:
             best_points = init_regular
         else:
             # Check if regular is better (unlikely, but safe)
             reg_score = _exact_score(init_regular)
             if reg_score > best_score:
                 best_points = init_regular
                 best_score = reg_score

    return best_points
