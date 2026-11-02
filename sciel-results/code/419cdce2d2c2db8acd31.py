"""
Heilbronn convex regions n=14
Maximize minimum triangle area over all triples, normalized by convex hull area.
"""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 14

def _min_triangle_area(points):
    """
    Compute the minimum area of all triangles formed by the points.
    points: (N, 2) array
    """
    n = len(points)
    if n < 3:
        return 0.0
    
    # Vectorized computation of cross products for all triples
    # To save memory and time for N=14, we can loop over i, j and vectorize k
    min_area = float('inf')
    x = points[:, 0]
    y = points[:, 1]
    
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            # Vectors from i to j and i to k
            dx_ji = x[j] - x[i]
            dy_ji = y[j] - y[i]
            
            # k ranges from j+1 to n-1
            k_indices = np.arange(j + 1, n)
            dx_ki = x[k_indices] - x[i]
            dy_ki = y[k_indices] - y[i]
            
            # Cross product (2D) = dx_ji * dy_ki - dx_ki * dy_ji
            cross = dx_ji * dy_ki - dx_ki * dy_ji
            areas = 0.5 * np.abs(cross)
            
            local_min = np.min(areas)
            if local_min < min_area:
                min_area = local_min
                if min_area < 1e-12:
                    return 0.0
                    
    return min_area

def _hull_area(points):
    """
    Compute the area of the convex hull of the points.
    """
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """
    Calculate the score: min_triangle_area / hull_area.
    """
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = _min_triangle_area(points)
    if min_area < 1e-12:
        return 0.0
    return min_area / hull_area

def _all_triangle_areas(points):
    """
    Return an array of all triangle areas.
    """
    n = len(points)
    x = points[:, 0]
    y = points[:, 1]
    areas = []
    
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            dx_ji = x[j] - x[i]
            dy_ji = y[j] - y[i]
            k_indices = np.arange(j + 1, n)
            dx_ki = x[k_indices] - x[i]
            dy_ki = y[k_indices] - y[i]
            cross = dx_ji * dy_ki - dx_ki * dy_ji
            areas.append(0.5 * np.abs(cross))
            
    if not areas:
        return np.array([0.0])
    return np.concatenate(areas)

def _objective_hard(points_flat):
    """
    Objective function to minimize: hull_area / min_triangle_area.
    Minimizing this is equivalent to maximizing min_triangle_area / hull_area.
    """
    points = points_flat.reshape(N, 2)
    if np.any(~np.isfinite(points)):
        return 1e10
        
    min_area = _min_triangle_area(points)
    if min_area < 1e-12:
        return 1e10
        
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 1e10
        
    return hull_area / min_area

def _objective_soft(points_flat, beta=100.0):
    """
    Smooth soft-min objective: hull_area / soft_min(triangle areas).
    Useful for gradient-based methods like L-BFGS.
    """
    points = points_flat.reshape(N, 2)
    if np.any(~np.isfinite(points)):
        return 1e10
        
    areas = _all_triangle_areas(points)
    if len(areas) == 0 or np.min(areas) < 1e-12:
        return 1e10
        
    # Soft min: -1/beta * log(sum(exp(-beta * A)))
    z = -beta * areas
    max_z = np.max(z)
    # Avoid overflow
    if max_z < -500:
        log_sum = max_z
    else:
        log_sum = max_z + np.log(np.sum(np.exp(z - max_z)))
        
    soft_min_val = -log_sum / beta
    
    if soft_min_val < 1e-12:
        return 1e10
        
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 1e10
        
    return hull_area / soft_min_val

def _normalize_unit_area(points):
    """
    Scale points so that the convex hull area is 1.
    """
    hull_area = _hull_area(points)
    if hull_area > 1e-10:
        scale = np.sqrt(1.0 / hull_area)
        return points * scale
    return points

def _coordinate_descent(pts, deadline, max_rounds=20):
    """
    Cyclic coordinate descent on the hard objective.
    """
    pts = pts.copy()
    n = len(pts)
    step0 = 0.01
    best_score = _score(pts)
    
    for rnd in range(max_rounds):
        if time.time() > deadline - 5:
            break
        improved = False
        for i in range(n):
            if time.time() > deadline - 5:
                break
            for coord in range(2):
                orig = pts[i, coord]
                step = step0
                for _ in range(15):
                    pts[i, coord] = orig + step
                    s_plus = _score(pts)
                    pts[i, coord] = orig - step
                    s_minus = _score(pts)
                    pts[i, coord] = orig
                    
                    if s_plus > best_score:
                        pts[i, coord] = orig + step
                        best_score = s_plus
                        improved = True
                    elif s_minus > best_score:
                        pts[i, coord] = orig - step
                        best_score = s_minus
                        improved = True
                    else:
                        break
                    
                    step *= 0.5
                    if step < 1e-8:
                        break
        if not improved:
            break
            
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_score = 0.0
    best_pts = None
    
    def _consider(pts):
        nonlocal best_score, best_pts
        pts = np.asarray(pts, dtype=float)
        if pts.shape != (N, 2) or np.any(~np.isfinite(pts)):
            return
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()
            
    # Initializations
    inits = []
    
    # 1. Regular polygon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    inits.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # 2. Jittered regular polygons
    rng = np.random.default_rng(42)
    for js in [0.01, 0.05, 0.1, 0.2]:
        pts = np.column_stack([np.cos(angles), np.sin(angles)])
        pts += rng.normal(0, js, size=(N, 2))
        inits.append(pts)
        
    # 3. Random convex-like points (sorted by angle)
    for seed in range(10):
        rs = np.random.default_rng(seed)
        a = rs.uniform(0, 2 * np.pi, N)
        a = np.sort(a)
        # Add some radial noise to break symmetry
        r = rs.uniform(0.8, 1.2, N)
        pts = np.column_stack([r * np.cos(a), r * np.sin(a)])
        inits.append(pts)
        
    # 4. Two-cluster angular distribution
    for shift in [0.1, 0.3, 0.5]:
        a1 = np.linspace(0, 2 * np.pi, 7, endpoint=False)
        a2 = np.linspace(0, 2 * np.pi, 7, endpoint=False) + shift
        a = np.sort(np.concatenate([a1, a2]))
        pts = np.column_stack([np.cos(a), np.sin(a)])
        inits.append(pts)

    # Optimization Loop
    for x0 in inits:
        if time.time() > deadline - 15:
            break
            
        x0_flat = x0.flatten()
        
        # Phase 1: Nelder-Mead (robust, derivative-free)
        try:
            res_nm = minimize(_objective_hard, x0_flat, method='Nelder-Mead',
                              options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            if res_nm.fun < 1e9:
                _consider(res_nm.x.reshape(N, 2))
        except Exception:
            pass
            
        # Phase 2: Powell (often better for polishing)
        if time.time() < deadline - 10:
            # Start from the best of initial or NM result
            start_powell = best_pts.flatten() if best_pts is not None else x0_flat
            try:
                res_powell = minimize(_objective_hard, start_powell, method='Powell',
                                      options={'maxiter': 1000, 'xtol': 1e-12, 'ftol': 1e-14})
                if res_powell.fun < 1e9:
                    _consider(res_powell.x.reshape(N, 2))
            except Exception:
                pass
                
        # Phase 3: Soft-min L-BFGS with annealing
        if time.time() < deadline - 5:
            start_lbfgs = best_pts.flatten() if best_pts is not None else x0_flat
            for beta in [10, 50, 200]:
                if time.time() > deadline - 2:
                    break
                try:
                    res_lbfgs = minimize(_objective_soft, start_lbfgs, args=(beta,), method='L-BFGS-B',
                                         options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
                    if res_lbfgs.fun < 1e9:
                        _consider(res_lbfgs.x.reshape(N, 2))
                        start_lbfgs = res_lbfgs.x
                except Exception:
                    pass

    # Final Polish: Coordinate Descent
    if best_pts is not None and time.time() < deadline - 2:
        refined = _coordinate_descent(best_pts, deadline)
        _consider(refined)
        
    # Fallback
    if best_pts is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
        
    # Normalize to unit area
    final_points = _normalize_unit_area(best_pts)
    
    # Safety check
    try:
        hull = ConvexHull(final_points)
        if hull.volume < 1e-10:
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
            final_points = _normalize_unit_area(np.column_stack([np.cos(angles), np.sin(angles)]))
    except Exception:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        final_points = _normalize_unit_area(np.column_stack([np.cos(angles), np.sin(angles)]))
        
    return final_points

# EVOLVE-BLOCK-END
