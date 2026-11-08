"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull

DEADLINE = 135.0  # Safety margin before 150s kill


def _soft_min_lse(areas, beta):
    """Compute soft minimum via log-sum-exp trick."""
    z = -beta * areas
    max_z = np.max(z)
    if max_z < -500:
        log_sum = max_z
    else:
        log_sum = max_z + np.log(np.sum(np.exp(z - max_z)))
    return -log_sum / beta


def _hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0


def _min_triangle_area_vec(points):
    """Vectorized minimum triangle area for N points."""
    n = len(points)
    min_area = float('inf')
    x = points[:, 0]
    y = points[:, 1]
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            cross = (x[j] - x[i]) * dy_ki - dx_ki * (y[j] - y[i])
            local_min = 0.5 * np.min(np.abs(cross))
            if local_min < min_area:
                min_area = local_min
                if min_area < 1e-14:
                    return min_area
    return min_area


def _all_triangle_areas(points):
    """Return array of all triangle areas (N choose 3)."""
    n = len(points)
    x = points[:, 0]
    y = points[:, 1]
    areas_list = []
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            cross = (x[j] - x[i]) * dy_ki - dx_ki * (y[j] - y[i])
            areas_list.append(0.5 * np.abs(cross))
    if not areas_list:
        return np.array([0.0])
    return np.concatenate(areas_list)


def _score(pts):
    """Compute min triangle area / hull area."""
    hull_a = _hull_area(pts)
    if hull_a <= 0:
        return 0.0
    min_area = _min_triangle_area_vec(pts)
    return min_area / hull_a


def _neg_score(pts_flat):
    """Negative score for minimization."""
    pts = pts_flat.reshape(-1, 2)
    s = _score(pts)
    return -s


def _soft_obj(pts_flat, beta=150.0):
    """Soft objective: hull_area / soft_min(triangle areas)."""
    points = pts_flat.reshape(N, 2)
    if np.any(~np.isfinite(points)):
        return 1e10
    areas = _all_triangle_areas(points)
    if len(areas) == 0 or np.min(areas) < 1e-12:
        return 1e10
    z = -beta * areas
    max_z = np.max(z)
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


def _multi_beta_soft_obj(pts_flat, betas=(50.0, 150.0, 300.0)):
    """Multi-beta soft objective: weighted combination of soft-mins."""
    points = pts_flat.reshape(N, 2)
    if np.any(~np.isfinite(points)):
        return 1e10
    areas = _all_triangle_areas(points)
    if len(areas) == 0 or np.min(areas) < 1e-12:
        return 1e10
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 1e10
    total = 0.0
    for beta in betas:
        soft_min_val = _soft_min_lse(areas, beta)
        if soft_min_val < 1e-12:
            return 1e10
        total += hull_area / soft_min_val
    return total / len(betas)


def _lbfgs_obj(pts_flat, beta=200.0):
    """Smooth objective for L-BFGS-B: hull_area / soft_min(triangle areas)."""
    points = pts_flat.reshape(N, 2)
    if np.any(~np.isfinite(points)):
        return 1e10
    areas = _all_triangle_areas(points)
    if len(areas) == 0 or np.min(areas) < 1e-12:
        return 1e10
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 1e10
    soft_min_val = _soft_min_lse(areas, beta)
    if soft_min_val < 1e-12:
        return 1e10
    return hull_area / soft_min_val


def _lbfgs_obj_grad(pts_flat, beta=200.0):
    """Objective and numerical gradient for L-BFGS-B."""
    points = pts_flat.reshape(N, 2)
    if np.any(~np.isfinite(points)):
        return 1e10, np.zeros_like(pts_flat)
    
    areas = _all_triangle_areas(points)
    if len(areas) == 0 or np.min(areas) < 1e-12:
        return 1e10, np.zeros_like(pts_flat)
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 1e10, np.zeros_like(pts_flat)
    soft_min_val = _soft_min_lse(areas, beta)
    if soft_min_val < 1e-12:
        return 1e10, np.zeros_like(pts_flat)
    
    obj = hull_area / soft_min_val
    
    # Numerical gradient (central differences)
    h = 1e-7
    grad = np.zeros_like(pts_flat)
    for i in range(len(pts_flat)):
        pts_plus = pts_flat.copy()
        pts_plus[i] += h
        obj_plus = _lbfgs_obj(pts_plus, beta)
        
        pts_minus = pts_flat.copy()
        pts_minus[i] -= h
        obj_minus = _lbfgs_obj(pts_minus, beta)
        
        grad[i] = (obj_plus - obj_minus) / (2 * h)
    
    return obj, grad


def _coordinate_descent_polish(pts, max_iter=500, tol=1e-10):
    """Coordinate descent: optimize each coordinate independently."""
    pts = pts.copy()
    start_time = time.time()
    
    for iteration in range(max_iter):
        if time.time() - start_time > DEADLINE - 8:
            break
        
        improved = False
        for i in range(N):
            for coord in range(2):
                # Try small perturbations
                best_val = _score(pts)
                best_pos = pts[i, coord]
                
                for delta in [1e-4, -1e-4, 1e-5, -1e-5, 1e-6, -1e-6]:
                    test_pts = pts.copy()
                    test_pts[i, coord] += delta
                    test_score = _score(test_pts)
                    if test_score > best_val:
                        best_val = test_score
                        best_pos = test_pts[i, coord]
                
                if abs(best_pos - pts[i, coord]) > tol:
                    pts[i, coord] = best_pos
                    improved = True
        
        if not improved:
            break
    
    return pts


def _normalize_unit_area(pts):
    """Scale points so convex hull area is 1."""
    hull_a = _hull_area(pts)
    if hull_a > 0:
        return pts / np.sqrt(hull_a)
    return pts


def _initial_regular_polygon(n):
    """Points on a regular n-gon inscribed in unit circle, scaled to unit area."""
    angles = 2 * np.pi * np.arange(n) / n
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    return _normalize_unit_area(pts)


def _initial_mixed(n):
    """Mix of boundary and interior points for better minimum triangle area."""
    outer_n = 10
    inner_n = n - outer_n
    angles = 2 * np.pi * np.arange(outer_n) / outer_n
    outer = np.column_stack([np.cos(angles), np.sin(angles)])
    
    inner_angles = 2 * np.pi * np.arange(inner_n) / inner_n + np.pi / inner_n
    inner = 0.4 * np.column_stack([np.cos(inner_angles), np.sin(inner_angles)])
    
    pts = np.vstack([outer, inner])
    return _normalize_unit_area(pts)


def _initial_hexagonal(n):
    """Hexagonal lattice inspired initialization."""
    # Create a hexagonal arrangement
    # 14 points: center + ring of 6 + ring of 7 (approx)
    pts = []
    # Center
    pts.append([0.0, 0.0])
    # First ring (6 points)
    for i in range(6):
        angle = 2 * np.pi * i / 6
        pts.append([np.cos(angle), np.sin(angle)])
    # Second ring (7 points to make 14 total)
    for i in range(7):
        angle = 2 * np.pi * i / 7 + np.pi / 7
        r = 1.8
        pts.append([r * np.cos(angle), r * np.sin(angle)])
    
    pts = np.array(pts)
    return _normalize_unit_area(pts)


def _initial_cosine_mod(n):
    """Regular polygon with cosine-modulated radii."""
    angles = 2 * np.pi * np.arange(n) / n
    # Modulate radii with a cosine wave
    mod = 1.0 + 0.1 * np.cos(3 * angles)
    pts = np.column_stack([mod * np.cos(angles), mod * np.sin(angles)])
    return _normalize_unit_area(pts)


def construct():
    start_time = time.time()
    
    # Try multiple initial configurations
    inits = [
        _initial_regular_polygon(N),
        _initial_mixed(N),
        _initial_hexagonal(N),
        _initial_cosine_mod(N),
    ]
    
    # Also try a few random perturbations of the regular polygon
    rng = np.random.RandomState(42)
    for _ in range(3):
        base = _initial_regular_polygon(N)
        noise = rng.normal(0, 0.05, size=base.shape)
        perturbed = base + noise
        perturbed = _normalize_unit_area(perturbed)
        inits.append(perturbed)
    
    best_pts = None
    best_score = 0.0
    
    # 1. Global Search using Differential Evolution (shorter to leave time for local refinement)
    if time.time() - start_time < DEADLINE - 40:
        try:
            bounds = [(-2.0, 2.0)] * (2 * N)
            res_de = differential_evolution(
                _neg_score, 
                bounds, 
                maxiter=50, 
                popsize=12, 
                tol=1e-5, 
                polish=True, 
                seed=42,
                update_interval=1,
                callback=lambda xk, convergence: convergence and (time.time() - start_time > DEADLINE - 45)
            )
            if res_de.x is not None:
                cand = res_de.x.reshape(-1, 2)
                s = _score(cand)
                if s > best_score:
                    best_score = s
                    best_pts = cand
        except Exception:
            pass

    # 2. Local Refinement from Inits and DE Result
    candidates_to_polish = [init for init in inits]
    if best_pts is not None:
        candidates_to_polish.append(best_pts)
    
    for init in candidates_to_polish:
        if time.time() - start_time > DEADLINE - 15:
            break
        
        x0 = init.flatten()
        current_best_from_this = init.copy()
        current_best_score_from_this = _score(init)
        
        # Pass 1: L-BFGS-B with smooth objective (gradient-based, faster convergence)
        if time.time() - start_time < DEADLINE - 12:
            try:
                res = minimize(_lbfgs_obj, x0, method='L-BFGS-B',
                              options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8,
                                      'maxfun': 2000})
                if res.x is not None:
                    cand = res.x.reshape(-1, 2)
                    s = _score(cand)
                    if s > current_best_score_from_this:
                        current_best_score_from_this = s
                        current_best_from_this = cand
                    if s > best_score:
                        best_score = s
                        best_pts = cand
            except Exception:
                pass
        
        # Pass 2: Nelder-Mead on soft objective from L-BFGS result
        if time.time() - start_time < DEADLINE - 12:
            try:
                x_start = current_best_from_this.flatten()
                res = minimize(_multi_beta_soft_obj, x_start, method='Nelder-Mead',
                              options={'maxiter': 3000, 'xatol': 1e-7, 'fatol': 1e-9,
                                      'adaptive': True})
                if res.x is not None:
                    cand = res.x.reshape(-1, 2)
                    s = _score(cand)
                    if s > current_best_score_from_this:
                        current_best_score_from_this = s
                        current_best_from_this = cand
                    if s > best_score:
                        best_score = s
                        best_pts = cand
            except Exception:
                pass
        
        # Pass 3: Nelder-Mead on hard objective
        if time.time() - start_time < DEADLINE - 10:
            try:
                x_start = current_best_from_this.flatten()
                res = minimize(_neg_score, x_start, method='Nelder-Mead',
                              options={'maxiter': 3000, 'xatol': 1e-7, 'fatol': 1e-9,
                                      'adaptive': True})
                if res.x is not None:
                    cand = res.x.reshape(-1, 2)
                    s = _score(cand)
                    if s > current_best_score_from_this:
                        current_best_score_from_this = s
                        current_best_from_this = cand
                    if s > best_score:
                        best_score = s
                        best_pts = cand
            except Exception:
                pass
    
    # 3. Final High-Precision Polish: Nelder-Mead then coordinate descent
    if best_pts is not None and time.time() - start_time < DEADLINE - 5:
        try:
            x0 = best_pts.flatten()
            res = minimize(_neg_score, x0, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12,
                                  'adaptive': True})
            if res.x is not None:
                cand = res.x.reshape(-1, 2)
                s = _score(cand)
                if s > best_score:
                    best_score = s
                    best_pts = cand
        except Exception:
            pass
        
        # Coordinate descent for final squeeze
        if best_pts is not None and time.time() - start_time < DEADLINE - 3:
            try:
                polished = _coordinate_descent_polish(best_pts, max_iter=200)
                s = _score(polished)
                if s > best_score:
                    best_score = s
                    best_pts = polished
            except Exception:
                pass

    if best_pts is None:
        best_pts = _initial_regular_polygon(N)
    
    return best_pts
# EVOLVE-BLOCK-END
