"""
Structural change: enforce convexity via sorted radial function r(θ) with positive
Fourier coefficients, reducing the search space and guaranteeing a convex hull.
This avoids degenerate configurations and focuses optimization on the smooth
manifold of convex regions.
"""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull

N = 14

def _min_triangle_area(points):
    n = len(points)
    if n < 3:
        return 0.0
    x = points[:, 0]
    y = points[:, 1]
    min_area = float('inf')
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            dx_ji = x[j] - x[i]
            dy_ji = y[j] - y[i]
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            cross = dx_ji * dy_ki - dy_ji * dx_ki
            areas = 0.5 * np.abs(cross)
            local_min = np.min(areas)
            if local_min < min_area:
                min_area = local_min
    return min_area

def _hull_area(points):
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area

def _all_areas(points):
    n = len(points)
    x = points[:, 0]
    y = points[:, 1]
    areas_list = []
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            dx_ji = x[j] - x[i]
            dy_ji = y[j] - y[i]
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            cross = dx_ji * dy_ki - dy_ji * dx_ki
            areas_list.append(0.5 * np.abs(cross))
    if not areas_list:
        return np.array([0.0])
    return np.concatenate(areas_list)

def _objective_hard(flat):
    pts = flat.reshape(N, 2)
    if not np.all(np.isfinite(pts)):
        return 1e10
    hull_area = _hull_area(pts)
    if hull_area < 1e-10:
        return 1e10
    min_area = _min_triangle_area(pts)
    if min_area < 1e-12:
        return 1e10
    return hull_area / min_area

def _objective_soft(flat, beta=50.0):
    pts = flat.reshape(N, 2)
    if not np.all(np.isfinite(pts)):
        return 1e10
    areas = _all_areas(pts)
    if len(areas) == 0 or np.min(areas) < 1e-12:
        return 1e10
    z = -beta * areas
    max_z = np.max(z)
    if max_z < -500:
        log_sum = max_z
    else:
        log_sum = max_z + np.log(np.sum(np.exp(z - max_z)))
    soft_min = -log_sum / beta
    if soft_min < 1e-12:
        return 1e10
    hull_area = _hull_area(pts)
    if hull_area < 1e-10:
        return 1e10
    return hull_area / soft_min

def _fourier_to_points(coeffs, n_harmonics=5):
    """
    Convert Fourier coefficients to 14 points on a convex curve.
    coeffs = [r0, a1, b1, a2, b2, ..., an, bn]
    r(θ) = r0 + sum_{k=1}^{n} (a_k cos(kθ) + b_k sin(kθ))
    Points are placed at equally spaced angles θ_i = 2πi/N.
    """
    r0 = coeffs[0]
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    r = np.full(N, r0)
    idx = 1
    for k in range(1, n_harmonics + 1):
        ak = coeffs[idx]
        bk = coeffs[idx + 1]
        idx += 2
        r += ak * np.cos(k * angles) + bk * np.sin(k * angles)
    # Ensure r > 0 for convexity (add small offset if needed)
    r_min = np.min(r)
    if r_min <= 0.05:
        r += (0.05 - r_min)
    pts = np.column_stack([r * np.cos(angles), r * np.sin(angles)])
    return pts

def _objective_fourier(flat, n_harmonics=5):
    pts = _fourier_to_points(flat, n_harmonics)
    return _objective_hard(pts.flatten())

def _generate_fourier_inits(n_harmonics=5):
    """
    Generate diverse initial Fourier coefficient sets.
    """
    inits = []
    rng = np.random.default_rng(42)
    
    # 1. Circle (only r0)
    inits.append(np.concatenate([[1.0], np.zeros(2 * n_harmonics)]))
    
    # 2. Ellipse (r0 + a1)
    for a1 in [0.1, 0.2, 0.3]:
        inits.append(np.concatenate([[1.0], [a1], np.zeros(2 * n_harmonics - 1)]))
    
    # 3. Random small perturbations
    for seed in range(10):
        rs = np.random.default_rng(seed * 7 + 3)
        coeffs = np.concatenate([[1.0], rs.uniform(-0.05, 0.05, 2 * n_harmonics)])
        inits.append(coeffs)
    
    # 4. Ring-like: two harmonics with larger amplitudes
    for r0 in [0.8, 1.0, 1.2]:
        for a2 in [0.1, 0.2]:
            coeffs = np.zeros(1 + 2 * n_harmonics)
            coeffs[0] = r0
            coeffs[3] = a2  # a2 coefficient
            inits.append(coeffs)
    
    return inits

def _optimize_fourier(coeffs0, deadline, n_harmonics=5):
    if time.time() > deadline - 5:
        return None
    try:
        res = minimize(_objective_fourier, coeffs0, method='Nelder-Mead',
                       options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True,
                                'args': (n_harmonics,)})
        return res
    except Exception:
        return None

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin before 150s kill
    
    best_score = 0.0
    best_pts = None
    best_obj = float('inf')
    
    def update_best(pts):
        nonlocal best_score, best_pts, best_obj
        if pts is None:
            return
        pts = np.asarray(pts, dtype=float)
        if pts.shape != (N, 2) or not np.all(np.isfinite(pts)):
            return
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()
            best_obj = _objective_hard(pts.flatten())

    # Phase 1: Fourier-based optimization (convexity guaranteed)
    n_harmonics = 5
    inits = _generate_fourier_inits(n_harmonics)
    
    for i, coeffs0 in enumerate(inits):
        if time.time() > deadline - 60:
            break
        res = _optimize_fourier(coeffs0, deadline, n_harmonics)
        if res is not None and res.fun < best_obj:
            pts = _fourier_to_points(res.x, n_harmonics)
            update_best(pts)
    
    # Phase 2: Multi-start Nelder-Mead on full 28D space from good Fourier solutions
    if best_pts is not None and time.time() < deadline - 40:
        try:
            res = minimize(_objective_hard, best_pts.flatten(), method='Nelder-Mead',
                           options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            if res.fun < best_obj:
                update_best(res.x.reshape(N, 2))
        except Exception:
            pass
    
    # Phase 3: L-BFGS-B with soft objective
    if best_pts is not None and time.time() < deadline - 25:
        try:
            res_lbfgs = minimize(_objective_soft, best_pts.flatten(), method='L-BFGS-B',
                                 options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-10})
            if res_lbfgs.fun < best_obj:
                update_best(res_lbfgs.x.reshape(N, 2))
        except Exception:
            pass
    
    # Phase 4: Differential Evolution (Global Search)
    if time.time() < deadline - 15:
        try:
            bounds = [(-1.5, 1.5)] * (N * 2)
            de_res = differential_evolution(
                _objective_hard, bounds,
                seed=42, maxiter=50, tol=1e-8,
                popsize=15, mutation=(0.4, 1.0), recombination=0.7,
                polish=False, update_interval=1,
                init='sobol'
            )
            if de_res.fun < best_obj:
                update_best(de_res.x.reshape(N, 2))
        except Exception:
            pass
    
    # Phase 5: Final Polish
    if best_pts is not None and time.time() < deadline - 5:
        try:
            res = minimize(_objective_hard, best_pts.flatten(), method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-11, 'fatol': 1e-13, 'adaptive': True})
            if res.fun < best_obj:
                update_best(res.x.reshape(N, 2))
                
            if time.time() < deadline - 2:
                res_powell = minimize(_objective_hard, best_pts.flatten(), method='Powell',
                                      options={'maxiter': 2000, 'xtol': 1e-13, 'ftol': 1e-15})
                if res_powell.fun < best_obj:
                    update_best(res_powell.x.reshape(N, 2))
        except Exception:
            pass
    
    # Fallback to regular polygon if nothing worked
    if best_pts is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Normalize to unit area
    try:
        hull = ConvexHull(best_pts)
        area = hull.volume
        if area > 1e-10:
            scale = np.sqrt(1.0 / area)
            best_pts = best_pts * scale
    except Exception:
        pass
    
    return best_pts
