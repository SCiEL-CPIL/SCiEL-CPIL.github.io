"""
Heilbronn convex regions n=14: maximize min triangle area / convex hull area.
Structural change: adaptive arc-length reparameterization with cumulative
arc-length fractions, combined with Fourier shape parameters and full 2D
coordinate refinement.
"""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 14
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0

def _min_triangle_area(points):
    """Calculate the minimum area of any triangle formed by triples of points."""
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
    """Calculate the area of the convex hull of the points."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """Calculate the score: min_triangle_area / hull_area."""
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area

def _normalize_unit_area(points):
    """Normalize points to unit convex hull area."""
    try:
        hull = ConvexHull(points)
        area = hull.volume
        if area > 1e-10:
            scale = np.sqrt(1.0 / area)
            return points * scale
    except Exception:
        pass
    xs = points[:, 0]
    ys = points[:, 1]
    w = xs.max() - xs.min()
    h = ys.max() - ys.min()
    if w > 1e-10 and h > 1e-10:
        scale = 1.0 / np.sqrt(w * h)
        return points * scale
    return points

def _fourier_boundary(theta, coeffs):
    """
    Evaluate a smooth convex boundary at angles theta.
    coeffs: [r0, a1, b1, a2, b2, ..., aK, bK]
    r(θ) = r0 + Σ a_k cos(kθ) + b_k sin(kθ)
    """
    K = (len(coeffs) - 1) // 2
    r = np.full_like(theta, coeffs[0], dtype=float)
    for k in range(1, K + 1):
        a = coeffs[2 * k - 1]
        b = coeffs[2 * k]
        r += a * np.cos(k * theta) + b * np.sin(k * theta)
    return np.maximum(r, 0.05)

def _decode_arc_params(params):
    """
    Decode optimization parameters into N points.
    params: [r0, a1, b1, a2, b2, a3, b3, f0, f1, ..., f_{N-1}]
    where f_i are cumulative arc-length fractions (monotone, in [0,1]).
    """
    K = 3  # 3 harmonics
    n_shape = 2 * K + 1  # r0 + 2*K coefficients
    r0 = max(params[0], 0.5)
    shape_coeffs = np.concatenate([[r0], params[1:n_shape]])
    
    # Decode arc-length fractions: use softplus to ensure monotonicity
    raw_f = params[n_shape:n_shape + N]
    # Convert to cumulative fractions: f_i = sum of positive increments
    # Use a differentiable monotone mapping
    increments = np.exp(raw_f)  # positive increments
    f = np.cumsum(increments)
    f = f / f[-1]  # normalize to [0, 1]
    
    # Map fractions to angles
    angles = f * 2 * np.pi
    
    # Evaluate boundary at these angles
    radii = _fourier_boundary(angles, shape_coeffs)
    
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    
    return np.column_stack([x, y])

def _objective_arc(params, deadline=None):
    """Objective for arc-length based optimization."""
    if deadline is not None and time.time() > deadline:
        return 1e10
    pts = _decode_arc_params(params)
    if np.any(~np.isfinite(pts)):
        return 1e10
    s = _score(pts)
    if s <= 0:
        return 1e10
    return -s

def _decode_2d(flat):
    """Decode 28D flat vector into 14x2 points."""
    pts = flat.reshape(N, 2)
    return pts

def _objective_2d(flat, deadline=None):
    """Objective for full 2D coordinate optimization."""
    if deadline is not None and time.time() > deadline:
        return 1e10
    pts = _decode_2d(flat)
    if np.any(~np.isfinite(pts)):
        return 1e10
    s = _score(pts)
    if s <= 0:
        return 1e10
    return -s

def _coordinate_descent_polish(pts, deadline, max_rounds=5):
    """Polish points using coordinate descent on each coordinate."""
    pts = pts.copy()
    for _ in range(max_rounds):
        if time.time() > deadline:
            break
        improved = False
        for i in range(N):
            for dim in range(2):
                def obj_1d(val):
                    pts_tmp = pts.copy()
                    pts_tmp[i, dim] = val
                    s = _score(pts_tmp)
                    return -s if s > 0 else 1e10
                
                try:
                    res = minimize(obj_1d, pts[i, dim], method='Brent',
                                   bounds=(-5.0, 5.0), options={'xtol': 1e-10})
                    if res.fun < obj_1d(pts[i, dim]) - 1e-12:
                        pts[i, dim] = res.x
                        improved = True
                except Exception:
                    pass
        if not improved:
            break
    return pts

def _regular_polygon():
    """Generate a regular 14-gon."""
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    return np.column_stack([np.cos(angles), np.sin(angles)])

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT - SAFETY_MARGIN
    
    best_score_val = 0.0
    best_pts = None
    
    def _consider(pts):
        nonlocal best_score_val, best_pts
        pts = np.asarray(pts, dtype=float)
        if pts.shape != (N, 2) or np.any(~np.isfinite(pts)):
            return
        s = _score(pts)
        if s > best_score_val:
            best_score_val = s
            best_pts = pts.copy()
    
    # ---- Phase 1: Arc-length reparameterization optimization ----
    if time.time() < deadline - 60:
        # Initial parameters: near-circle with equal arc spacing
        K = 3
        n_shape = 2 * K + 1
        # r0 = 1.0, small harmonics
        shape_init = np.array([1.0, 0.05, 0.02, 0.01, 0.005, 0.0, 0.0])
        # Equal arc fractions: increments all equal
        raw_f_init = np.zeros(N)  # exp(0)=1, so equal increments
        init_params = np.concatenate([shape_init, raw_f_init])
        
        # Powell on arc-length params
        if time.time() < deadline - 50:
            try:
                res1 = minimize(_objective_arc, init_params, method='Powell',
                                options={'maxiter': 1000, 'ftol': 1e-10, 'xtol': 1e-10})
                pts1 = _decode_arc_params(res1.x)
                _consider(pts1)
            except Exception:
                pass
        
        # Nelder-Mead refinement
        if time.time() < deadline - 35:
            try:
                res2 = minimize(_objective_arc, res1.x, method='Nelder-Mead',
                                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
                pts2 = _decode_arc_params(res2.x)
                _consider(pts2)
            except Exception:
                pass
    
    # ---- Phase 2: Try multiple random restarts with different initial shapes ----
    if time.time() < deadline - 30:
        np.random.seed(42)
        for restart in range(3):
            if time.time() > deadline - 25:
                break
            
            # Different initial shapes
            if restart == 0:
                # Ellipse-like
                shape_init = np.array([1.0, 0.15, 0.0, 0.0, 0.0, 0.0, 0.0])
            elif restart == 1:
                # Slightly irregular
                shape_init = np.array([1.0, 0.08, 0.06, 0.04, 0.02, 0.01, 0.0])
            else:
                # Near regular with small random perturbation
                shape_init = np.array([1.0, 0.03, 0.02, 0.02, 0.01, 0.01, 0.01])
            
            raw_f_init = np.zeros(N)
            init_params = np.concatenate([shape_init, raw_f_init])
            
            if time.time() < deadline - 20:
                try:
                    res = minimize(_objective_arc, init_params, method='Powell',
                                   options={'maxiter': 500, 'ftol': 1e-9, 'xtol': 1e-9})
                    pts = _decode_arc_params(res.x)
                    _consider(pts)
                except Exception:
                    pass
    
    # ---- Phase 3: Full 2D coordinate optimization from best so far ----
    if best_pts is None:
        best_pts = _regular_polygon()
        _consider(best_pts)
    
    if best_pts is not None and time.time() < deadline - 20:
        pts_flat = best_pts.flatten().copy()
        
        # Nelder-Mead on full 2D
        if time.time() < deadline - 12:
            try:
                res3 = minimize(_objective_2d, pts_flat, method='Nelder-Mead',
                                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
                pts3 = res3.x.reshape(N, 2)
                _consider(pts3)
            except Exception:
                pass
        
        # Powell on full 2D
        if time.time() < deadline - 8:
            try:
                res4 = minimize(_objective_2d, pts_flat, method='Powell',
                                options={'maxiter': 300, 'ftol': 1e-10, 'xtol': 1e-10})
                pts4 = res4.x.reshape(N, 2)
                _consider(pts4)
            except Exception:
                pass
    
    # ---- Phase 4: Coordinate descent polish ----
    if best_pts is not None and time.time() < deadline - 3:
        pts_cd = _coordinate_descent_polish(best_pts, deadline, max_rounds=8)
        _consider(pts_cd)
    
    # ---- Final: Try a direct 2D optimization with different starting points ----
    if time.time() < deadline - 5:
        # Start from regular polygon
        reg = _regular_polygon().flatten()
        try:
            res5 = minimize(_objective_2d, reg, method='Nelder-Mead',
                            options={'maxiter': 1500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            pts5 = res5.x.reshape(N, 2)
            _consider(pts5)
        except Exception:
            pass
    
    # Normalize to unit area
    if best_pts is not None:
        best_pts = _normalize_unit_area(best_pts)
    
    # Fallback
    if best_pts is None:
        best_pts = _normalize_unit_area(_regular_polygon())
    
    return best_pts
