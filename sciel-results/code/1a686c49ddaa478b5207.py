"""
Fourier-parameterized convex boundary approach for Heilbronn n=14.
Points are constrained to a smooth convex curve defined by Fourier coefficients,
with optional radial jitter. This reduces 28 free coordinates to ~12 structured
parameters, exploiting the known structure of optimal configurations.
"""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 14
DEADLINE_MARGIN = 15.0  # safety margin before 150s kill

def _min_triangle_area(pts):
    """Vectorized min triangle area."""
    n = len(pts)
    if n < 3:
        return 0.0
    # All pairs (i, j) with i < j, then for each pair compute cross with all k > j
    x = pts[:, 0]
    y = pts[:, 1]
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

def _hull_area(pts):
    """Convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _score(pts):
    """Score: min_triangle_area / hull_area."""
    hull_area = _hull_area(pts)
    if hull_area < 1e-10:
        return 0.0
    min_area = _min_triangle_area(pts)
    return min_area / hull_area

def _objective(pts_flat):
    """Objective to minimize: hull_area / min_triangle_area."""
    pts = pts_flat.reshape(N, 2)
    if not np.all(np.isfinite(pts)):
        return 1e10
    hull_area = _hull_area(pts)
    if hull_area < 1e-10:
        return 1e10
    min_area = _min_triangle_area(pts)
    if min_area < 1e-12:
        return 1e10
    return hull_area / min_area

def _fourier_boundary(theta, coeffs, n_harmonics=3):
    """
    Compute convex boundary radius as function of angle.
    coeffs: array of [a0, b1, a1, b2, a2, b3, a3, ...]
    r(theta) = a0 + sum_{k=1}^{K} (a_k*cos(k*theta) + b_k*sin(k*theta))
    We ensure convexity by keeping coefficients small relative to a0.
    """
    r = np.full_like(theta, coeffs[0])
    for k in range(1, n_harmonics + 1):
        idx_a = 1 + 2 * (k - 1)
        idx_b = 2 + 2 * (k - 1)
        if idx_a < len(coeffs):
            r = r + coeffs[idx_a] * np.cos(k * theta)
        if idx_b < len(coeffs):
            r = r + coeffs[idx_b] * np.sin(k * theta)
    return r

def _decode_fourier(params, n_harmonics=3, jitter_std=0.0):
    """
    Decode Fourier params to N points.
    params: [a0, b1, a1, b2, a2, b3, a3, phase, jitter_scale]
    - a0: base radius (must be > max perturbation)
    - b_k, a_k: Fourier coefficients for harmonic k
    - phase: rotation of the whole configuration
    - jitter_scale: std of radial jitter (0 = exactly on curve)
    Points are placed at equally-spaced angles (regular polygon angles) 
    but with radius determined by the Fourier boundary + jitter.
    """
    n_h = n_harmonics
    n_coeffs = 1 + 2 * n_h  # a0 + n_h pairs
    a0 = params[0]
    fourier_coeffs = params[1:1 + 2 * n_h]
    phase = params[1 + 2 * n_h] if len(params) > 1 + 2 * n_h else 0.0
    jitter_scale = params[2 + 2 * n_h] if len(params) > 2 + 2 * n_h else 0.0
    
    # Equally spaced base angles
    base_angles = np.linspace(0, 2 * np.pi, N, endpoint=False) + phase
    
    # Compute boundary radius at each angle
    r = _fourier_boundary(base_angles, np.concatenate([[a0], fourier_coeffs]), n_h)
    
    # Add radial jitter if requested
    if jitter_scale > 1e-6:
        # Use deterministic jitter based on index for reproducibility
        rng = np.random.default_rng(42)
        jitter = rng.normal(0, jitter_scale, N)
        r = r + jitter
    
    # Ensure positive radius
    r = np.maximum(r, 0.05)
    
    pts = np.column_stack([r * np.cos(base_angles), r * np.sin(base_angles)])
    return pts

def _objective_fourier(params, n_harmonics=3):
    """Objective in Fourier parameter space."""
    pts = _decode_fourier(params, n_harmonics)
    return _objective(pts.flatten())

def _regular_polygon(n, radius=1.0, phase=0.0):
    """Regular n-gon."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + phase
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def _two_ring_config(r_inner, n_inner, n_outer, phase_offset=0.0):
    """Two concentric rings."""
    a1 = np.linspace(0, 2 * np.pi, n_inner, endpoint=False)
    a2 = np.linspace(0, 2 * np.pi, n_outer, endpoint=False) + phase_offset
    p1 = r_inner * np.column_stack([np.cos(a1), np.sin(a1)])
    p2 = np.column_stack([np.cos(a2), np.sin(a2)])
    return np.vstack([p1, p2])

def construct():
    start_time = time.time()
    deadline = start_time + 150.0 - DEADLINE_MARGIN  # 135s effective
    
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
            best_obj = _objective(pts.flatten())
    
    def time_left():
        return deadline - time.time()
    
    # === Phase 0: Quick evaluation of structured initial configs ===
    inits = []
    # Regular 14-gon
    inits.append(_regular_polygon(N))
    # Two rings: 7+7
    for r in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        inits.append(_two_ring_config(r, 7, 7, np.pi / 7))
    # Two rings: 5+9, 4+10, 6+8
    for n1, n2 in [(5, 9), (4, 10), (6, 8), (9, 5), (10, 4), (8, 6)]:
        for r in [0.4, 0.6, 0.7]:
            inits.append(_two_ring_config(r, n1, n2, np.pi / max(n1, 1)))
    # Three rings
    for r1, r2 in [(0.3, 0.6), (0.4, 0.7), (0.5, 0.8)]:
        a1 = np.linspace(0, 2*np.pi, 5, endpoint=False)
        a2 = np.linspace(0, 2*np.pi, 4, endpoint=False) + np.pi/4
        a3 = np.linspace(0, 2*np.pi, 5, endpoint=False) + np.pi/5
        p1 = r1 * np.column_stack([np.cos(a1), np.sin(a1)])
        p2 = r2 * np.column_stack([np.cos(a2), np.sin(a2)])
        p3 = np.column_stack([np.cos(a3), np.sin(a3)])
        inits.append(np.vstack([p1, p2, p3]))
    
    for pts0 in inits:
        if time_left() < 20:
            break
        update_best(pts0)
    
    # === Phase 1: Fourier parameter optimization (structured search) ===
    n_harmonics = 3
    n_fourier_params = 1 + 2 * n_harmonics + 2  # a0 + 6 coeffs + phase + jitter
    
    # Multiple starting points in Fourier space
    fourier_starts = []
    # Start 1: nearly circular (a0=1, small perturbations)
    fourier_starts.append(np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    # Start 2: slight elliptical
    fourier_starts.append(np.array([1.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    # Start 3: slight 3-fold
    fourier_starts.append(np.array([1.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0]))
    # Start 4: slight 7-fold (matches n=14)
    fourier_starts.append(np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0]))
    # Start 5: with small jitter
    fourier_starts.append(np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02]))
    
    for fstart in fourier_starts:
        if time_left() < 25:
            break
        try:
            res = minimize(_objective_fourier, fstart, method='Nelder-Mead',
                           options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True})
            if res.fun < best_obj:
                pts_f = _decode_fourier(res.x, n_harmonics)
                update_best(pts_f)
        except Exception:
            pass
    
    # === Phase 2: Multi-start Nelder-Mead on full 28D space from best configs ===
    if best_pts is not None:
        # Use best config as starting point
        if time_left() > 30:
            try:
                res = minimize(_objective, best_pts.flatten(), method='Nelder-Mead',
                               options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True})
                if res.fun < best_obj:
                    update_best(res.x.reshape(N, 2))
            except Exception:
                pass
        
        # Jittered versions of best config
        rng = np.random.default_rng(123)
        for std in [0.05, 0.1, 0.15, 0.2]:
            if time_left() < 20:
                break
            jittered = best_pts + rng.normal(0, std, (N, 2))
            try:
                res = minimize(_objective, jittered.flatten(), method='Nelder-Mead',
                               options={'maxiter': 2000, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True})
                if res.fun < best_obj:
                    update_best(res.x.reshape(N, 2))
            except Exception:
                pass
    
    # === Phase 3: Powell method for final polish ===
    if best_pts is not None and time_left() > 15:
        try:
            res_powell = minimize(_objective, best_pts.flatten(), method='Powell',
                                  options={'maxiter': 2000, 'xtol': 1e-12, 'ftol': 1e-14})
            if res_powell.fun < best_obj:
                update_best(res_powell.x.reshape(N, 2))
        except Exception:
            pass
    
    # === Phase 4: Coarse-to-fine Fourier sweep ===
    if time_left() > 20:
        # Sweep over a0 and first harmonic
        for a0 in [0.9, 1.0, 1.1]:
            for b1 in [-0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15]:
                for a1 in [-0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15]:
                    if time_left() < 15:
                        break
                    params_sweep = np.array([a0, b1, a1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                    try:
                        res = minimize(_objective_fourier, params_sweep, method='Nelder-Mead',
                                       options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
                        if res.fun < best_obj:
                            pts_s = _decode_fourier(res.x, n_harmonics)
                            update_best(pts_s)
                    except Exception:
                        pass
                if time_left() < 15:
                    break
            if time_left() < 15:
                break
    
    # === Phase 5: Final tight polish ===
    if best_pts is not None and time_left() > 5:
        try:
            res = minimize(_objective, best_pts.flatten(), method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-11, 'fatol': 1e-13, 'adaptive': True})
            if res.fun < best_obj:
                update_best(res.x.reshape(N, 2))
        except Exception:
            pass
    
    # Fallback
    if best_pts is None:
        best_pts = _regular_polygon(N)
    
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
