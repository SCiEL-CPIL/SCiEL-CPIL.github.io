"""Program to maximize the product of distances for 8 planar points with fixed diameter."""
import numpy as np
import time
from math import cos, sin, pi, log, sqrt, exp
from scipy.optimize import minimize, differential_evolution

def _score(points):
    """Compute Delta / n^n where Delta is product of ordered-pair distances after scaling to diameter 2."""
    n = len(points)
    pts = np.array(points, dtype=np.float64)
    if n < 2:
        return 0.0
    
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    diam = np.max(dists)
    if diam == 0:
        return 0.0
    
    scale = 2.0 / diam
    dists_scaled = dists * scale
    
    mask = ~np.eye(n, dtype=bool)
    if np.any(dists_scaled[mask] < 1e-12):
        return 0.0
        
    log_prod = np.sum(np.log(dists_scaled[mask]))
    score = np.exp(log_prod) / (n ** n)
    return score

def _neg_log_score(pts):
    """Negative log of the score for minimization."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    diam = np.max(dists)
    if diam < 1e-12:
        return 1e10
    scale = 2.0 / diam
    dists_scaled = dists * scale
    mask = ~np.eye(n, dtype=bool)
    if np.any(dists_scaled[mask] < 1e-12):
        return 1e10
    log_prod = np.sum(np.log(dists_scaled[mask]))
    return n * log(n) - log_prod

def _angular_modulation_model(params):
    """
    8 points as a regular octagon with angular and radial modulation.
    params = [R, a, m, phi, psi, delta]
    - R: base radius
    - a: radial modulation amplitude
    - m: radial harmonic order (2 or 4)
    - phi: radial modulation phase
    - psi: angular modulation phase
    - delta: angular modulation amplitude
    Points: theta_k = 2*pi*k/8 + delta*sin(2*theta_k + psi)
            r_k = R * (1 + a*cos(m*theta_k + phi))
    """
    R, a, m, phi, psi, delta = params
    if R < 0.05:
        return None
    
    base_angles = 2 * pi * np.arange(8) / 8
    # Angular modulation
    angles = base_angles + delta * sin(2 * base_angles + psi)
    # Radial modulation
    radii = R * (1 + a * cos(m * angles + phi))
    
    # Ensure positive radii
    if np.any(radii < 0.01):
        return None
    
    pts = np.column_stack([radii * cos(angles), radii * sin(angles)])
    return pts

def _angular_modulation_neglog(params):
    """Negative log score for angular modulation model."""
    pts = _angular_modulation_model(params)
    if pts is None:
        return 1e10
    return _neg_log_score(pts)

def _centrally_symmetric_model(params):
    """
    Generate 8 points as 4 unique points and their negations.
    params = 8 floats representing 4 points (x1,y1, x2,y2, x3,y3, x4,y4)
    """
    pts_half = params.reshape(4, 2)
    pts = np.vstack([pts_half, -pts_half])
    return pts

def _centrally_symmetric_neglog(params):
    """Negative log score for centrally symmetric model."""
    pts = _centrally_symmetric_model(params)
    return _neg_log_score(pts)

def _two_squares_model(params):
    """
    Generate 8 points: two concentric squares.
    params = [r1, r2, theta1, theta2]
    """
    r1, r2, theta1, theta2 = params
    angles1 = theta1 + np.array([0, pi/2, pi, 3*pi/2])
    angles2 = theta2 + np.array([0, pi/2, pi, 3*pi/2])
    pts1 = np.column_stack([r1 * cos(angles1), r1 * sin(angles1)])
    pts2 = np.column_stack([r2 * cos(angles2), r2 * sin(angles2)])
    return np.vstack([pts1, pts2])

def _two_squares_neglog(params):
    """Negative log score for two-squares model."""
    pts = _two_squares_model(params)
    return _neg_log_score(pts)

def _polish_16d(pts, deadline, max_time_per_method=8.0):
    """
    Apply multiple local optimization methods to polish 16D point configuration.
    Returns the best result found.
    """
    best_pts = pts.copy()
    best_val = _neg_log_score(pts)
    
    def _obj(x_flat):
        return _neg_log_score(x_flat.reshape(8, 2))
    
    # SLSQP
    if time.time() < deadline - 5.0:
        try:
            res = minimize(_obj, best_pts.flatten(), method='SLSQP',
                          options={'maxiter': 500, 'ftol': 1e-15})
            if res.fun < best_val - 1e-15:
                best_val = res.fun
                best_pts = res.x.reshape(8, 2)
        except Exception:
            pass
    
    # Powell
    if time.time() < deadline - 5.0:
        try:
            res = minimize(_obj, best_pts.flatten(), method='Powell',
                          options={'maxiter': 2000, 'ftol': 1e-15, 'xtol': 1e-15})
            if res.fun < best_val - 1e-15:
                best_val = res.fun
                best_pts = res.x.reshape(8, 2)
        except Exception:
            pass
    
    # Nelder-Mead with tight tolerance
    if time.time() < deadline - 3.0:
        try:
            res = minimize(_obj, best_pts.flatten(), method='Nelder-Mead',
                          options={'xatol': 1e-14, 'fatol': 1e-14,
                                   'maxiter': 50000, 'maxfev': 50000})
            if res.fun < best_val - 1e-15:
                best_val = res.fun
                best_pts = res.x.reshape(8, 2)
        except Exception:
            pass
    
    # Final ultra-tight Nelder-Mead
    if time.time() < deadline - 1.0:
        try:
            res = minimize(_obj, best_pts.flatten(), method='Nelder-Mead',
                          options={'xatol': 1e-16, 'fatol': 1e-16,
                                   'maxiter': 100000, 'maxfev': 100000})
            if res.fun < best_val - 1e-15:
                best_val = res.fun
                best_pts = res.x.reshape(8, 2)
        except Exception:
            pass
    
    return best_pts

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    n = 8
    best_score = 0.0
    best_pts = None
    
    # Regular octagon as baseline
    oct_pts = np.array([[cos(2 * pi * k / 8), sin(2 * pi * k / 8)] for k in range(8)])
    best_score = _score(oct_pts)
    best_pts = oct_pts.copy()
    
    # === PHASE 1: Angular modulation model (6 params) - primary search ===
    if time.time() < deadline - 60:
        # Multiple DE runs with different seeds on the angular modulation model
        for seed in [42, 123, 456, 789]:
            if time.time() > deadline - 20:
                break
            bounds_am = [(0.5, 2.0), (-0.3, 0.3), (2.0, 5.0), (0, 2*pi), (0, 2*pi), (-0.2, 0.2)]
            try:
                de_res = differential_evolution(
                    _angular_modulation_neglog, bounds_am,
                    seed=seed, maxiter=120, popsize=25,
                    tol=1e-12, polish=True, init='sobol'
                )
                if de_res.fun < 1e6:
                    pts = _angular_modulation_model(de_res.x)
                    if pts is not None:
                        sc = _score(pts)
                        if sc > best_score:
                            best_score = sc
                            best_pts = pts.copy()
            except Exception:
                pass
        
        # Local polish on the angular modulation parameters
        if time.time() < deadline - 25:
            # Start from regular octagon in parameter space
            x0 = [1.0, 0.0, 4.0, 0.0, 0.0, 0.0]
            for method in ['Nelder-Mead', 'Powell']:
                if time.time() > deadline - 10:
                    break
                try:
                    res = minimize(_angular_modulation_neglog, x0, method=method,
                                  options={'maxiter': 20000, 'fatol': 1e-15, 'xatol': 1e-15})
                    if res.fun < 1e6:
                        pts = _angular_modulation_model(res.x)
                        if pts is not None:
                            sc = _score(pts)
                            if sc > best_score:
                                best_score = sc
                                best_pts = pts.copy()
                            x0 = res.x  # Use result as next starting point
                except Exception:
                    pass
    
    # === PHASE 2: Centrally symmetric 8D model ===
    if time.time() < deadline - 40:
        bounds_cs = [(-2.0, 2.0)] * 8
        for seed in [42, 123]:
            if time.time() > deadline - 15:
                break
            try:
                de_res = differential_evolution(
                    _centrally_symmetric_neglog, bounds_cs,
                    seed=seed, maxiter=100, popsize=20,
                    tol=1e-12, polish=True
                )
                if de_res.fun < 1e6:
                    pts = _centrally_symmetric_model(de_res.x)
                    sc = _score(pts)
                    if sc > best_score:
                        best_score = sc
                        best_pts = pts.copy()
            except Exception:
                pass
    
    # === PHASE 3: Two-squares 4D model ===
    if time.time() < deadline - 30:
        bounds_2s = [(0.1, 3.0), (0.1, 3.0), (0, pi/2), (0, pi/2)]
        try:
            de_res = differential_evolution(
                _two_squares_neglog, bounds_2s,
                seed=42, maxiter=150, popsize=25,
                tol=1e-12, polish=True, init='sobol'
            )
            if de_res.fun < 1e6:
                pts = _two_squares_model(de_res.x)
                sc = _score(pts)
                if sc > best_score:
                    best_score = sc
                    best_pts = pts.copy()
        except Exception:
            pass
    
    # === PHASE 4: Full 16D DE search (broader exploration) ===
    if time.time() < deadline - 25:
        bounds_16d = [(-2.0, 2.0)] * 16
        def _de_16d(x_flat):
            if time.time() > deadline - 5:
                return 1e10
            pts = x_flat.reshape(8, 2)
            return _neg_log_score(pts)
        
        try:
            de_res = differential_evolution(
                _de_16d, bounds_16d,
                seed=42, maxiter=60, popsize=30,
                tol=1e-10, polish=False, init='sobol'
            )
            if de_res.fun < 1e6:
                pts = de_res.x.reshape(8, 2)
                sc = _score(pts)
                if sc > best_score:
                    best_score = sc
                    best_pts = pts.copy()
        except Exception:
            pass
    
    # === PHASE 5: Perturbation-based restarts from best found ===
    rng = np.random.RandomState(42)
    if time.time() < deadline - 15 and best_pts is not None:
        def _obj_16d(x_flat):
            return _neg_log_score(x_flat.reshape(8, 2))
        
        for trial in range(20):
            if time.time() > deadline - 5:
                break
            for noise_scale in [0.1, 0.05, 0.02]:
                if time.time() > deadline - 3:
                    break
                noise = rng.randn(8, 2) * noise_scale
                perturbed = best_pts + noise
                try:
                    res = minimize(_obj_16d, perturbed.flatten(), method='Nelder-Mead',
                                  options={'maxiter': 5000, 'fatol': 1e-14, 'xatol': 1e-14})
                    if res.fun < 1e6:
                        opt_pts = res.x.reshape(8, 2)
                        sc = _score(opt_pts)
                        if sc > best_score:
                            best_score = sc
                            best_pts = opt_pts.copy()
                except Exception:
                    pass
    
    # === PHASE 6: Final high-precision polish ===
    if time.time() < deadline - 2 and best_pts is not None:
        best_pts = _polish_16d(best_pts, deadline)
        sc = _score(best_pts)
        if sc > best_score:
            best_score = sc
    
    # Fallback: Regular Octagon
    if best_pts is None:
        best_pts = np.array([[cos(2 * pi * k / 8), sin(2 * pi * k / 8)] for k in range(8)])
        best_score = _score(best_pts)
        
    return best_pts.tolist()
