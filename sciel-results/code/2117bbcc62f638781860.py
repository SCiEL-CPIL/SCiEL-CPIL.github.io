# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

def _score(pts):
    """Return (d_min/d_max)^2, or 0 on degenerate input."""
    try:
        if pts.shape != (14, 3):
            return 0.0
        if not np.all(np.isfinite(pts)):
            return 0.0
        d = pdist(pts)
        if d.size == 0:
            return 0.0
        dmin = d.min()
        dmax = d.max()
        if dmax <= 1e-12 or not np.isfinite(dmin) or not np.isfinite(dmax):
            return 0.0
        r = dmin / dmax
        return float(r * r)
    except Exception:
        return 0.0

def _build_points(params):
    """
    Build 14 points from 5 parameters:
    z_p: z-coordinate of pole 1 (pole 2 is -z_p)
    r_1, z_1: radius and z-height of ring 1 (6 points, phase 0)
    r_2, z_2: radius and z-height of ring 2 (6 points, phase pi/6)
    """
    z_p, r_1, z_1, r_2, z_2 = params
    points = np.zeros((14, 3))
    
    # Poles
    points[0] = [0.0, 0.0, z_p]
    points[1] = [0.0, 0.0, -z_p]
    
    # Ring 1: 6 points, phase 0
    angles1 = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    for i in range(6):
        points[2 + i] = [r_1 * np.cos(angles1[i]), r_1 * np.sin(angles1[i]), z_1]
        
    # Ring 2: 6 points, phase pi/6 (staggered)
    angles2 = angles1 + np.pi / 6.0
    for i in range(6):
        points[8 + i] = [r_2 * np.cos(angles2[i]), r_2 * np.sin(angles2[i]), z_2]
        
    return points

def _neg_score_params(params):
    """Negative score for minimization."""
    pts = _build_points(params)
    return -_score(pts)

def _smooth_score(pts, beta=50.0):
    """
    Smooth surrogate: use soft-min and soft-max of pairwise distances.
    soft_min = -1/beta * log(sum(exp(-beta*d)))
    soft_max = 1/beta * log(sum(exp(beta*d)))
    ratio = soft_min / soft_max
    return ratio^2
    """
    try:
        d = pdist(pts)
        if d.size == 0:
            return 0.0
        d_clip = np.clip(d, 1e-10, None)
        neg_beta_d = -beta * d_clip
        neg_beta_d -= neg_beta_d.max()
        soft_min = -1.0 / beta * np.log(np.sum(np.exp(neg_beta_d)))
        beta_d = beta * d_clip
        beta_d -= beta_d.max()
        soft_max = 1.0 / beta * np.log(np.sum(np.exp(beta_d)))
        if soft_max <= 1e-12:
            return 0.0
        r = soft_min / soft_max
        return float(r * r)
    except Exception:
        return 0.0

def _neg_smooth_score_params(params, beta=50.0):
    pts = _build_points(params)
    return -_smooth_score(pts, beta)

def min_max_dist_dim3_14() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590
    
    best_pts = None
    best_val = -1.0
    
    def _update_best(pts, val):
        nonlocal best_pts, best_val
        if val > best_val:
            best_val = val
            best_pts = pts.copy()
    
    # Initial candidates based on symmetric configurations
    # Tetrahedron-inspired: 4 vertices + 6 edge midpoints + 2 face centers + 2 extra
    # But we stick with our 5-param family: poles + 2 rings
    
    # Good initial guesses from known good configurations
    initial_guesses = [
        np.array([0.5, 0.8, 0.0, 0.8, 0.0]),
        np.array([0.7, 0.7, 0.2, 0.7, -0.2]),
        np.array([0.6, 0.9, 0.1, 0.9, -0.1]),
        np.array([0.4, 0.6, 0.3, 0.6, -0.3]),
        np.array([0.8, 1.0, 0.0, 1.0, 0.0]),
        np.array([0.3, 0.5, 0.5, 0.5, -0.5]),
    ]
    
    # Quick evaluation of initial guesses
    best_param = initial_guesses[0].copy()
    best_score = -1.0
    for guess in initial_guesses:
        s = _score(_build_points(guess))
        if s > best_score:
            best_score = s
            best_param = guess.copy()
    
    _update_best(_build_points(best_param), best_score)
    
    # Multi-stage smooth optimization with L-BFGS-B
    # Stage 1: Coarse smooth optimization (low beta)
    for beta in [10.0, 25.0, 50.0, 100.0, 200.0]:
        if time.time() > deadline - 30:
            break
        
        # Multiple restarts from perturbed best
        rng = np.random.default_rng(int(start_time * 1000) % 10000 + int(beta))
        n_restarts = 3 if beta < 50 else 2
        
        for restart in range(n_restarts):
            if time.time() > deadline - 20:
                break
            
            if restart == 0:
                x0 = best_param.copy()
            else:
                perturb = rng.normal(0, 0.1, 5)
                x0 = best_param + perturb
                x0[0] = np.clip(x0[0], 0.0, 1.5)
                x0[1] = np.clip(x0[1], 0.1, 1.5)
                x0[2] = np.clip(x0[2], -1.5, 1.5)
                x0[3] = np.clip(x0[3], 0.1, 1.5)
                x0[4] = np.clip(x0[4], -1.5, 1.5)
            
            try:
                res = minimize(
                    lambda p: _neg_smooth_score_params(p, beta),
                    x0,
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
                )
                s = _score(_build_points(res.x))
                if s > best_score:
                    best_score = s
                    best_param = res.x.copy()
                _update_best(_build_points(best_param), best_score)
            except Exception:
                pass
    
    # Stage 2: Exact local optimization with Nelder-Mead (non-smooth)
    if time.time() < deadline - 15:
        try:
            res = minimize(
                _neg_score_params,
                best_param,
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            s = _score(_build_points(res.x))
            if s > best_score:
                best_score = s
                best_param = res.x.copy()
            _update_best(_build_points(best_param), best_score)
        except Exception:
            pass
    
    # Stage 3: Powell polish for high precision
    if time.time() < deadline - 10:
        try:
            res = minimize(
                _neg_score_params,
                best_param,
                method='Powell',
                options={'maxiter': 2000, 'xtol': 1e-15, 'ftol': 1e-17}
            )
            s = _score(_build_points(res.x))
            if s > best_score:
                best_score = s
                best_param = res.x.copy()
            _update_best(_build_points(best_param), best_score)
        except Exception:
            pass
    
    # Stage 4: Final random restarts with exact optimization
    if time.time() < deadline - 5:
        rng = np.random.default_rng(42)
        for _ in range(3):
            if time.time() > deadline - 3:
                break
            perturb = rng.normal(0, 0.03, 5)
            perturbed = best_param + perturb
            perturbed[0] = np.clip(perturbed[0], 0.0, 1.5)
            perturbed[1] = np.clip(perturbed[1], 0.1, 1.5)
            perturbed[2] = np.clip(perturbed[2], -1.5, 1.5)
            perturbed[3] = np.clip(perturbed[3], 0.1, 1.5)
            perturbed[4] = np.clip(perturbed[4], -1.5, 1.5)
            try:
                res = minimize(
                    _neg_score_params,
                    perturbed,
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
                )
                s = _score(_build_points(res.x))
                if s > best_score:
                    best_score = s
                    best_param = res.x.copy()
                _update_best(_build_points(best_param), best_score)
            except Exception:
                pass
    
    # Final validation
    if best_pts is None or not np.all(np.isfinite(best_pts)):
        best_pts = _build_points(np.array([0.5, 0.8, 0.0, 0.8, 0.0]))
    
    return best_pts

# EVOLVE-BLOCK-END
