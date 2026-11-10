"""Program to maximize product of distances for 8 planar points at fixed diameter."""
# EVOLVE-BLOCK-START
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
    log_prod = np.sum(np.log(dists_scaled[mask]))
    score = np.exp(log_prod) / (n ** n)
    return score

def _neg_log_score(points):
    """Negative log of the score for minimization."""
    n = len(points)
    pts = np.array(points, dtype=np.float64)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    diam = np.max(dists)
    if diam == 0:
        return 1e10
    scale = 2.0 / diam
    dists_scaled = dists * scale
    mask = ~np.eye(n, dtype=bool)
    log_prod = np.sum(np.log(dists_scaled[mask]))
    return n * log(n) - log_prod

def _six_param_model(params):
    """
    6-parameter model: two concentric rotated squares with independent radii, orientations,
    and independent x/y offsets for each square (though we link offsets to reduce complexity,
    here we allow independent offsets for more flexibility).
    
    Actually, to keep it manageable and symmetric enough:
    Square 1: 4 points at angles theta1 + k*pi/2, radius r1, center (cx1, cy1)
    Square 2: 4 points at angles theta2 + k*pi/2, radius r2, center (cx2, cy2)
    
    To reduce parameters while allowing deformation:
    We can assume the two squares are centered at (0, y_off) and (0, -y_off) or similar.
    However, the 'inspiration' mentioned a 6-param model adding x-offset.
    Let's try:
    r1, r2, theta1, theta2, x_off, y_off
    
    Square 1 center: (x_off, y_off)
    Square 2 center: (-x_off, -y_off)  -> This preserves point symmetry around origin if r1=r2, theta1=theta2.
    If r1 != r2 or theta1 != theta2, it breaks symmetry.
    
    Let's use this model:
    P1..P4 on circle centered at (x_off, y_off) with radius r1, angle offset theta1.
    P5..P8 on circle centered at (-x_off, -y_off) with radius r2, angle offset theta2.
    """
    r1, r2, theta1, theta2, x_off, y_off = params
    if r1 < 0.01 or r2 < 0.01:
        return None
    
    # Square 1
    angles1 = theta1 + np.array([0, pi/2, pi, 3*pi/2])
    pts1 = np.column_stack([r1 * cos(angles1) + x_off, r1 * sin(angles1) + y_off])
    
    # Square 2
    angles2 = theta2 + np.array([0, pi/2, pi, 3*pi/2])
    pts2 = np.column_stack([r2 * cos(angles2) - x_off, r2 * sin(angles2) - y_off])
    
    return np.vstack([pts1, pts2])

def _six_param_neglog(params):
    """Negative log score for 6-parameter model."""
    pts = _six_param_model(params)
    if pts is None:
        return 1e10
    return _neg_log_score(pts)

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin
    n = 8
    best_score = 0.0
    best_pts = None
    
    # Regular octagon as baseline
    oct_pts = np.array([[cos(2 * pi * k / 8), sin(2 * pi * k / 8)] for k in range(8)])
    best_score = _score(oct_pts)
    best_pts = oct_pts.copy()
    
    rng = np.random.RandomState(42)
    
    # === PHASE 1: Differential evolution on 6-parameter model ===
    if time.time() < deadline - 60:
        # bounds for [r1, r2, theta1, theta2, x_offset, y_offset]
        # Radii should be around 1 for diameter 2
        # Offsets should be small
        bounds = [(0.1, 2.0), (0.1, 2.0), (0, pi/2), (0, pi/2), (-1.0, 1.0), (-1.0, 1.0)]
        
        try:
            de_result = differential_evolution(
                _six_param_neglog, bounds,
                seed=42, maxiter=300, popsize=25,
                tol=1e-10, polish=True,
                workers=1
            )
            if de_result.fun < 1e6:
                pts = _six_param_model(de_result.x)
                if pts is not None:
                    sc = _score(pts)
                    if sc > best_score:
                        best_score = sc
                        best_pts = pts.copy()
        except Exception:
            pass
    
    # Try a second DE run with different seed if time permits
    if time.time() < deadline - 40:
        try:
            de_result2 = differential_evolution(
                _six_param_neglog, bounds,
                seed=123, maxiter=200, popsize=20,
                tol=1e-10, polish=True,
                workers=1
            )
            if de_result2.fun < 1e6:
                pts = _six_param_model(de_result2.x)
                if pts is not None:
                    sc = _score(pts)
                    if sc > best_score:
                        best_score = sc
                        best_pts = pts.copy()
        except Exception:
            pass

    # === PHASE 2: Local optimization on full 16-dim space ===
    # We use the best point found on the manifold as a starting point for full optimization.
    # This allows breaking the square symmetry if it helps.
    
    def objective(x):
        pts_x = x.reshape(n, 2)
        return _neg_log_score(pts_x)
    
    # Constraint: diameter <= 2
    def constraint_diameter(x):
        pts_x = x.reshape(n, 2)
        diff = pts_x[:, np.newaxis, :] - pts_x[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        return 2.0 - np.max(dists)
    
    constraints = [{'type': 'ineq', 'fun': constraint_diameter}]
    
    # Initial optimization from best_pts
    if time.time() < deadline - 20:
        try:
            result = minimize(objective, best_pts.flatten(), method='SLSQP',
                            constraints=constraints, options={'maxiter': 1000, 'ftol': 1e-14})
            if result.fun < 1e6:
                opt_pts = result.x.reshape(n, 2)
                sc = _score(opt_pts)
                if sc > best_score:
                    best_score = sc
                    best_pts = opt_pts.copy()
        except Exception:
            pass
        
        # L-BFGS-B from best_pts (unconstrained, but usually converges to similar if scaled well)
        try:
            result = minimize(objective, best_pts.flatten(), method='L-BFGS-B',
                            options={'maxiter': 1000, 'ftol': 1e-14})
            if result.fun < 1e6:
                opt_pts = result.x.reshape(n, 2)
                sc = _score(opt_pts)
                if sc > best_score:
                    best_score = sc
                    best_pts = opt_pts.copy()
        except Exception:
            pass

    # === PHASE 3: Perturbation-based polish in full space ===
    if time.time() < deadline - 5:
        for trial in range(20):
            if time.time() > deadline - 2:
                break
            noise = rng.randn(n, 2) * 0.02
            perturbed = best_pts + noise
            try:
                result = minimize(objective, perturbed.flatten(), method='SLSQP',
                                constraints=constraints, options={'maxiter': 500, 'ftol': 1e-15})
                if result.fun < 1e6:
                    opt_pts = result.x.reshape(n, 2)
                    sc = _score(opt_pts)
                    if sc > best_score:
                        best_score = sc
                        best_pts = opt_pts.copy()
            except Exception:
                pass

    # Final check for coordinate bounds
    if np.any(np.abs(best_pts) >= 1e6):
        best_pts = oct_pts.copy()

    return best_pts.tolist()
# EVOLVE-BLOCK-END
