import numpy as np
from scipy.optimize import minimize, basinhopping, differential_evolution
from math import cos, sin, pi, log, exp
import time

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

def _two_squares_model(params):
    """
    3-parameter model: two concentric squares.
    params: [r1, r2, theta_diff]
    Square 1: radius r1, angle offset 0
    Square 2: radius r2, angle offset theta_diff
    """
    r1, r2, theta_diff = params
    if r1 < 1e-5 or r2 < 1e-5:
        return None
    
    angles1 = np.array([0, pi/2, pi, 3*pi/2])
    pts1 = np.column_stack([r1 * cos(angles1), r1 * sin(angles1)])
    
    angles2 = theta_diff + np.array([0, pi/2, pi, 3*pi/2])
    pts2 = np.column_stack([r2 * cos(angles2), r2 * sin(angles2)])
    
    return np.vstack([pts1, pts2])

def _two_squares_neglog(params):
    pts = _two_squares_model(params)
    if pts is None:
        return 1e10
    return _neg_log_score(pts)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    n = 8
    best_score = 0.0
    best_pts = None
    
    # Regular octagon as baseline
    oct_pts = np.array([[cos(2 * pi * k / 8), sin(2 * pi * k / 8)] for k in range(8)])
    best_score = _score(oct_pts)
    best_pts = oct_pts.copy()
    
    rng = np.random.RandomState(42)
    
    # === PHASE 1: Optimize the Two-Squares Model ===
    # This provides a strong, symmetric starting point.
    if time.time() < deadline - 50:
        try:
            # Initial guess: r1=1, r2=1, theta_diff=pi/4 (regular octagon)
            # But we can explore other radii.
            initial_guess = [1.0, 1.0, pi/4]
            
            # Use Powell for robustness on the 3D manifold
            res_sq = minimize(_two_squares_neglog, initial_guess, method='Powell',
                              options={'maxiter': 1000, 'xtol': 1e-10, 'ftol': 1e-10})
            
            if res_sq.fun < 1e6:
                pts_sq = _two_squares_model(res_sq.x)
                if pts_sq is not None:
                    sc = _score(pts_sq)
                    if sc > best_score:
                        best_score = sc
                        best_pts = pts_sq.copy()
        except Exception:
            pass

    # === PHASE 2: Basin Hopping in Full 16D Space ===
    # Basin Hopping is effective for finding global minima in high-dimensional spaces
    # with multiple local minima.
    
    def objective_16d(x):
        pts = x.reshape(n, 2)
        return _neg_log_score(pts)
        
    if time.time() < deadline - 20:
        try:
            # Use the best point from Phase 1 as the starting point
            x0 = best_pts.flatten()
            
            # Basin Hopping parameters
            # niter: number of iterations
            # miniter: max iterations for local minimization
            # stepsize: perturbation size
            bh_result = basinhopping(
                objective_16d, 
                x0, 
                niter=50, 
                miniter=100, 
                stepsize=0.5, 
                minimizer_kwargs={'method': 'L-BFGS-B', 'options': {'maxiter': 100}},
                seed=42
            )
            
            if bh_result.fun < 1e6:
                opt_pts = bh_result.x.reshape(n, 2)
                sc = _score(opt_pts)
                if sc > best_score:
                    best_score = sc
                    best_pts = opt_pts.copy()
        except Exception:
            pass

    # === PHASE 3: High-Precision Polishing ===
    # Use derivative-free methods for the final polish to handle any non-smoothness
    # from the diameter scaling.
    
    if time.time() < deadline - 5:
        try:
            # Powell Method
            res_powell = minimize(objective_16d, best_pts.flatten(), method='Powell',
                                  options={'maxiter': 500, 'xtol': 1e-12, 'ftol': 1e-12})
            if res_powell.fun < 1e6:
                opt_pts = res_powell.x.reshape(n, 2)
                sc = _score(opt_pts)
                if sc > best_score:
                    best_score = sc
                    best_pts = opt_pts.copy()
        except Exception:
            pass
            
        if time.time() < deadline - 2:
            try:
                # Nelder-Mead Method
                res_nm = minimize(objective_16d, best_pts.flatten(), method='Nelder-Mead',
                                  options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-12})
                if res_nm.fun < 1e6:
                    opt_pts = res_nm.x.reshape(n, 2)
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
