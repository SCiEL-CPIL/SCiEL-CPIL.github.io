import time
import math
import numpy as np
from scipy.optimize import minimize

def construct():
    n = 8
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    def get_points(params):
        """
        params: [log_r1, theta1, log_r2, theta2, log_r3, theta3, log_r4, theta4]
        Returns 8 points as (8, 2) array.
        """
        pts = np.zeros((8, 2))
        for k in range(4):
            log_r = params[2*k]
            theta = params[2*k + 1]
            r = np.exp(log_r)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            pts[2*k] = [x, y]
            pts[2*k + 1] = [x, -y]
        return pts

    def objective(params):
        """
        Minimizes - (sum(log(d_ij)) - 14*log(D))
        which is equivalent to maximizing the score.
        """
        if not np.all(np.isfinite(params)):
            return 1e10
        
        pts = get_points(params)
        if not np.all(np.isfinite(pts)):
            return 1e10
            
        # Compute pairwise distances
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        
        # Get upper triangle indices
        iu = np.triu_indices(n, k=1)
        d = dists[iu]
        
        if np.min(d) < 1e-12:
            return 1e10
            
        D = np.max(d)
        if D < 1e-12:
            return 1e10
            
        # log_sum = sum(log(d))
        # We want to maximize sum(log(d)) - 14*log(D)
        # So minimize -(sum(log(d)) - 14*log(D))
        log_sum = np.sum(np.log(d))
        val = -(log_sum - 14.0 * np.log(D))
        
        return val

    def compute_score(pts):
        """Compute the actual score for reporting."""
        pts = np.asarray(pts)
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        iu = np.triu_indices(n, k=1)
        d = dists[iu]
        if np.min(d) < 1e-12:
            return 0.0
        D = np.max(d)
        if D < 1e-12:
            return 0.0
        scaled_d = d * (2.0 / D)
        prod_unordered = np.prod(scaled_d)
        Delta = prod_unordered ** 2
        return Delta / (n ** n)

    best_score = 0.0
    best_params = None
    best_pts = None

    def try_candidate(params):
        nonlocal best_score, best_params, best_pts
        pts = get_points(params)
        if not np.all(np.isfinite(pts)):
            return 0.0
        # Check distinctness
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        iu = np.triu_indices(n, k=1)
        d = dists[iu]
        if np.min(d) < 1e-8:
            return 0.0
        s = compute_score(pts)
        if s > best_score:
            best_score = s
            best_params = params.copy()
            best_pts = pts.copy()
        return s

    # Initial guesses
    inits = []
    
    # 1. Regular Octagon
    # Radii 1 (log_r = 0), Angles 0, pi/4, pi/2, 3pi/4
    inits.append([0, 0, 0, np.pi/4, 0, np.pi/2, 0, 3*np.pi/4])
    
    # 2. Perturbed Regular Octagon
    np.random.seed(42)
    for _ in range(10):
        p = np.array([0, 0, 0, np.pi/4, 0, np.pi/2, 0, 3*np.pi/4])
        p += np.random.randn(8) * 0.1
        inits.append(p)
        
    # 3. Two Squares: Radii r1, r2. Angles 0, pi/2 for both?
    # Pairs: (r1, 0), (r1, pi/2), (r2, 0), (r2, pi/2)
    # This creates points on axes.
    for r1 in [0.5, 0.8, 1.0, 1.2]:
        for r2 in [0.5, 0.8, 1.0, 1.2]:
            if abs(r1 - r2) < 1e-3: continue
            inits.append([np.log(r1), 0, np.log(r1), np.pi/2, np.log(r2), 0, np.log(r2), np.pi/2])
            
    # 4. Random
    for _ in range(10):
        inits.append(np.random.randn(8) * 0.5)

    # Optimization
    for init in inits:
        if time.time() > deadline - 10.0:
            break
        try:
            res = minimize(
                objective,
                init,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if res.fun < 1e9:
                s = try_candidate(res.x)
                
                # Polish with Nelder-Mead if good
                if s > 1.0 and time.time() < deadline - 5.0:
                    res2 = minimize(
                        objective,
                        res.x,
                        method='Nelder-Mead',
                        options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12}
                    )
                    if res2.fun < 1e9:
                        try_candidate(res2.x)
        except Exception:
            pass

    # Fallback
    if best_pts is None or not np.all(np.isfinite(best_pts)):
        angles_reg = np.linspace(0, 2 * np.pi, n, endpoint=False)
        best_pts = np.column_stack([np.cos(angles_reg), np.sin(angles_reg)])

    # Final validation
    best_pts = np.asarray(best_pts)
    diffs = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    iu = np.triu_indices(n, k=1)
    d = dists[iu]
    if np.min(d) < 1e-6 or not np.all(np.isfinite(best_pts)):
        angles_reg = np.linspace(0, 2 * np.pi, n, endpoint=False)
        best_pts = np.column_stack([np.cos(angles_reg), np.sin(angles_reg)])

    # Scale to ensure coords are small
    max_coord = np.max(np.abs(best_pts))
    if max_coord > 1e6:
        best_pts = best_pts / max_coord * 1e5
    elif max_coord < 1e-6 and max_coord > 0:
        best_pts = best_pts / max_coord * 1.0

    return best_pts.tolist()
