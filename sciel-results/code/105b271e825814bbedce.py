import time
import math
import numpy as np
from scipy.optimize import minimize

def construct():
    n = 8
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    def compute_score(pts):
        """Compute the score for a set of points."""
        pts = np.asarray(pts)
        if not np.all(np.isfinite(pts)):
            return 0.0
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        iu = np.triu_indices(n, k=1)
        d = dists[iu]
        if np.min(d) < 1e-12:
            return 0.0
        D = np.max(d)
        if D < 1e-12:
            return 0.0
        # Scale to diameter 2
        scaled_d = d * (2.0 / D)
        prod_unordered = np.prod(scaled_d)
        Delta = prod_unordered ** 2  # Product over ordered pairs
        return Delta / (n ** n)

    def objective_general(x):
        """
        Objective for general optimization.
        Minimizes -sum(log(d_ij)) + penalty for diameter > 2.
        """
        pts = x.reshape(n, 2)
        if not np.all(np.isfinite(pts)):
            return 1e10
        
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        iu = np.triu_indices(n, k=1)
        d = dists[iu]
        
        if np.min(d) < 1e-10:
            return 1e10
            
        D = np.max(d)
        if D < 1e-10:
            return 1e10
            
        # Maximize sum(log(d)) -> Minimize -sum(log(d))
        log_sum = np.sum(np.log(d))
        
        # Penalty for exceeding diameter 2
        # We use a smooth max approximation or just a hard penalty
        # Since we want to stay within diameter 2, and the optimal is on the boundary,
        # a penalty of (D-2)^2 if D>2 is good.
        # However, if D < 2, we don't penalize, but the log_sum will push D to increase.
        # This works because scaling up increases log_sum.
        penalty = 0.0
        if D > 2.0:
            penalty = 100.0 * (D - 2.0)**2
            
        return -log_sum + penalty

    def objective_two_squares(params):
        """
        Objective for two concentric squares configuration.
        params: [r1, r2, theta]
        """
        r1, r2, theta = params
        if r1 < 0 or r2 < 0:
            return 1e10
            
        # Square 1: angles 0, pi/2, pi, 3pi/2
        angles1 = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
        pts1 = np.column_stack([r1 * np.cos(angles1), r1 * np.sin(angles1)])
        
        # Square 2: angles theta, theta+pi/2, ...
        angles2 = angles1 + theta
        pts2 = np.column_stack([r2 * np.cos(angles2), r2 * np.sin(angles2)])
        
        pts = np.vstack([pts1, pts2])
        
        # Reuse logic from objective_general but without penalty if we constrain r1, r2 such that diameter <= 2?
        # Diameter is max of:
        # - r1*sqrt(2) (diag of sq1)
        # - r2*sqrt(2) (diag of sq2)
        # - dist between points in sq1 and sq2
        # We can just compute the score and return negative log score?
        # No, we want to maximize the actual score.
        # Let's just return -log(score) effectively.
        
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        iu = np.triu_indices(n, k=1)
        d = dists[iu]
        if np.min(d) < 1e-10:
            return 1e10
        D = np.max(d)
        if D < 1e-10:
            return 1e10
            
        # If D > 2, we should scale down? 
        # The problem is fixed diameter 2.
        # If we parameterize by r1, r2, the resulting set has diameter D.
        # The score is invariant to scaling.
        # So we can just compute the log of the product of scaled distances.
        # log_score = 2 * (sum(log(d)) + 28*log(2) - 28*log(D) - 8*log(8))
        log_score = 2.0 * (np.sum(np.log(d)) + 28.0 * np.log(2.0) - 28.0 * np.log(D) - 8.0 * np.log(8.0))
        return -log_score

    best_score = 0.0
    best_pts = None

    def try_candidate(pts):
        nonlocal best_score, best_pts
        pts = np.asarray(pts)
        if not np.all(np.isfinite(pts)) or np.min(pts[:, np.newaxis, :] - pts[np.newaxis, :, :]) ** 2 < 1e-12:
            return 0.0
        s = compute_score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()
        return s

    # 1. Regular Octagon
    angles_reg = np.linspace(0, 2 * np.pi, n, endpoint=False)
    reg_pts = np.column_stack([np.cos(angles_reg), np.sin(angles_reg)])
    try_candidate(reg_pts)

    # 2. Optimize Two Concentric Squares
    # Initial guesses for r1, r2, theta
    # Due to symmetry, we can assume r1 <= r2 and theta in [0, pi/4]
    # The optimal is likely near r1 ~ 0.5-0.7, r2 ~ 1.0-1.2, theta ~ pi/4
    guesses = [
        [0.5, 1.0, np.pi/4],
        [0.6, 1.0, np.pi/4],
        [0.7, 1.0, np.pi/4],
        [0.7, 1.1, np.pi/4],
        [0.7, 1.2, np.pi/4],
        [0.8, 1.0, np.pi/4],
        [0.8, 1.1, np.pi/4],
        [0.9, 1.0, np.pi/4],
        [0.7, 1.0, np.pi/8],
        [0.7, 1.0, 3*np.pi/8],
        [0.6, 1.1, np.pi/4],
        [0.8, 1.2, np.pi/4],
    ]
    
    for guess in guesses:
        if time.time() > deadline - 15.0:
            break
        try:
            res = minimize(
                objective_two_squares,
                guess,
                method='L-BFGS-B',
                bounds=[(0.01, 1.5), (0.01, 1.5), (0.0, np.pi/4)],
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if res.success or res.fun < 1e9:
                r1, r2, theta = res.x
                angles1 = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
                angles2 = angles1 + theta
                pts1 = np.column_stack([r1 * np.cos(angles1), r1 * np.sin(angles1)])
                pts2 = np.column_stack([r2 * np.cos(angles2), r2 * np.sin(angles2)])
                pts = np.vstack([pts1, pts2])
                s = try_candidate(pts)
                
                # If we found a good score, do a tighter polish
                if s > 1.01 and time.time() < deadline - 5.0:
                    res2 = minimize(
                        objective_two_squares,
                        res.x,
                        method='L-BFGS-B',
                        bounds=[(0.01, 1.5), (0.01, 1.5), (0.0, np.pi/4)],
                        options={'maxiter': 1000, 'ftol': 1e-18, 'gtol': 1e-15}
                    )
                    if np.all(np.isfinite(res2.x)):
                        r1, r2, theta = res2.x
                        angles1 = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
                        angles2 = angles1 + theta
                        pts1 = np.column_stack([r1 * np.cos(angles1), r1 * np.sin(angles1)])
                        pts2 = np.column_stack([r2 * np.cos(angles2), r2 * np.sin(angles2)])
                        pts = np.vstack([pts1, pts2])
                        try_candidate(pts)
        except Exception:
            pass

    # 3. General Optimization with Penalty
    # Start from several random and structured initializations
    np.random.seed(42)
    inits = []
    
    # Random in disk
    for _ in range(5):
        r = np.random.rand(n) * 1.0
        theta = np.random.rand(n) * 2 * np.pi
        pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        inits.append(pts)
        
    # Perturbed Regular Octagon
    for _ in range(3):
        angles = angles_reg + 0.1 * np.random.randn(n)
        radii = 1.0 + 0.1 * np.random.randn(n)
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        inits.append(pts)

    # Two squares random init
    for _ in range(3):
        r1 = 0.5 + np.random.rand() * 0.5
        r2 = 0.5 + np.random.rand() * 0.5
        theta = np.random.rand() * np.pi/4
        angles1 = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
        angles2 = angles1 + theta
        pts1 = np.column_stack([r1 * np.cos(angles1), r1 * np.sin(angles1)])
        pts2 = np.column_stack([r2 * np.cos(angles2), r2 * np.sin(angles2)])
        inits.append(np.vstack([pts1, pts2]))

    for init in inits:
        if time.time() > deadline - 5.0:
            break
        try:
            x0 = init.flatten()
            # Bounds to keep points in a reasonable box
            bounds = [(-1.5, 1.5)] * 16
            res = minimize(
                objective_general,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-12}
            )
            if np.all(np.isfinite(res.x)):
                refined_pts = res.x.reshape(n, 2)
                # Check if this is a valid improvement
                # The objective minimized -log_sum + penalty.
                # If penalty is 0, it's a good candidate for the shape.
                # We must scale to diameter 2 for scoring.
                s = try_candidate(refined_pts)
                
                # If we found a good score, try a tighter refinement
                if s > 1.01:
                     if time.time() < deadline - 2.0:
                         res2 = minimize(
                             objective_general,
                             res.x,
                             method='L-BFGS-B',
                             bounds=bounds,
                             options={'maxiter': 200, 'ftol': 1e-14}
                         )
                         if np.all(np.isfinite(res2.x)):
                             refined_pts2 = res2.x.reshape(n, 2)
                             try_candidate(refined_pts2)
        except Exception:
            pass

    # Fallback
    if best_pts is None or not np.all(np.isfinite(best_pts)):
        best_pts = reg_pts

    # Final validation
    best_pts = np.asarray(best_pts)
    diffs = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    iu = np.triu_indices(n, k=1)
    d = dists[iu]
    if np.min(d) < 1e-6 or not np.all(np.isfinite(best_pts)):
        best_pts = reg_pts

    # Scale to ensure coords are small (optional, but good for stability)
    max_coord = np.max(np.abs(best_pts))
    if max_coord > 1e6:
        best_pts = best_pts / max_coord * 1e5

    return best_pts.tolist()
