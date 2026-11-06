import time
import math
import numpy as np
from scipy.optimize import minimize, differential_evolution

def construct():
    n = 8
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    def compute_score(pts):
        """Compute the score for a set of points."""
        pts = np.asarray(pts, dtype=np.float64)
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
        penalty = 0.0
        if D > 2.0:
            penalty = 100.0 * (D - 2.0)**2
            
        return -log_sum + penalty

    best_score = 0.0
    best_pts = None

    def try_candidate(pts):
        nonlocal best_score, best_pts
        pts = np.asarray(pts, dtype=np.float64)
        if not np.all(np.isfinite(pts)):
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

    # 2. Two Concentric Squares Strategy (from Inspiration)
    # This structure often outperforms regular polygons for n=8.
    # We parameterize as two squares with radii r1, r2 and relative rotation theta.
    
    def objective_two_squares(params):
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
        
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        iu = np.triu_indices(n, k=1)
        d = dists[iu]
        if np.min(d) < 1e-10:
            return 1e10
        D = np.max(d)
        if D < 1e-10:
            return 1e10
            
        # Maximize log product of scaled distances
        # log(Delta/n^n) = 2 * (sum(log(d_scaled)) - n*log(n))
        # d_scaled = d * (2/D)
        # sum(log(d_scaled)) = sum(log(d)) + 28*log(2) - 28*log(D)
        log_score = 2.0 * (np.sum(np.log(d)) + 28.0 * np.log(2.0) - 28.0 * np.log(D) - 8.0 * np.log(8.0))
        return -log_score

    # Initial guesses for r1, r2, theta
    guesses = [
        [1.0, 0.0, np.pi/4], # Degenerate (one square shrunk)
        [0.5, 1.0, np.pi/4],
        [0.7, 1.0, np.pi/4],
        [0.8, 0.8, np.pi/4], # Two identical squares rotated
        [1.0, 1.0, 0.0],     # Overlapping
        [0.6, 1.2, np.pi/8],
        [0.9, 0.9, np.pi/4],
        [1.0, 0.5, np.pi/4],
        [1.0, 0.7, np.pi/4],
        [0.8, 1.0, np.pi/8],
    ]
    
    for guess in guesses:
        if time.time() > deadline - 20.0:
            break
        try:
            res = minimize(
                objective_two_squares,
                guess,
                method='L-BFGS-B',
                bounds=[(0.01, 1.5), (0.01, 1.5), (0.0, np.pi/2)],
                options={'maxiter': 300, 'ftol': 1e-12}
            )
            if res.success or res.fun < 1e9:
                r1, r2, theta = res.x
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
    for _ in range(10):
        r = np.random.rand(n) * 1.0
        theta = np.random.rand(n) * 2 * np.pi
        pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        inits.append(pts)
        
    # Perturbed Regular Octagon
    for _ in range(5):
        angles = angles_reg + 0.1 * np.random.randn(n)
        radii = 1.0 + 0.1 * np.random.randn(n)
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        inits.append(pts)

    # Two squares random init
    for _ in range(5):
        r1 = 0.5 + np.random.rand() * 0.5
        r2 = 0.5 + np.random.rand() * 0.5
        theta = np.random.rand() * np.pi/2
        angles1 = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
        angles2 = angles1 + theta
        pts1 = np.column_stack([r1 * np.cos(angles1), r1 * np.sin(angles1)])
        pts2 = np.column_stack([r2 * np.cos(angles2), r2 * np.sin(angles2)])
        inits.append(np.vstack([pts1, pts2]))

    bounds_general = [(-1.5, 1.5)] * (2 * n)

    for init in inits:
        if time.time() > deadline - 5.0:
            break
        try:
            x0 = init.flatten()
            res = minimize(
                objective_general,
                x0,
                method='L-BFGS-B',
                bounds=bounds_general,
                options={'maxiter': 200, 'ftol': 1e-12}
            )
            if np.all(np.isfinite(res.x)):
                refined_pts = res.x.reshape(n, 2)
                try_candidate(refined_pts)
        except Exception:
            pass

    # 4. Differential Evolution for Global Search (if time permits)
    # DE explores the 16D coordinate space.
    # We define a fitness function that returns the negative score (since DE minimizes).
    
    def de_objective(x):
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
        log_sum = np.sum(np.log(d))
        penalty = 0.0
        if D > 2.0:
            penalty = 100.0 * (D - 2.0)**2
        return -log_sum + penalty

    if time.time() < deadline - 10.0:
        try:
            # Run DE with a moderate number of iterations
            res_de = differential_evolution(
                de_objective,
                bounds_general,
                maxiter=50,
                popsize=15,
                tol=1e-8,
                seed=42,
                polish=True
            )
            if res_de.success or res_de.fun < 1e9:
                de_pts = res_de.x.reshape(n, 2)
                try_candidate(de_pts)
        except Exception:
            pass

    # 5. Final Refinement of the Best Candidate Found
    if best_pts is not None:
        try:
            res = minimize(
                objective_general,
                best_pts.flatten(),
                method='L-BFGS-B',
                bounds=bounds_general,
                options={'maxiter': 500, 'ftol': 1e-14}
            )
            if np.all(np.isfinite(res.x)):
                refined_pts = res.x.reshape(n, 2)
                try_candidate(refined_pts)
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
