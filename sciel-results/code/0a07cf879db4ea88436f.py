import numpy as np
from scipy.optimize import minimize
import time
import math

def construct():
    n = 8
    # Regular octagon as fallback
    def regular_octagon():
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return np.column_stack([np.cos(angles), np.sin(angles)])
    
    def score(points):
        # points: (n, 2)
        # Scale to diameter 2
        diffs = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        D = np.max(dists)
        if D == 0:
            return 0.0
        scaled_dists = dists * (2.0 / D)
        # Product of distances for i < j
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        prod = np.prod(scaled_dists[mask])
        return prod / (n ** n)

    start_time = time.time()
    timeout = 135  # seconds, keep 15s margin for safety
    
    best_score = score(regular_octagon())
    best_points = regular_octagon()
    
    # Parametric optimization:
    # Fix p0 = (-1, 0), p1 = (1, 0) as the diameter pair.
    # Optimize p2..p7.
    # Constraints: d(pi, pj) <= 2 for all i,j.
    
    def objective(x):
        # x: (12,) representing p2..p7
        pts = np.zeros((n, 2))
        pts[0] = [-1.0, 0.0]
        pts[1] = [1.0, 0.0]
        pts[2:] = x.reshape(n-2, 2)
        
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        
        # Check validity (distinct points)
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        if np.min(dists[mask]) < 1e-6:
            return 1e10
            
        # Log sum of distances
        log_sum = np.sum(np.log(dists[mask]))
        return -log_sum

    # Constraints for SLSQP: g(x) >= 0
    # We want d(pi, pj) <= 2, so 2 - d(pi, pj) >= 0
    def make_constraint(i, j):
        def constraint(x):
            pts = np.zeros((n, 2))
            pts[0] = [-1.0, 0.0]
            pts[1] = [1.0, 0.0]
            pts[2:] = x.reshape(n-2, 2)
            d = np.linalg.norm(pts[i] - pts[j])
            return 2.0 - d
        return constraint

    constraints = []
    # Pairs among 2..7
    for i in range(2, n):
        for j in range(i+1, n):
            constraints.append({'type': 'ineq', 'fun': make_constraint(i, j)})
    # Pairs between 0 and 2..7
    for i in range(2, n):
        constraints.append({'type': 'ineq', 'fun': make_constraint(0, i)})
    # Pairs between 1 and 2..7
    for i in range(2, n):
        constraints.append({'type': 'ineq', 'fun': make_constraint(1, i)})
    
    # Bounds for x (coordinates of p2..p7)
    # x in [-1, 1], y in [-1.8, 1.8]
    # This is tighter than [-3, 3] and safe for the lens.
    bounds = [(-1.0, 1.0), (-1.8, 1.8)] * (n-2)
    
    # Initial guesses
    # 1. Regular octagon perturbed to match p0, p1
    oct_pts = regular_octagon()
    # Indices: 0:(1,0), 1:(.707,.707), 2:(0,1), 3:(-.707,.707), 4:(-1,0), 5:(-.707,-.707), 6:(0,-1), 7:(.707,-.707)
    # p0=(-1,0) is idx 4. p1=(1,0) is idx 0.
    # Remaining: 1,2,3,5,6,7
    init_x = np.array([
        oct_pts[1][0], oct_pts[1][1],
        oct_pts[2][0], oct_pts[2][1],
        oct_pts[3][0], oct_pts[3][1],
        oct_pts[5][0], oct_pts[5][1],
        oct_pts[6][0], oct_pts[6][1],
        oct_pts[7][0], oct_pts[7][1]
    ])
    
    def run_slsqp(x0, maxiter, ftol):
        try:
            res = minimize(
                objective, 
                x0, 
                method='SLSQP', 
                bounds=bounds, 
                constraints=constraints,
                options={'maxiter': maxiter, 'ftol': ftol}
            )
            if res.success:
                pts = np.zeros((n, 2))
                pts[0] = [-1.0, 0.0]
                pts[1] = [1.0, 0.0]
                pts[2:] = res.x.reshape(n-2, 2)
                
                diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
                dists = np.linalg.norm(diffs, axis=2)
                D = np.max(dists)
                
                if D <= 2.0 + 1e-5:
                    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
                    if np.min(dists[mask]) > 1e-5:
                        return pts
            return None
        except Exception:
            return None

    def run_nelder_mead(x0, maxiter, tol):
        """Nelder-Mead on the exact (non-smooth) objective with penalty for constraint violations."""
        def penalized_obj(x):
            pts = np.zeros((n, 2))
            pts[0] = [-1.0, 0.0]
            pts[1] = [1.0, 0.0]
            pts[2:] = x.reshape(n-2, 2)
            
            diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dists = np.linalg.norm(diffs, axis=2)
            mask = np.triu(np.ones((n, n), dtype=bool), k=1)
            
            # Penalty for constraint violations
            max_dist = np.max(dists[mask])
            if max_dist > 2.0:
                penalty = 1e6 * (max_dist - 2.0) ** 2
            else:
                penalty = 0.0
            
            # Penalty for too-close points
            min_dist = np.min(dists[mask])
            if min_dist < 1e-5:
                penalty += 1e6 * (1e-5 - min_dist) ** 2
            
            log_sum = np.sum(np.log(np.maximum(dists[mask], 1e-10)))
            return -log_sum + penalty
        
        try:
            res = minimize(
                penalized_obj,
                x0,
                method='Nelder-Mead',
                options={'maxiter': maxiter, 'xatol': tol, 'fatol': tol, 'adaptive': True}
            )
            pts = np.zeros((n, 2))
            pts[0] = [-1.0, 0.0]
            pts[1] = [1.0, 0.0]
            pts[2:] = res.x.reshape(n-2, 2)
            
            diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dists = np.linalg.norm(diffs, axis=2)
            D = np.max(dists)
            
            if D <= 2.0 + 1e-4:
                mask = np.triu(np.ones((n, n), dtype=bool), k=1)
                if np.min(dists[mask]) > 1e-5:
                    return pts
            return None
        except Exception:
            return None

    # Run 1: From Octagon with SLSQP
    if time.time() - start_time < timeout:
        pts = run_slsqp(init_x, 500, 1e-10)
        if pts is not None:
            cand_score = score(pts)
            if cand_score > best_score:
                best_score = cand_score
                best_points = pts

    # Run 2: Nelder-Mead polish from SLSQP result
    if time.time() - start_time < timeout:
        x0 = best_points[2:].flatten()
        pts = run_nelder_mead(x0, 5000, 1e-10)
        if pts is not None:
            cand_score = score(pts)
            if cand_score > best_score:
                best_score = cand_score
                best_points = pts

    # Run 3..N: Random restarts with SLSQP then Nelder-Mead
    if time.time() - start_time < timeout:
        np.random.seed(42)
        for k in range(12):
            if time.time() - start_time >= timeout - 25:
                break
            pts_init = np.zeros((n, 2))
            pts_init[0] = [-1.0, 0.0]
            pts_init[1] = [1.0, 0.0]
            for i in range(2, n):
                # Sample x in [-1, 1]
                x = np.random.uniform(-1.0, 1.0)
                # Max y^2 is min(4-(x-1)^2, 4-(x+1)^2)
                max_y_sq = 3.0 - x*x - 2.0*abs(x)
                if max_y_sq < 0:
                    max_y_sq = 0.0
                max_y = np.sqrt(max_y_sq)
                y = np.random.uniform(-max_y, max_y)
                pts_init[i] = [x, y]
            
            x0 = pts_init[2:].flatten()
            # First SLSQP
            pts = run_slsqp(x0, 300, 1e-10)
            if pts is not None:
                cand_score = score(pts)
                if cand_score > best_score:
                    best_score = cand_score
                    best_points = pts
                    x0 = pts[2:].flatten()
            
            # Then Nelder-Mead polish
            if time.time() - start_time < timeout - 10:
                pts = run_nelder_mead(x0, 3000, 1e-10)
                if pts is not None:
                    cand_score = score(pts)
                    if cand_score > best_score:
                        best_score = cand_score
                        best_points = pts

    # Final polish: Nelder-Mead with high precision on best found
    if time.time() - start_time < timeout - 10:
        x_best = best_points[2:].flatten()
        pts = run_nelder_mead(x_best, 10000, 1e-12)
        if pts is not None:
            cand_score = score(pts)
            if cand_score > best_score:
                best_score = cand_score
                best_points = pts

    # Coordinate descent polish: iterate over each coordinate with fine 1D search
    if time.time() - start_time < timeout - 5:
        pts_cd = best_points.copy()
        n_iters_cd = 3
        for it in range(n_iters_cd):
            if time.time() - start_time >= timeout - 3:
                break
            improved = False
            for i in range(n):
                for dim in range(2):
                    # Try small perturbations around current value
                    base = pts_cd[i, dim]
                    best_val = pts_cd[i, dim]
                    best_sc = score(pts_cd)
                    # Geometrically shrinking step sizes
                    for step in [0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001]:
                        for direction in [-1, 1]:
                            candidate = pts_cd.copy()
                            candidate[i, dim] = base + direction * step
                            # Check validity
                            diffs = candidate[:, np.newaxis, :] - candidate[np.newaxis, :, :]
                            dists = np.linalg.norm(diffs, axis=2)
                            D = np.max(dists)
                            mask = np.triu(np.ones((n, n), dtype=bool), k=1)
                            if D > 2.0 + 1e-8 or np.min(dists[mask]) < 1e-8:
                                continue
                            sc = score(candidate)
                            if sc > best_sc:
                                best_sc = sc
                                best_val = candidate[i, dim]
                        # Also try bisection-like refinement around best found so far
                    if best_val != pts_cd[i, dim]:
                        pts_cd[i, dim] = best_val
                        improved = True
            # If no improvement in a full sweep, try smaller steps
            if not improved:
                break
        cand_score = score(pts_cd)
        if cand_score > best_score:
            best_score = cand_score
            best_points = pts_cd

    # One more NM polish if time allows
    if time.time() - start_time < timeout - 2:
        x_best = best_points[2:].flatten()
        pts = run_nelder_mead(x_best, 5000, 1e-13)
        if pts is not None:
            cand_score = score(pts)
            if cand_score > best_score:
                best_score = cand_score
                best_points = pts

    return best_points.tolist()
