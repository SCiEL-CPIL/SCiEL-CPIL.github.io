"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    def score(pts):
        """Calculate the raw score Delta/n^n for a set of 8 points."""
        if len(pts) != 8:
            return 0.0
        pts = np.array(pts)
        
        # Calculate all pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        
        # Exclude self-distances (diagonal)
        idx = np.arange(8)
        dists[idx, idx] = 1.0
        
        # Diameter is the max distance
        d = np.max(dists)
        if d == 0:
            return 0.0
        
        # Scale to diameter 2
        scaled_dists = dists * (2.0 / d)
        
        # Product of distances
        # log-sum-exp trick is safer, but for n=8, 56 terms, product is usually fine.
        # However, if any distance is 0 (duplicate points), log(0) = -inf.
        # We assume distinct points from the parameterization or DE.
        
        # Check for very small distances to avoid log(0)
        min_d = np.min(scaled_dists[idx, idx] == 1.0) # This check is slightly off, let's just check min off-diagonal
        off_diag_mask = ~np.eye(8, dtype=bool)
        min_off_diag = np.min(scaled_dists[off_diag_mask])
        if min_off_diag < 1e-12:
            return 0.0

        log_prod = np.sum(np.log(scaled_dists))
        prod = np.exp(log_prod)
        
        # Score is Delta / n^n
        score_val = prod / (8.0 ** 8)
        
        return score_val

    def objective(x):
        """Objective function for minimization (negative score)."""
        pts = x.reshape(8, 2)
        s = score(pts)
        if not np.isfinite(s) or s == 0:
            return 1e10
        return -s

    # Regular octagon initialization
    reg_oct = np.array([[np.cos(2 * np.pi * k / 8), np.sin(2 * np.pi * k / 8)] for k in range(8)])
    best_score = score(reg_oct)
    best_points = reg_oct.copy()

    # 1. Global Search: Differential Evolution
    # We search in the full 16D space.
    # Bounds: coordinates in [-1, 1] is sufficient due to scale invariance, 
    # but we can use [-2, 2] to be safe. The score is scale invariant.
    bounds = [(-1.0, 1.0)] * 16
    
    # DE settings
    # popsize=15 for 16D is reasonable (15*16=240 evals per gen)
    # maxiter=30 to keep time low, then refine.
    # tol=1e-10
    
    de_result = None
    if time.time() < deadline - 60:
        try:
            de_result = differential_evolution(
                objective,
                bounds,
                seed=42,
                maxiter=20,
                tol=1e-8,
                popsize=15,
                init='sobol', # More uniform than 'latinhypercube' sometimes
                polish=False, # We will polish manually
                updating='deferred'
            )
            if de_result.fun < objective(best_points.flatten()):
                cand_pts = de_result.x.reshape(8, 2)
                cand_score = score(cand_pts)
                if cand_score > best_score:
                    best_score = cand_score
                    best_points = cand_pts
        except Exception:
            pass

    # 2. Local Refinement: Multi-stage
    # We refine the best point found by DE and also the regular octagon (in case DE failed or got stuck)
    
    start_candidates = []
    if de_result is not None and de_result.fun < 1e9:
        start_candidates.append(de_result.x)
    start_candidates.append(reg_oct.flatten())
    
    # Add some random perturbations of the best point to escape sharp local minima
    if time.time() < deadline - 10:
        for _ in range(3):
            noise = np.random.normal(0, 0.1, 16)
            start_candidates.append(best_points.flatten() + noise)

    for start in start_candidates:
        if time.time() > deadline - 5:
            break
        
        current = start
        current_score = -objective(current)
        
        # Stage 1: L-BFGS-B
        try:
            res1 = minimize(
                objective, 
                current, 
                method='L-BFGS-B', 
                options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if res1.fun < current_score:
                current = res1.x
                current_score = -res1.fun
        except Exception:
            pass

        # Stage 2: SLSQP (often finds slightly better solutions than L-BFGS-B for this problem)
        try:
            res2 = minimize(
                objective, 
                current, 
                method='SLSQP', 
                options={'maxiter': 100, 'ftol': 1e-15}
            )
            if res2.fun < current_score:
                current = res2.x
                current_score = -res2.fun
        except Exception:
            pass

        # Stage 3: Final L-BFGS-B polish
        try:
            res3 = minimize(
                objective, 
                current, 
                method='L-BFGS-B', 
                options={'maxiter': 100, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if res3.fun < current_score:
                current = res3.x
                current_score = -res3.fun
        except Exception:
            pass

        # Update best
        if current_score > best_score:
            best_score = current_score
            best_points = current.reshape(8, 2)

    # Final check
    if best_points is None:
        return reg_oct.tolist()

    return best_points.tolist()

# EVOLVE-BLOCK-END
