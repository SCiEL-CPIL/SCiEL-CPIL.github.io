import numpy as np
from scipy.optimize import minimize, basinhopping
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    n = 8
    
    def compute_log_score_smooth(points, alpha=100.0):
        """
        Computes a smooth approximation of the log score.
        Uses log-sum-exp to approximate the log of the max distance (diameter).
        This makes the objective differentiable everywhere, allowing gradient-based
        optimizers to work effectively.
        
        Score = Delta / n^n
        Delta = prod_{i != j} d'_ij
        d'_ij = d_ij * (2 / D)
        D = max_{i != j} d_ij
        
        log(Score) = sum_{i != j} log(d_ij) + n(n-1)(log(2) - log(D)) - n log(n)
        
        Approximation:
        log(D) ≈ (1/alpha) * logsumexp(alpha * d_ij)
        """
        # points: (n, 2) array
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        
        # Mask diagonal
        mask = ~np.eye(n, dtype=bool)
        valid_dists_sq = dists_sq[mask]
        
        # Check for distinct points (avoid log(0))
        if np.any(valid_dists_sq < 1e-12):
            return -1e10
            
        valid_dists = np.sqrt(valid_dists_sq)
        
        # log(D) approximation
        # We want log(max(d)). 
        # log(max(d)) = log(exp(log(max(d)))) = log(max(exp(log(d))))
        # max(exp(x)) ≈ (1/alpha) * logsumexp(alpha * x)
        # Here x = log(d). But logsumexp operates on values, so:
        # log(max(d)) ≈ (1/alpha) * logsumexp(alpha * log(d))
        # This is equivalent to (1/alpha) * log( sum(d^alpha) )
        
        # Using log(d) directly in logsumexp can be numerically tricky if d varies widely,
        # but for alpha=100, it's a good approximation of the max.
        # Let's use the property: log(max(d)) ≈ (1/alpha) * logsumexp(alpha * log(d))
        
        log_dists = np.log(valid_dists)
        
        # Stable logsumexp: max(log_dists) + log(sum(exp(log_dists - max(log_dists))))
        max_log_d = np.max(log_dists)
        logsumexp_val = max_log_d + np.log(np.sum(np.exp(log_dists - max_log_d)))
        
        # log(D) ≈ (1/alpha) * logsumexp(alpha * log_dists)
        # Wait, logsumexp(alpha * x) = (1/alpha) * logsumexp(x) is NOT correct.
        # logsumexp(alpha * x) = log(sum(exp(alpha * x)))
        # We want log(max(x)).
        # max(x) = (1/alpha) * logsumexp(alpha * x) is incorrect.
        # max(x) ≈ (1/alpha) * log( sum(exp(alpha * x)) ) = (1/alpha) * logsumexp(alpha * x)
        
        # So: log(D) ≈ (1/alpha) * logsumexp(alpha * log(dists))
        
        scaled_log_dists = alpha * log_dists
        max_scaled = np.max(scaled_log_dists)
        logsumexp_scaled = max_scaled + np.log(np.sum(np.exp(scaled_log_dists - max_scaled)))
        
        log_D_approx = (1.0 / alpha) * logsumexp_scaled
        
        # log(Delta) = sum(log(d'_ij)) = sum(log(d_ij)) + n(n-1)*(log(2) - log(D))
        log_prod_unscaled = np.sum(log_dists)
        log_scaled_prod = log_prod_unscaled + n * (n - 1) * (np.log(2.0) - log_D_approx)
        
        # log(Score) = log_scaled_prod - n * log(n)
        return log_scaled_prod - n * np.log(n)

    def objective(x):
        """
        Objective function for minimization (we maximize log_score, so minimize -log_score).
        x is a flat array of 16 coordinates.
        """
        pts = x.reshape((n, 2))
        
        # Basic validity check
        diff = pts[:, np.newaxis, :] - points[np.newaxis, :, :] if False else pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        mask = ~np.eye(n, dtype=bool)
        if np.any(dists_sq[mask] < 1e-10):
            return 1e10
            
        s = compute_log_score_smooth(pts)
        if not np.isfinite(s):
            return 1e10
            
        return -s

    def step(x, stepsize):
        """
        Perturbation function for Basin Hopping.
        """
        x = np.array(x)
        x += stepsize * np.random.normal(0, 1, size=x.shape)
        return x

    # Initial guesses
    # 1. Regular Octagon
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    pts_oct = np.column_stack([np.cos(angles), np.sin(angles)])
    x_oct = pts_oct.flatten()
    
    best_score = -np.inf
    best_x = x_oct
    
    # Evaluate regular octagon
    s_oct = compute_log_score_smooth(pts_oct)
    best_score = s_oct
    best_x = x_oct.copy()
    
    # Generate diverse initial guesses
    structured_starts = []
    rng = np.random.default_rng(42)
    
    # Variation 1: Two concentric squares (D4 symmetry)
    for r2 in np.linspace(0.5, 1.5, 5):
        a1 = np.linspace(0, 2*np.pi, 4, endpoint=False)
        a2 = np.linspace(np.pi/4, 2*np.pi + np.pi/4, 4, endpoint=False)
        pts1 = r2 * np.column_stack([np.cos(a1), np.sin(a1)])
        pts2 = 1.0 * np.column_stack([np.cos(a2), np.sin(a2)])
        pts = np.vstack([pts1, pts2])
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        D = np.max(dists[~np.eye(n, dtype=bool)])
        if D > 0:
            pts = pts * (2.0 / D)
        structured_starts.append(pts.flatten())
        
    # Variation 2: Random Jittered Octagons
    for _ in range(20):
        jitter = rng.normal(0, 0.3, size=(n, 2))
        pts = pts_oct + jitter
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        D = np.max(dists[~np.eye(n, dtype=bool)])
        if D > 0:
            pts = pts * (2.0 / D)
        structured_starts.append(pts.flatten())
        
    # Evaluate structured starts
    for x0 in structured_starts:
        if time.time() > deadline:
            break
        s = compute_log_score_smooth(x0.reshape((n, 2)))
        if s > best_score:
            best_score = s
            best_x = x0.copy()

    # Optimization Phase 1: Basin Hopping with Smooth Objective
    # The smooth objective allows L-BFGS-B to converge reliably to local maxima.
    # Basin Hopping then jumps between these basins.
    
    candidates = [best_x]
    if len(structured_starts) > 0:
        candidates.append(structured_starts[0])
    if len(structured_starts) > 10:
        candidates.append(structured_starts[10])
        
    # Deduplicate
    unique_candidates = []
    for c in candidates:
        is_dup = False
        for uc in unique_candidates:
            if np.linalg.norm(c - uc) < 1e-3:
                is_dup = True
                break
        if not is_dup:
            unique_candidates.append(c)
            
    for x0 in unique_candidates:
        if time.time() > deadline - 20:
            break
            
        try:
            bh_res = basinhopping(
                objective, 
                x0, 
                step=step, 
                stepsize=0.5, 
                minimizer_kwargs={'method': 'L-BFGS-B', 'bounds': [(-2.0, 2.0)] * (2 * n)},
                niter=30, 
                T=100.0, 
                accept_test=None,
                seed=42
            )
            
            final_score = -bh_res.fun
            if final_score > best_score:
                best_score = final_score
                best_x = bh_res.x.copy()
                
        except Exception:
            pass

    # Optimization Phase 2: Nelder-Mead Polish
    # The smooth objective has a minimum at the same location as the non-smooth one,
    # but the landscape near the maximum of the non-smooth function might have flat
    # regions or kinks that L-BFGS-B handles poorly if the approximation isn't perfect
    # or if we want to refine the exact non-smooth objective.
    # The cell findings suggest Nelder-Mead on the EXACT objective helps escape flat regions.
    # We will run Nelder-Mead on the EXACT (non-smooth) objective using the best point
    # found by the smooth optimization as a starting point.
    
    def compute_log_score_exact(points):
        """
        Computes the exact log score (non-smooth).
        """
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        mask = ~np.eye(n, dtype=bool)
        valid_dists = dists[mask]
        
        if np.any(valid_dists < 1e-10):
            return -np.inf
            
        D = np.max(valid_dists)
        
        if D < 1e-10:
            return -np.inf
            
        log_prod_unscaled = np.sum(np.log(valid_dists))
        log_scaled_prod = log_prod_unscaled + n * (n - 1) * (np.log(2.0) - np.log(D))
        
        return log_scaled_prod - n * np.log(n)

    def objective_exact(x):
        pts = x.reshape((n, 2))
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        mask = ~np.eye(n, dtype=bool)
        if np.any(dists_sq[mask] < 1e-10):
            return 1e10
        s = compute_log_score_exact(pts)
        if not np.isfinite(s):
            return 1e10
        return -s

    # Run Nelder-Mead on the exact objective
    if time.time() < deadline - 5:
        try:
            nm_res = minimize(
                objective_exact, 
                best_x, 
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            
            final_score = -nm_res.fun
            if final_score > best_score:
                best_score = final_score
                best_x = nm_res.x.copy()
                
        except Exception:
            pass

    # Final L-BFGS-B polish on exact objective if time permits
    if time.time() < deadline - 2:
        try:
            res = minimize(
                objective_exact, 
                best_x, 
                method='L-BFGS-B', 
                bounds=[(-2.0, 2.0)] * (2 * n),
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res.success:
                final_score = -res.fun
                if final_score > best_score:
                    best_score = final_score
                    best_x = res.x.copy()
        except Exception:
            pass

    # Final check: Ensure points are distinct and within bounds
    pts = best_x.reshape((n, 2))
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    mask = ~np.eye(n, dtype=bool)
    valid_dists = dists[mask]
    
    if np.any(valid_dists < 1e-5):
        # Fallback to regular octagon if points collapsed
        pts = pts_oct
        
    # Ensure coordinates are within |coord| < 10^6 (they should be, but just in case)
    # The optimization keeps them in [-2, 2], so this is safe.
    
    return [[float(x), float(y)] for x, y in pts]
