"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from math import cos, sin, pi

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    n = 8
    
    # Regular octagon as initial best
    octagon = np.array([[cos(2 * pi * k / 8), sin(2 * pi * k / 8)] for k in range(8)], dtype=float)
    best_points = octagon.copy()
    
    def _raw_score(points):
        """Compute Delta / n^n for given points."""
        pts = np.array(points, dtype=float)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        sq_dists = np.sum(diff**2, axis=2)
        dists = np.sqrt(np.maximum(sq_dists, 1e-30))
        diam = np.max(dists)
        if diam < 1e-15:
            return -1e30
        scale = 2.0 / diam
        scaled_dists = dists * scale
        mask = ~np.eye(n, dtype=bool)
        log_delta = np.sum(np.log(scaled_dists[mask]))
        log_score = log_delta - n * np.log(n)
        return log_score
    
    best_log_score = _raw_score(best_points)
    
    def _smooth_objective(x):
        """
        Smooth objective for basin hopping.
        Uses log-sum-exp to smoothly approximate the max distance (diameter).
        Minimize negative log score.
        """
        pts = x.reshape(n, 2)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        sq_dists = np.sum(diff**2, axis=2)
        
        # Log-sum-exp for smooth max of squared distances
        # We want smooth approximation of max(dists) = max(sqrt(sq_dists))
        # Use log-sum-exp on sq_dists with a temperature
        temp = 50.0  # temperature for soft-max
        sq_dists_clipped = np.clip(sq_dists, 1e-30, None)
        # Mask diagonal
        mask = ~np.eye(n, dtype=bool)
        sq_dists_masked = np.where(mask, sq_dists_clipped, -np.inf)
        
        # log-sum-exp: log(sum(exp(sq_dists/temp))) * temp
        max_sq = np.max(sq_dists_masked, axis=None)
        log_sum_exp = max_sq + temp * np.log(np.sum(np.exp((sq_dists_masked - max_sq) / temp)))
        smooth_sq_diam = log_sum_exp
        smooth_diam = np.sqrt(smooth_sq_diam)
        
        # Compute scaled distances
        dists = np.sqrt(sq_dists_clipped)
        scale = 2.0 / smooth_diam
        scaled_dists = dists * scale
        scaled_dists = np.where(mask, scaled_dists, 1.0)  # avoid log(0) on diagonal
        
        log_delta = np.sum(np.log(np.maximum(scaled_dists, 1e-30))[mask])
        log_score = log_delta - n * np.log(n)
        
        # Check for distinct points
        min_off_diag = np.min(dists[mask])
        if min_off_diag < 1e-10:
            return 1e6
        
        return -log_score
    
    rng = np.random.RandomState(12345)
    
    def _is_valid(candidate):
        """Check if candidate points are distinct."""
        diff = candidate[:, np.newaxis, :] - candidate[np.newaxis, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        return np.min(dists) > 1e-8
    
    # Basin hopping with multiple restarts
    n_restarts = 15
    bh_iterations_per_restart = 50
    step_size = 0.3
    
    for restart in range(n_restarts):
        if time.time() > deadline - 20.0:
            break
        
        # Generate starting point
        if restart == 0:
            x0 = octagon.flatten().copy()
        elif restart % 3 == 1:
            # Perturbed octagon
            x0 = octagon.flatten().copy()
            x0 += rng.normal(0, 0.2, size=16)
        else:
            # Random points in unit disk
            angles = rng.uniform(0, 2*pi, size=n)
            radii = rng.uniform(0.5, 1.0, size=n)
            x0 = np.column_stack([radii * cos(angles), radii * sin(angles)]).flatten()
        
        current_x = x0.copy()
        current_f = _smooth_objective(current_x)
        
        for it in range(bh_iterations_per_restart):
            if time.time() > deadline - 10.0:
                break
            
            # Random perturbation
            step = rng.normal(0, step_size, size=16)
            trial_x = current_x + step
            
            # Local optimization from trial point
            try:
                result = minimize(_smooth_objective, trial_x, method='L-BFGS-B',
                                options={'maxiter': 200, 'ftol': 1e-10})
                trial_x = result.x
                trial_f = _smooth_objective(trial_x)
            except Exception:
                continue
            
            # Accept if better
            if trial_f < current_f:
                current_x = trial_x
                current_f = trial_f
            
            # Check if we found a better global best
            candidate = current_x.reshape(n, 2)
            if _is_valid(candidate):
                cand_log_score = _raw_score(candidate)
                if cand_log_score > best_log_score:
                    best_log_score = cand_log_score
                    best_points = candidate.copy()
        
        # After basin hopping, do a final local optimization from the best point
        try:
            result = minimize(_smooth_objective, current_x, method='L-BFGS-B',
                            options={'maxiter': 1000, 'ftol': 1e-12})
            candidate = result.x.reshape(n, 2)
            if _is_valid(candidate):
                cand_log_score = _raw_score(candidate)
                if cand_log_score > best_log_score:
                    best_log_score = cand_log_score
                    best_points = candidate.copy()
        except Exception:
            pass
    
    # Final high-precision polish using L-BFGS-B with tight tolerance
    if time.time() < deadline - 5.0:
        try:
            x0_flat = best_points.flatten()
            result = minimize(_smooth_objective, x0_flat, method='L-BFGS-B',
                            options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-12,
                                    'maxfun': 20000})
            candidate = result.x.reshape(n, 2)
            if _is_valid(candidate):
                cand_log_score = _raw_score(candidate)
                if cand_log_score > best_log_score:
                    best_log_score = cand_log_score
                    best_points = candidate.copy()
        except Exception:
            pass
    
    # Final Nelder-Mead polish for any remaining improvements
    if time.time() < deadline - 2.0:
        try:
            x0_flat = best_points.flatten()
            result = minimize(_smooth_objective, x0_flat, method='Nelder-Mead',
                            options={'maxiter': 20000, 'xatol': 1e-14, 'fatol': 1e-16})
            candidate = result.x.reshape(n, 2)
            if _is_valid(candidate):
                cand_log_score = _raw_score(candidate)
                if cand_log_score > best_log_score:
                    best_log_score = cand_log_score
                    best_points = candidate.copy()
        except Exception:
            pass
    
    return [[float(p[0]), float(p[1])] for p in best_points]
# EVOLVE-BLOCK-END
