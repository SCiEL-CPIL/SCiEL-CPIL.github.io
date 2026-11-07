"""Sparse Grid Interpolation + Adaptive Refinement optimizer for BBOB f22."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import qmc
from scipy.interpolate import RBFInterpolator

class Optimizer:
    """Sparse grid + adaptive refinement optimizer for BBOB f22."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 310.0  # Safety margin before 330s kill
        dim = self.dim
        budget = self.budget

        eval_count = 0
        best_x = np.zeros(dim)
        best_f = float('inf')

        # Fixed seed for reproducibility
        rng = np.random.default_rng(42)

        def wrapped_func(x):
            nonlocal eval_count, best_x, best_f
            if eval_count >= budget:
                return float('inf')
            eval_count += 1
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            f_val = float(func(x))
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        bounds = [(-5.0, 5.0)] * dim

        # --- Phase 1: Sparse Grid Initial Exploration ---
        # Use a structured sparse grid (full factorial at low resolution)
        # For 5D, use 4 levels per dimension = 4^5 = 1024 points (too many)
        # Use 3 levels per dimension = 3^5 = 243 points, or mix with Sobol
        
        # Strategy: Use Sobol for 64 points, then build RBF surrogate
        n_initial = 64
        try:
            sampler = qmc.Sobol(d=dim, scramble=True, seed=42)
            initial_points = sampler.random(n_initial) * 10.0 - 5.0
        except Exception:
            initial_points = rng.uniform(-5.0, 5.0, (n_initial, dim))

        # Evaluate initial points
        X_eval = []
        f_eval = []
        for x in initial_points:
            if eval_count >= budget or time.time() > deadline:
                break
            f_val = wrapped_func(x)
            X_eval.append(x.copy())
            f_eval.append(f_val)

        if not X_eval:
            return best_x, best_f

        X_eval = np.array(X_eval)
        f_eval = np.array(f_eval)

        # --- Phase 2: Adaptive Refinement with Surrogate Model ---
        # Build RBF surrogate and iteratively select points to evaluate
        # based on uncertainty (prediction variance proxy) or predicted minimum
        
        max_adaptive_iterations = 200
        adaptive_evals = 0
        
        for iteration in range(max_adaptive_iterations):
            if eval_count >= budget or time.time() > deadline:
                break
            
            # Build or update RBF surrogate
            try:
                # Use thin-plate spline or multiquadric
                interpolator = RBFInterpolator(
                    X_eval, f_eval, 
                    kernel='thin_plate_spline', 
                    smoothing=0.01
                )
            except Exception:
                # Fallback: use inverse distance weighting
                interpolator = None

            # Generate candidate points for next evaluation
            # Mix of:
            # 1. Random points
            # 2. Points near current best (exploit)
            # 3. Points where surrogate predicts low value (if available)
            
            candidates = []
            
            # Exploit: perturb current best
            for sigma in [0.3, 0.5, 1.0]:
                x_cand = best_x + rng.normal(0, sigma, dim)
                x_cand = np.clip(x_cand, -5.0, 5.0)
                candidates.append(x_cand)
            
            # Explore: random points
            for _ in range(3):
                x_cand = rng.uniform(-5.0, 5.0, dim)
                candidates.append(x_cand)
            
            # If we have a surrogate, find where it predicts low values
            if interpolator is not None:
                # Sample random points and evaluate surrogate
                n_surrogate_samples = 20
                surrogate_samples = rng.uniform(-5.0, 5.0, (n_surrogate_samples, dim))
                try:
                    surrogate_vals = interpolator(surrogate_samples)
                    # Pick the 2 lowest predicted values
                    idx_sorted = np.argsort(surrogate_vals)
                    for idx in idx_sorted[:2]:
                        candidates.append(surrogate_samples[idx])
                except Exception:
                    pass
            
            # Evaluate all candidates, pick the best one
            best_candidate = None
            best_candidate_f = float('inf')
            
            for x_cand in candidates:
                if eval_count >= budget or time.time() > deadline:
                    break
                x_cand = np.clip(x_cand, -5.0, 5.0)
                f_cand = wrapped_func(x_cand)
                if f_cand < best_candidate_f:
                    best_candidate_f = f_cand
                    best_candidate = x_cand.copy()
            
            if best_candidate is not None:
                X_eval.append(best_candidate)
                f_eval.append(best_candidate_f)
                X_eval = np.array(X_eval)
                f_eval = np.array(f_eval)
            
            # If we found a new local minimum, run L-BFGS-B to refine
            if best_candidate_f < best_f - 1e-10 and best_candidate is not None:
                try:
                    res = minimize(
                        wrapped_func,
                        best_candidate,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 30}
                    )
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x.copy()
                        # Add the refined point to our data
                        X_eval.append(res.x)
                        f_eval.append(res.fun)
                        X_eval = np.array(X_eval)
                        f_eval = np.array(f_eval)
                except Exception:
                    pass

        # --- Phase 3: Final Polish with L-BFGS-B ---
        if eval_count < budget and time.time() < deadline:
            try:
                res = minimize(
                    wrapped_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-15, 'gtol': 1e-10}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                pass

        return best_x, best_f

# EVOLVE-BLOCK-END
