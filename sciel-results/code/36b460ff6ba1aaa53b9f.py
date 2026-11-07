"""Optimizer for BBOB-style f11 (Discus) using robust Bayesian Optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
import time
import sys

# Try to import sklearn for GP, but fall back to a simpler approach if not available or too slow
try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, RBF, ConstantKernel
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

class Optimizer:
    """Robust Bayesian Optimizer for BBOB f11."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: 300s limit, we stop at 280s to be safe
        deadline = start_time + 280
        
        dim = self.dim
        budget = self.budget
        bounds = [(-5.0, 5.0)] * dim
        bounds_np = (np.full(dim, -5.0), np.full(dim, 5.0))
        
        # Track function calls
        call_count = 0
        call_limit = budget
        
        def wrapped_func(x):
            nonlocal call_count
            if call_count >= call_limit:
                # Should not happen if we check before calling, but safety net
                return np.inf
            call_count += 1
            try:
                val = func(x)
                if not np.isfinite(val):
                    return np.inf
                return val
            except Exception:
                return np.inf

        best_x = None
        best_f = np.inf
        
        # Store observations
        X_obs = []
        y_obs = []
        
        # --- Phase 1: Initial Sampling ---
        # Use Sobol sequence for better coverage
        # scipy.stats.qmc.Sobol is available in recent SciPy versions
        try:
            from scipy.stats.qmc import Sobol
            n_initial = min(30, budget // 5)
            sampler = Sobol(d=dim, scramble=True, seed=42)
            # Sobol requires power of 2 for optimal properties, but works for others
            # We'll just sample n_initial points
            points = sampler.random(n_initial)
            # Scale from [0,1] to [-5,5]
            points = points * 10.0 - 5.0
            
            # Always include the center
            center = np.zeros(dim)
            points = np.vstack([center, points])
            
        except Exception:
            # Fallback to random if Sobol fails
            n_initial = min(20, budget // 5)
            rng = np.random.default_rng(42)
            points = rng.uniform(-5.0, 5.0, size=(n_initial, dim))
            center = np.zeros(dim)
            points = np.vstack([center, points])

        for x0 in points:
            if call_count >= budget - 50 or time.time() > deadline:
                break
            fx0 = wrapped_func(x0)
            X_obs.append(x0.copy())
            y_obs.append(fx0)
            if fx0 < best_f:
                best_f = fx0
                best_x = x0.copy()
        
        if len(X_obs) < 2:
             # If we failed to get enough initial points, just return what we have
             return best_x, best_f

        X_obs = np.array(X_obs)
        y_obs = np.array(y_obs)
        
        # --- Phase 2: Bayesian Optimization Loop ---
        # We reserve some budget for polishing
        reserved_for_polish = 50
        max_bo_calls = budget - call_count - reserved_for_polish
        
        rng = np.random.default_rng(123)
        
        for it in range(max_bo_calls):
            if call_count >= budget - reserved_for_polish:
                break
            if time.time() > deadline:
                break
                
            # Fit GP Model
            if HAS_SKLEARN:
                try:
                    # Normalize y for GP stability
                    y_mean = np.mean(y_obs)
                    y_std = np.std(y_obs) + 1e-8
                    y_norm = (y_obs - y_mean) / y_std
                    
                    # Use Matern kernel with nu=2.5, which is good for BBOB
                    # Length scales are optimized
                    kernel = ConstantKernel(1.0, (1e-2, 1e2)) * Matern(
                        length_scale=1.0, 
                        length_scale_bounds=(0.1, 10.0), 
                        nu=2.5
                    )
                    
                    gp = GaussianProcessRegressor(
                        kernel=kernel, 
                        alpha=1e-6, 
                        n_restarts_optimizer=2,
                        normalize_y=True
                    )
                    gp.fit(X_obs, y_obs)
                    
                    # Define EI acquisition
                    def ei(X):
                        X = np.atleast_2d(X)
                        y_pred, y_std_pred = gp.predict(X, return_std=True)
                        # y_pred is normalized by GP if normalize_y=True
                        # But we need to compare with best observed in original scale?
                        # Actually, EI is scale-invariant if we use the GP's internal scale.
                        # Let's stick to the GP's normalized scale for EI calculation
                        # and just pick the point that maximizes EI in that space.
                        
                        best_y_norm = np.min((y_obs - y_mean) / y_std)
                        
                        # Avoid division by zero
                        y_std_pred = np.maximum(y_std_pred, 1e-8)
                        
                        improvement = best_y_norm - y_pred
                        Z = improvement / y_std_pred
                        ei_val = improvement * norm.cdf(Z) + y_std_pred * norm.pdf(Z)
                        return -np.mean(ei_val) # Minimize negative EI

                    # Optimize EI with multiple restarts
                    n_restarts = 10
                    best_acq_x = None
                    best_acq_val = np.inf
                    
                    # Start from best point and random points
                    start_points = [best_x]
                    for _ in range(n_restarts - 1):
                        start_points.append(rng.uniform(-4.0, 4.0, size=dim))
                        
                    for x0 in start_points:
                        try:
                            res = minimize(ei, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 50, 'ftol': 1e-6})
                            if res.fun < best_acq_val:
                                best_acq_val = res.fun
                                best_acq_x = res.x
                        except Exception:
                            continue
                    
                    if best_acq_x is None:
                        best_acq_x = rng.uniform(-4.0, 4.0, size=dim)
                        
                except Exception:
                    # If GP fails, fallback to random search
                    best_acq_x = rng.uniform(-4.0, 4.0, size=dim)
            else:
                # Fallback: Random search
                best_acq_x = rng.uniform(-4.0, 4.0, size=dim)

            # Evaluate new point
            fx_new = wrapped_func(best_acq_x)
            X_obs = np.vstack([X_obs, [best_acq_x]])
            y_obs = np.append(y_obs, fx_new)
            
            if fx_new < best_f:
                best_f = fx_new
                best_x = best_acq_x.copy()

        # --- Phase 3: Polishing ---
        # Use local optimizer to refine the best point found
        if best_x is not None and call_count < budget - 5 and time.time() < deadline:
            try:
                # Powell is robust for non-smooth functions
                res = minimize(
                    wrapped_func, 
                    best_x, 
                    method='Powell', 
                    bounds=bounds, 
                    options={'maxiter': 50, 'maxfev': 100}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass
            
            # Try L-BFGS-B as well if budget allows
            if call_count < budget - 5 and time.time() < deadline:
                try:
                    res = minimize(
                        wrapped_func, 
                        best_x, 
                        method='L-BFGS-B', 
                        bounds=bounds, 
                        options={'maxiter': 50, 'maxls': 20}
                    )
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x.copy()
                except Exception:
                    pass

        return best_x, best_f
# EVOLVE-BLOCK-END
