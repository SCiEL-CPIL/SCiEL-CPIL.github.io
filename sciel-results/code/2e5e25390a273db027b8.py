"""Optimizer for BBOB-style f11 (Discus) using Bayesian Optimization with GP surrogate."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

class Optimizer:
    """Bayesian Optimization with Gaussian Process surrogate for BBOB f11 (Discus)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        bounds = [(-5.0, 5.0)] * dim
        bounds_np = (np.full(dim, -5.0), np.full(dim, 5.0))
        
        # Track function calls
        call_tracker = [0]
        
        def wrapped_func(x):
            call_tracker[0] += 1
            return func(x)
        
        best_x = None
        best_f = np.inf
        
        # Store observations for GP
        X_obs = []
        y_obs = []
        
        # Phase 1: Initial random sampling to build initial GP model
        n_initial = min(20, budget // 10)
        rng = np.random.default_rng(42)
        
        # Include center and some structured points
        initial_points = [np.zeros(dim)]
        for _ in range(n_initial - 1):
            x = rng.uniform(-4, 4, size=dim)
            initial_points.append(x)
        
        for x0 in initial_points:
            if call_tracker[0] >= budget - 50:
                break
            if time.time() > deadline:
                break
            
            fx0 = wrapped_func(x0)
            X_obs.append(x0.copy())
            y_obs.append(fx0)
            if fx0 < best_f:
                best_f = fx0
                best_x = x0.copy()
        
        X_obs = np.array(X_obs)
        y_obs = np.array(y_obs)
        
        # Phase 2: Bayesian Optimization loop with GP surrogate
        # Use Expected Improvement acquisition function
        n_bo_iterations = min(50, budget - call_tracker[0] - 20)
        
        for it in range(n_bo_iterations):
            if call_tracker[0] >= budget - 20:
                break
            if time.time() > deadline:
                break
            
            # Fit GP model
            try:
                from sklearn.gaussian_process import GaussianProcessRegressor
                from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
            except ImportError:
                # Fallback to random sampling if sklearn not available
                x_new = rng.uniform(-4, 4, size=dim)
                fx_new = wrapped_func(x_new)
                X_obs = np.vstack([X_obs, [x_new]])
                y_obs = np.append(y_obs, fx_new)
                if fx_new < best_f:
                    best_f = fx_new
                    best_x = x_new.copy()
                continue
            
            # Normalize y for better GP performance
            y_mean = np.mean(y_obs)
            y_std = np.std(y_obs) + 1e-8
            y_norm = (y_obs - y_mean) / y_std
            
            # Define kernel
            length_scale = 1.0
            kernel = ConstantKernel(1.0) * RBF(length_scale=length_scale, length_scale_bounds=(0.1, 10.0))
            
            # Fit GP
            gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, n_restarts_optimizer=1)
            gp.fit(X_obs, y_norm)
            
            # Define Expected Improvement acquisition function
            def EI(X):
                """Expected Improvement for minimization."""
                X = np.atleast_2d(X)
                y_pred, y_std = gp.predict(X, return_std=True)
                # Convert back to original scale
                y_pred_orig = y_pred * y_std + y_mean
                y_std_orig = y_std * y_std
                
                # Best observed value
                best_y = np.min(y_obs)
                
                # Avoid division by zero
                y_std_orig = np.maximum(y_std_orig, 1e-8)
                
                # Improvement
                improvement = best_y - y_pred_orig
                Z = improvement / y_std_orig
                from scipy.stats import norm
                EI_val = improvement * norm.cdf(Z) + y_std_orig * norm.pdf(Z)
                return -np.mean(EI_val)  # Negate for minimization
            
            # Optimize acquisition function using multiple random restarts
            n_restarts = 5
            best_acq_x = None
            best_acq_val = np.inf
            
            for _ in range(n_restarts):
                x0 = rng.uniform(-4, 4, size=dim)
                try:
                    result = minimize(EI, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 50})
                    if result.fun < best_acq_val:
                        best_acq_val = result.fun
                        best_acq_x = result.x
                except Exception:
                    continue
            
            # Fallback to random if optimization fails
            if best_acq_x is None:
                best_acq_x = rng.uniform(-4, 4, size=dim)
            
            # Evaluate the selected point
            fx_new = wrapped_func(best_acq_x)
            X_obs = np.vstack([X_obs, [best_acq_x]])
            y_obs = np.append(y_obs, fx_new)
            
            if fx_new < best_f:
                best_f = fx_new
                best_x = best_acq_x.copy()
        
        # Phase 3: Final L-BFGS-B polish from the best point
        if best_x is not None and call_tracker[0] < budget - 5:
            if time.time() < deadline:
                try:
                    result = minimize(
                        wrapped_func,
                        best_x,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 50, 'maxls': 20}
                    )
                    
                    if result.fun < best_f:
                        best_f = result.fun
                        best_x = result.x.copy()
                        
                except Exception:
                    pass
        
        return best_x, best_f
# EVOLVE-BLOCK-END
