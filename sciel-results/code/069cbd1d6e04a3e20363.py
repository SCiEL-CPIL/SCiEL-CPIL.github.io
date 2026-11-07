"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, ConstantKernel, WhiteKernel
from scipy.stats import norm

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.func = None
        self.best_x = None
        self.best_f = float('inf')

    def _evaluate(self, x):
        if self.calls >= self.budget:
            return None
        self.calls += 1
        val = self.func(x)
        if val < self.best_f:
            self.best_f = val
            self.best_x = x.copy()
        return val

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 300s limit
        
        d = self.dim
        self.func = func
        self.calls = 0
        self.best_f = float('inf')
        self.best_x = None
        
        rng = np.random.default_rng(42)
        
        # Strategy:
        # 1. Multi-start L-BFGS-B to handle hidden shifts/rotations and find the basin.
        #    Bent Cigar is unimodal but ill-conditioned. L-BFGS-B with bounds is effective
        #    if started reasonably. Multi-start increases robustness.
        # 2. Bayesian Optimization (Gaussian Process) to efficiently map the landscape
        #    and guide sampling toward the narrow valley without relying on local gradients.
        # 3. Nelder-Mead for final polishing. NM is derivative-free and can handle
        #    the final convergence where other methods might stall due to ill-conditioning.
        
        def wrapper_func(x, deadline):
            if self.calls >= self.budget or time.time() > deadline - 2:
                return 1e10
            f_val = self._evaluate(x)
            if f_val is None:
                return 1e10
            return f_val

        # Phase 1: Multi-start L-BFGS-B
        # Generate a few starting points. Origin is usually good, but random points
        # within the box might help if the shift is significant.
        
        # Start with origin
        x0 = np.zeros(d)
        f0 = self._evaluate(x0)
        if f0 is None:
            return np.zeros(d), 0.0

        # Run L-BFGS-B from origin
        try:
            res = minimize(wrapper_func, x0, method='L-BFGS-B', 
                           bounds=[(-5, 5)] * d, 
                           args=(deadline,),
                           options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-6})
        except Exception:
            pass

        # Check budget and time
        if self.calls < self.budget - 50 and time.time() < deadline - 30:
            # Try a few random starts if we have budget
            for _ in range(3):
                if self.calls >= self.budget - 50 or time.time() > deadline - 20:
                    break
                
                # Random point within [-3, 3] to stay safely inside bounds
                x_start = rng.uniform(-3, 3, d)
                
                try:
                    res2 = minimize(wrapper_func, x_start, method='L-BFGS-B', 
                                    bounds=[(-5, 5)] * d, 
                                    args=(deadline,),
                                    options={'maxiter': 30, 'ftol': 1e-10, 'gtol': 1e-6})
                except Exception:
                    pass

        # Phase 2: Bayesian Optimization
        # Use a Gaussian Process to model the function and select the next point
        # to evaluate using Expected Improvement (EI).
        
        if self.calls < self.budget - 20 and time.time() < deadline - 30:
            # Collect data points from previous phase
            # Note: We didn't store all points, so we need to re-evaluate or keep a log.
            # To avoid re-evaluating, we should have stored points. 
            # Let's restart the BO phase with a small initial set if we didn't keep history.
            # For simplicity and to avoid excessive calls, we will do a quick local BO
            # around the current best.
            
            if self.best_x is not None:
                # Create a small initial set around best_x for BO
                X_bo = [self.best_x]
                y_bo = [self.best_f]
                
                # Add a few random points in a small neighborhood to give GP some variance
                n_init_bo = min(10, self.budget - self.calls - 10)
                for _ in range(n_init_bo):
                    if self.calls >= self.budget - 10 or time.time() > deadline - 20:
                        break
                    x_new = self.best_x + rng.normal(0, 0.5, d)
                    x_new = np.clip(x_new, -5, 5)
                    f_new = self._evaluate(x_new)
                    if f_new is not None:
                        X_bo.append(x_new)
                        y_bo.append(f_new)
                
                if len(X_bo) < 2:
                    return self.best_x, self.best_f

                X_bo = np.array(X_bo)
                y_bo = np.array(y_bo)
                
                # Normalize y for better GP performance
                y_min = np.min(y_bo)
                y_range = np.max(y_bo) - y_min
                if y_range < 1e-10:
                    y_norm = y_bo
                else:
                    y_norm = (y_bo - y_min) / y_range
                
                # Define kernel
                # Matern is good for smooth but non-analytic functions
                kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(d), nu=2.5) + WhiteKernel(noise_level=1e-6)
                gp = GaussianProcessRegressor(kernel=kernel, normalize_y=False, n_restarts_optimizer=1, alpha=1e-10)
                
                # Fit initial GP
                try:
                    gp.fit(X_bo, y_norm)
                except Exception:
                    pass
                else:
                    # BO Loop
                    n_bo_iter = min(30, self.budget - self.calls - 10)
                    for i in range(n_bo_iter):
                        if self.calls >= self.budget - 10 or time.time() > deadline - 20:
                            break
                        
                        # Generate candidate points
                        n_candidates = 500
                        X_candidates = rng.uniform(-5, 5, (n_candidates, d))
                        
                        # Predict mean and std
                        try:
                            mu, sigma = gp.predict(X_candidates, return_std=True)
                        except Exception:
                            break
                        
                        # Calculate Expected Improvement
                        # EI = (mu - y_best) * Phi(Z) + sigma * phi(Z)
                        # where Z = (mu - y_best) / sigma
                        # y_best is the minimum of y_norm
                        y_best = np.min(y_bo)
                        
                        with np.errstate(divide='ignore', invalid='ignore'):
                            Z = (mu - y_best) / sigma
                            # Handle sigma == 0
                            Z = np.where(sigma > 1e-10, Z, 0)
                            ei = (mu - y_best) * norm.cdf(Z) + sigma * norm.pdf(Z)
                        
                        # Select point with highest EI
                        idx_max = np.argmax(ei)
                        x_next = X_candidates[idx_max]
                        
                        # Evaluate
                        f_next = self._evaluate(x_next)
                        if f_next is None:
                            break
                        
                        # Update data
                        X_bo = np.vstack([X_bo, x_next])
                        y_bo = np.append(y_bo, f_next)
                        
                        # Update y_norm
                        y_min = np.min(y_bo)
                        y_range = np.max(y_bo) - y_min
                        if y_range < 1e-10:
                            y_norm = y_bo
                        else:
                            y_norm = (y_bo - y_min) / y_range
                        
                        # Refit GP
                        try:
                            gp.fit(X_bo, y_norm)
                        except Exception:
                            break

        # Phase 3: Nelder-Mead Polish
        # Use NM to refine the best point found so far.
        # NM is good for final convergence on smooth functions.
        
        if self.calls < self.budget - 10 and time.time() < deadline - 10:
            if self.best_x is None:
                self.best_x = x0.copy()
            
            try:
                res3 = minimize(wrapper_func, self.best_x, method='Nelder-Mead',
                                args=(deadline,),
                                options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-12})
            except Exception:
                pass
        
        # Return the best found
        if self.best_x is None:
            self.best_x = np.zeros(d)
            self.best_f = 0.0
            
        return self.best_x, self.best_f

# EVOLVE-BLOCK-END
