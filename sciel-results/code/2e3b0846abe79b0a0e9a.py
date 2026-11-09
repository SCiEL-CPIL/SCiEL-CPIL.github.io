"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel

class Optimizer:
    """Surrogate-assisted optimizer for BBOB f11 (Discus) using Gaussian Process."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Deadline guard: 330s wall-clock limit, keep 10s margin
        start_time = time.time()
        deadline = start_time + 320
        
        # State tracking
        self.call_count = 0
        self.best_x = None
        self.best_f = np.inf
        X_obs = []
        y_obs = []
        
        def wrapped_func(x):
            if self.call_count >= self.budget:
                return self.best_f
            self.call_count += 1
            fx = func(x)
            if fx < self.best_f:
                self.best_f = fx
                self.best_x = x.copy()
            return fx

        # 1. Initial Exploration: LHS (Latin Hypercube) or Random
        # Use a reasonable number of points to initialize the GP.
        # dim=5, budget=500. 20-30 points is a good start.
        n_init = 25
        # Latin Hypercube Sampling for better coverage
        # Simple implementation of LHS
        rng = np.random.default_rng(42)
        X_init = np.zeros((n_init, self.dim))
        for i in range(self.dim):
            points = np.linspace(-5, 5, n_init)
            rng.shuffle(points)
            X_init[:, i] = points
            
        for x in X_init:
            if self.call_count >= self.budget:
                break
            fx = wrapped_func(x)
            X_obs.append(x)
            y_obs.append(fx)
            
        X_obs = np.array(X_obs)
        y_obs = np.array(y_obs)
        
        # 2. Surrogate-Assisted Loop
        # Use a Matern kernel with length scales to capture the "Discus" shape (anisotropic)
        # nu=2.5 is a good default for smooth functions
        kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(self.dim), nu=2.5)
        
        # We will update the GP periodically or every step? 
        # Updating every step is O(N^3) and might be slow for 500 steps.
        # Let's update every 5-10 steps or when we have significantly new data.
        # Given dim=5 and budget=500, even O(500^3) is too slow if done naively every step.
        # 500^3 = 125M operations. In Python/NumPy, this might be slow.
        # Let's try updating every 10 steps.
        
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=1, normalize_y=True)
        
        # Fit initial GP
        try:
            gp.fit(X_obs, y_obs)
        except Exception:
            # Fallback if GP fails
            pass

        # Acquisition Function: Expected Improvement (EI)
        # We need to maximize EI, which is equivalent to minimizing -EI.
        # EI(x) = (mu_best - mu(x) - sigma) * Phi(Z) + sigma * phi(Z)
        # where Z = (mu_best - mu(x) - sigma) / sigma
        
        def ei(x):
            # x is a single point or array of points
            # GP.predict returns mean and std
            mu, std = gp.predict(x, return_std=True)
            mu = mu.flatten()
            std = std.flatten()
            
            # Avoid division by zero
            std = np.maximum(std, 1e-8)
            
            # best_f is the minimum found so far
            # We want to find points that might be lower than best_f
            # Standard EI for minimization:
            # improvement = best_f - mu
            # Z = improvement / std
            # EI = improvement * Phi(Z) + std * phi(Z)
            
            improvement = self.best_f - mu
            Z = improvement / std
            # Phi and phi from scipy.stats
            from scipy.stats import norm
            EI = improvement * norm.cdf(Z) + std * norm.pdf(Z)
            
            # Return negative EI because we will use a minimizer to find the max
            return -EI
        
        # We will use a local minimizer (Nelder-Mead) to find the maximum of EI
        # in the neighborhood of current best points or globally.
        
        while self.call_count < self.budget and time.time() < deadline:
            # If GP hasn't been fitted or is too small, just do random search
            if len(X_obs) < 10:
                x_new = rng.uniform(-5, 5, self.dim)
                fx_new = wrapped_func(x_new)
                X_obs.append(x_new)
                y_obs.append(fx_new)
                continue
                
            # Update GP periodically
            if self.call_count % 10 == 0 or len(X_obs) == n_init:
                try:
                    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=1, normalize_y=True)
                    gp.fit(np.array(X_obs), np.array(y_obs))
                except Exception:
                    pass

            # Find the point that maximizes EI
            # We can try starting the local search from a few random points or the current best
            # To avoid getting stuck in the same local optimum of the surrogate,
            # let's try a few random starts for the EI maximization
            
            best_ei_x = None
            best_ei_val = -np.inf
            
            # Try a few random starts for the EI maximizer
            for _ in range(5):
                x_start = rng.uniform(-5, 5, self.dim)
                try:
                    res = minimize(ei, x_start, method='Nelder-Mead', options={'xatol': 1e-4, 'fatol': 1e-4, 'maxiter': 200})
                    if res.success:
                        ei_val = -res.fun # res.fun is -EI
                        if ei_val > best_ei_val:
                            best_ei_val = ei_val
                            best_ei_x = res.x
                except Exception:
                    continue
                    
            if best_ei_x is None:
                # Fallback to random
                best_ei_x = rng.uniform(-5, 5, self.dim)
                
            # Evaluate the chosen point
            x_new = np.clip(best_ei_x, -5, 5)
            fx_new = wrapped_func(x_new)
            X_obs.append(x_new)
            y_obs.append(fx_new)
            
            # Check time
            if time.time() > deadline - 10:
                break

        return self.best_x, self.best_f
# EVOLVE-BLOCK-END
