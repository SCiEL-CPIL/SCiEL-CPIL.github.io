# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.stats import qmc


class Optimizer:
    """Gaussian Process Bayesian Optimizer with Lower Confidence Bound acquisition."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _gp_predict(self, X_new, X, y, length_scale, noise, L_cache=None):
        """
        Compute GP mean and variance at X_new.
        """
        n = X.shape[0]
        
        if L_cache is None:
            diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
            sq_dist = np.sum(diff ** 2, axis=-1)
            K = np.exp(-0.5 * sq_dist / (length_scale ** 2)) + noise * np.eye(n)
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                K += 1e-8 * np.eye(n)
                L = np.linalg.cholesky(K)
        else:
            L = L_cache

        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
        
        diff_new = X_new[:, np.newaxis, :] - X[np.newaxis, :, :]
        sq_dist_new = np.sum(diff_new ** 2, axis=-1)
        K_star = np.exp(-0.5 * sq_dist_new / (length_scale ** 2))
        
        mean = K_star @ alpha
        v = np.linalg.solve(L, K_star.T)
        var = 1.0 - np.sum(v ** 2, axis=0)
        var = np.maximum(var, 1e-10)
        
        return mean, var

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        # Initial points: Use Sobol sequence for better space filling
        n_init = min(20, budget)
        sampler = qmc.Sobol(d=dim, scramble=True)
        init_points = sampler.random(n_init)
        X = init_points * 10.0 - 5.0 # Map to [-5, 5]
        
        y = np.array([func(x) for x in X])
        
        # Initial hyperparameters
        length_scale = 1.0
        noise = 1e-5
        
        # Track best
        y_best = np.min(y)
        x_best = X[np.argmin(y)]
        
        # Cache Cholesky factor
        def compute_L():
            diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
            sq_dist = np.sum(diff ** 2, axis=-1)
            K = np.exp(-0.5 * sq_dist / (length_scale ** 2)) + noise * np.eye(X.shape[0])
            try:
                return np.linalg.cholesky(K)
            except:
                K += 1e-8 * np.eye(X.shape[0])
                return np.linalg.cholesky(K)
        
        L = compute_L()

        # Bayesian optimization loop
        for i in range(n_init, budget):
            # Adaptive beta for LCB
            # Start with high exploration, decay to exploitation
            progress = (i - n_init) / (budget - n_init)
            beta = 3.0 * (1.0 - progress) + 0.1
            
            # Generate candidate points
            n_candidates = 200
            
            # 1. Random points
            cand_random = rng.uniform(-5, 5, (60, dim))
            
            # 2. Sobol points for structured exploration
            sampler_cand = qmc.Sobol(d=dim, scramble=True)
            cand_sobol = sampler_cand.random(60) * 10.0 - 5.0
            
            # 3. Local perturbations around best
            n_local = 80
            # Adaptive local sigma: shrink as we progress
            local_sigma = max(0.05, 0.5 * (1.0 - progress))
            cand_local = x_best + local_sigma * rng.standard_normal((n_local, dim))
            cand_local = np.clip(cand_local, -5, 5)
            
            candidates = np.vstack([cand_random, cand_sobol, cand_local])
            
            # Evaluate GP
            mean, var = self._gp_predict(candidates, X, y, length_scale, noise, L)
            
            # LCB Acquisition: Minimize mean - beta * std
            std = np.sqrt(var)
            lcb = mean - beta * std
            
            # Select best candidate (lowest LCB)
            best_idx = np.argmin(lcb)
            x_new = candidates[best_idx]
            
            # Evaluate function
            f_new = func(x_new)
            
            # Update data
            X = np.vstack([X, x_new])
            y = np.append(y, f_new)
            
            # Update best
            if f_new < y_best:
                y_best = f_new
                x_best = x_new
            
            # Update Cholesky factor
            L = compute_L()

        return x_best, y_best
# EVOLVE-BLOCK-END
