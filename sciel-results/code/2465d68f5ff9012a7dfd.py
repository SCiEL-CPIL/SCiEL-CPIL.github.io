"""Surrogate-assisted optimizer using Gaussian Process + Expected Improvement."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.stats.qmc import Sobol

class Optimizer:
    """GP surrogate with Expected Improvement acquisition."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        # 1. Initialization with Sobol sequence for better space-filling
        n_init = min(20, budget)
        sampler = Sobol(d=dim, scramble=True)
        X = sampler.random(n_init) * 10 - 5  # Scale to [-5, 5]
        Y = np.array([func(x) for x in X])
        
        best_idx = np.argmin(Y)
        x_best = X[best_idx].copy()
        f_best = Y[best_idx]
        
        # Normalize Y for GP stability
        y_mean = np.mean(Y)
        y_std = np.std(Y)
        if y_std < 1e-8:
            y_std = 1.0
        Y_norm = (Y - y_mean) / y_std
        
        # GP Hyperparameters
        # We will use a fixed length scale for simplicity, but normalized Y helps.
        # Length scale of 1.0 in normalized space is a reasonable start.
        length_scale = 1.0
        signal_var = 1.0
        noise_var = 1e-3
        
        def kernel(A, B):
            """Squared exponential kernel."""
            diff = A[:, None, :] - B[None, :, :]
            sq_dist = np.sum(diff ** 2, axis=-1)
            return signal_var * np.exp(-sq_dist / (2 * length_scale ** 2))

        def gp_predict(X_new, X_train, Y_train_norm):
            """Predict mean and variance at X_new given training data."""
            K = kernel(X_train, X_train)
            K += noise_var * np.eye(len(X_train))
            K_new = kernel(X_train, X_new)
            
            try:
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_train_norm))
            except np.linalg.LinAlgError:
                K += 1e-6 * np.eye(len(X_train))
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_train_norm))
            
            mean_norm = K_new.T @ alpha
            v = np.linalg.solve(L, K_new.T)
            var_norm = signal_var - np.sum(v ** 2, axis=0)
            var_norm = np.maximum(var_norm, 1e-8)
            
            # Denormalize mean and variance
            mean = mean_norm * y_std + y_mean
            var = var_norm * (y_std ** 2)
            
            return mean, var

        def expected_improvement(x, f_best, X_train, Y_train_norm):
            """Compute EI for a single point x."""
            x = np.array([x])
            mean, var = gp_predict(x, X_train, Y_train_norm)
            sigma = np.sqrt(var[0])
            if sigma < 1e-8:
                return 0.0
            z = (f_best - mean[0]) / sigma
            ei = (f_best - mean[0]) * norm.cdf(z) + sigma * norm.pdf(z)
            return -ei  # Negative for minimization

        # Main optimization loop
        calls_made = n_init
        X_train = X
        Y_train_norm = Y_norm
        
        for _ in range(budget - n_init):
            # Find next point using EI
            # Multi-start L-BFGS-B
            n_restarts = 5
            best_ei = -np.inf
            x_next = None
            
            # Start points: current best, centroid, and random
            centroid = np.mean(X_train, axis=0)
            start_points = [x_best, centroid] + [rng.uniform(-5, 5, dim) for _ in range(n_restarts - 2)]
            
            for x0 in start_points:
                x0 = np.clip(x0, -5, 5)
                try:
                    res = minimize(
                        lambda x: expected_improvement(x, f_best, X_train, Y_train_norm),
                        x0,
                        bounds=[(-5, 5)] * dim,
                        method='L-BFGS-B',
                        options={'maxiter': 50}
                    )
                    if res.fun < best_ei:
                        best_ei = res.fun
                        x_next = np.clip(res.x, -5, 5)
                except Exception:
                    continue
            
            if x_next is None:
                x_next = rng.uniform(-5, 5, dim)
            
            # Evaluate the new point
            f_new = func(x_next)
            
            # Update training data
            X_train = np.vstack([X_train, x_next])
            
            # Update normalization
            y_mean = np.mean(np.append(Y_train_norm * y_std + y_mean, f_new))
            y_std = np.std(np.append(Y_train_norm * y_std + y_mean, f_new))
            if y_std < 1e-8:
                y_std = 1.0
            Y_all = np.append(Y_train_norm * y_std + y_mean, f_new)
            Y_train_norm = (Y_all - y_mean) / y_std
            
            calls_made += 1
            
            # Update best
            if f_new < f_best:
                f_best = f_new
                x_best = x_next.copy()
        
        return x_best, f_best
# EVOLVE-BLOCK-END
