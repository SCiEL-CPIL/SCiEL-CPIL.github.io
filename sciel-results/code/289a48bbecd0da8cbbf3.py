"""Surrogate-assisted optimizer using Gaussian Process + Expected Improvement."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


class Optimizer:
    """GP surrogate with Expected Improvement acquisition."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Initial random sampling for warm start
        n_init = min(10, budget)
        X = np.random.uniform(-5, 5, size=(n_init, dim))
        Y = np.array([func(x) for x in X])

        best_idx = np.argmin(Y)
        x_best = X[best_idx].copy()
        f_best = Y[best_idx]

        # GP hyperparameters (fixed for simplicity and speed)
        # We'll use a simple squared exponential kernel with fixed length scales
        length_scale = 1.5
        signal_var = 1.0
        noise_var = 1e-4

        def kernel(A, B):
            """Squared exponential kernel."""
            # A: (n, dim), B: (m, dim)
            diff = A[:, None, :] - B[None, :, :]
            sq_dist = np.sum(diff ** 2, axis=-1)
            return signal_var * np.exp(-sq_dist / (2 * length_scale ** 2))

        def gp_predict(X_new, X_train, Y_train):
            """Predict mean and variance at X_new given training data."""
            K = kernel(X_train, X_train)
            K += noise_var * np.eye(len(X_train))
            K_new = kernel(X_train, X_new)
            
            try:
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_train))
            except np.linalg.LinAlgError:
                K += 1e-6 * np.eye(len(X_train))
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_train))
            
            mean = K_new.T @ alpha
            v = np.linalg.solve(L, K_new.T)
            var = signal_var - np.sum(v ** 2, axis=0)
            var = np.maximum(var, 1e-8)
            return mean, var

        def expected_improvement(x, f_best, X_train, Y_train):
            """Compute EI for a single point x."""
            x = np.array([x])
            mean, var = gp_predict(x, X_train, Y_train)
            sigma = np.sqrt(var[0])
            if sigma < 1e-8:
                return 0.0
            z = (f_best - mean[0]) / sigma
            ei = (f_best - mean[0]) * norm.cdf(z) + sigma * norm.pdf(z)
            return -ei  # Negative for minimization

        # Main optimization loop
        calls_made = n_init
        for _ in range(budget - n_init):
            # Fit GP and find next point using EI
            X_train = X
            Y_train = Y
            
            # Use multiple random starts to find the argmax of EI
            n_restarts = 5
            best_ei = -np.inf
            x_next = None
            
            # Start from current best and random points
            start_points = [x_best] + [rng.uniform(-5, 5, dim) for _ in range(n_restarts - 1)]
            
            for x0 in start_points:
                x0 = np.clip(x0, -5, 5)
                res = minimize(
                    lambda x: expected_improvement(x, f_best, X_train, Y_train),
                    x0,
                    bounds=[(-5, 5)] * dim,
                    method='L-BFGS-B',
                    options={'maxiter': 50}
                )
                if res.fun < best_ei:
                    best_ei = res.fun
                    x_next = np.clip(res.x, -5, 5)
            
            if x_next is None:
                x_next = rng.uniform(-5, 5, dim)
            
            # Evaluate the new point
            f_new = func(x_next)
            
            # Update training data
            X = np.vstack([X, x_next])
            Y = np.append(Y, f_new)
            calls_made += 1
            
            # Update best
            if f_new < f_best:
                f_best = f_new
                x_best = x_next.copy()
        
        return x_best, f_best
# EVOLVE-BLOCK-END
