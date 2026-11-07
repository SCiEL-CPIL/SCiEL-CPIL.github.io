"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


class Optimizer:
    """Derivative-free optimizer using Bayesian Optimization with GP surrogate and Nelder-Mead refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim
        self.calls = 0

    def __call__(self, func):
        self.calls = 0
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf

        def wrapped_func(x):
            nonlocal best_x, best_f, self
            x = np.asarray(x, dtype=float)
            x = np.clip(x, -5.0, 5.0)
            val = float(func(x))
            self.calls += 1
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        # Phase 1: Initial random sampling to build training data for GP
        n_init = max(15, budget // 10)
        X_train = []
        y_train = []
        for _ in range(n_init):
            if self.calls >= budget:
                break
            x = rng.uniform(-5, 5, dim)
            val = wrapped_func(x)
            X_train.append(x)
            y_train.append(val)

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        # Phase 2: Bayesian Optimization with GP surrogate
        # Use about 50% of remaining budget for BO
        n_bo = max(20, (budget - self.calls) // 2)
        
        # GP parameters
        length_scale = 1.0
        sigma_f = 1.0
        sigma_n = 1e-4
        alpha = 1e-6

        def gp_kernel(X1, X2):
            """RBF kernel."""
            sq_dists = np.sum(X1**2, axis=1)[:, None] - 2 * X1 @ X2.T + np.sum(X2**2, axis=1)[None, :]
            sq_dists = np.maximum(sq_dists, 0)
            return sigma_f**2 * np.exp(-sq_dists / (2 * length_scale**2))

        def gp_predict(X_test, X_train, y_train):
            """GP mean and variance prediction."""
            K = gp_kernel(X_train, X_train) + alpha * np.eye(len(X_train))
            K_s = gp_kernel(X_train, X_test)
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                K += 1e-8 * np.eye(len(X_train))
                L = np.linalg.cholesky(K)
            
            v = np.linalg.solve(L, y_train)
            v = np.linalg.solve(L.T, v)
            
            mean = K_s.T @ v
            v2 = np.linalg.solve(L, K_s)
            var = sigma_f**2 - np.sum(v2**2, axis=0)
            var = np.maximum(var, 0)
            return mean, var

        def expected_improvement(x, y_best, mu, sigma):
            """Calculate Expected Improvement."""
            sigma = np.maximum(sigma, 1e-8)
            z = (y_best - mu) / sigma
            ei = (y_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
            return ei

        for _ in range(n_bo):
            if self.calls >= budget:
                break
            
            # Evaluate EI on a grid of candidate points
            n_candidates = 200
            X_candidates = rng.uniform(-5, 5, (n_candidates, dim))
            
            # Add some points near the current best
            if best_x is not None:
                X_near = best_x + rng.normal(0, 0.5, (50, dim))
                X_near = np.clip(X_near, -5, 5)
                X_candidates = np.vstack([X_candidates, X_near])
            
            mu, var = gp_predict(X_candidates, X_train, y_train)
            sigma = np.sqrt(var)
            
            y_best = np.min(y_train)
            ei = expected_improvement(X_candidates, y_best, mu, sigma)
            
            # Select the point with maximum EI
            idx = np.argmax(ei)
            x_next = X_candidates[idx]
            x_next = np.clip(x_next, -5, 5)
            
            val = wrapped_func(x_next)
            X_train = np.vstack([X_train, x_next])
            y_train = np.append(y_train, val)
            
            # Adapt length scale based on progress
            if len(y_train) % 10 == 0:
                # Increase length scale as we get more data (more exploitation)
                length_scale = min(2.0, length_scale * 1.1)

        # Phase 3: Nelder-Mead refinement from the best point
        if self.calls < budget:
            remaining = budget - self.calls
            nm_budget = remaining
            
            try:
                result = minimize(
                    wrapped_func,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': nm_budget, 'maxfev': nm_budget, 'xatol': 1e-8, 'fatol': 1e-10}
                )
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
