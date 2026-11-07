"""Surrogate-assisted Bayesian optimization for BBOB f9."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
import time


class Optimizer:
    """
    Surrogate-assisted optimizer using Gaussian Process and Expected Improvement.
    Uses local optimization to maximize the acquisition function.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = [(-5.0, 5.0)] * dim

    def _compute_kernel(self, X1, X2, length_scale):
        """Compute RBF kernel between X1 and X2."""
        # X1: (n1, d), X2: (n2, d)
        # diff: (n1, n2, d)
        diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
        sq_dists = np.sum(diff ** 2, axis=-1)
        return np.exp(-0.5 * sq_dists / (length_scale ** 2))

    def _gp_predict(self, X_train, y_train, X_test, length_scale, noise=1e-6):
        """
        Predict mean and variance using GP.
        Returns (mu, var).
        """
        n = X_train.shape[0]
        K = self._compute_kernel(X_train, X_train, length_scale)
        K += noise * np.eye(n)

        # Normalize y for numerical stability
        y_mean = np.mean(y_train)
        y_std = np.std(y_train) + 1e-10
        y_norm = (y_train - y_mean) / y_std

        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            # Add jitter
            K += 1e-4 * np.eye(n)
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                # Fallback
                return np.full(X_test.shape[0], y_mean), np.full(X_test.shape[0], y_std ** 2)

        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_norm))

        K_star = self._compute_kernel(X_test, X_train, length_scale)
        mu = K_star @ alpha * y_std + y_mean

        v = np.linalg.solve(L, K_star.T)
        var = 1.0 - np.sum(v ** 2, axis=0)
        var = np.maximum(var, 1e-10)
        var *= y_std ** 2

        return mu, var

    def _acquire_ei(self, x, X_train, y_train, length_scale, y_best, noise=1e-6):
        """
        Calculate negative Expected Improvement for a point x.
        We minimize -EI.
        """
        x = np.array(x).reshape(1, -1)
        mu, var = self._gp_predict(X_train, y_train, x, length_scale, noise)
        std = np.sqrt(var[0])
        
        if std < 1e-8:
            return 0.0
            
        improvement = y_best - mu[0]
        z = improvement / std
        ei = improvement * norm.cdf(z) + std * norm.pdf(z)
        
        # Return negative EI for minimization
        return -ei

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # 1. Initialization
        n_init = min(20, budget)
        # Use LHS for better initial coverage
        X_init = np.zeros((n_init, dim))
        for d in range(dim):
            perm = rng.permutation(n_init)
            X_init[:, d] = (perm + 0.5) / n_init * 10.0 - 5.0

        y_init = np.array([func(x) for x in X_init])
        calls_made = n_init

        best_idx = np.argmin(y_init)
        best_x = X_init[best_idx].copy()
        best_y = y_init[best_idx]

        X = X_init.copy()
        y = y_init.copy()

        # Initial length scale estimate
        # Heuristic: based on the spread of the best few points
        def estimate_length_scale(X, y, k=5):
            if len(y) < k:
                k = len(y)
            if k < 2:
                return 1.0
            sorted_idx = np.argsort(y)
            top_k = X[sorted_idx[:k]]
            dists = []
            for i in range(k):
                for j in range(i + 1, k):
                    dists.append(np.linalg.norm(top_k[i] - top_k[j]))
            if not dists:
                return 1.0
            # Use a fraction of the median distance
            return max(0.5, np.median(dists) * 0.5)

        length_scale = estimate_length_scale(X, y)

        # 2. Main Loop
        while calls_made < budget:
            # Adapt length scale occasionally to track the "scale" of the landscape
            if calls_made % 20 == 0:
                length_scale = estimate_length_scale(X, y)
                # Ensure it doesn't get too small
                length_scale = max(0.2, length_scale)

            # Find the next point by maximizing EI
            # We use multiple starting points for the local optimizer to avoid local maxima in EI
            n_starts = 10
            starts = []
            
            # Start 1: Current best
            starts.append(best_x)
            
            # Start 2-4: Random points
            for _ in range(3):
                starts.append(rng.uniform(-5, 5, dim))
                
            # Start 5-10: Points with high predicted variance
            # Sample some random points, predict variance, pick top ones
            X_sample = rng.uniform(-5, 5, (20, dim))
            mu_s, var_s = self._gp_predict(X, y, X_sample, length_scale)
            top_var_idx = np.argsort(var_s)[-5:]
            for idx in top_var_idx:
                starts.append(X_sample[idx])

            best_ei_val = -np.inf
            next_x = None

            for s in starts:
                # Ensure start is within bounds
                s = np.clip(s, -5, 5)
                res = minimize(
                    self._acquire_ei,
                    s,
                    args=(X, y, length_scale, best_y),
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-4}
                )
                if res.fun < best_ei_val:
                    best_ei_val = res.fun
                    next_x = res.x

            # If optimization failed or returned invalid, fallback to random
            if next_x is None or not np.all(np.isfinite(next_x)):
                next_x = rng.uniform(-5, 5, dim)
            else:
                next_x = np.clip(next_x, -5, 5)

            # Evaluate
            f_new = func(next_x)
            calls_made += 1

            # Update data
            X = np.vstack([X, next_x])
            y = np.append(y, f_new)

            # Update best
            if f_new < best_y:
                best_y = f_new
                best_x = next_x.copy()
                # When we find a new best, we might want to refine length scale quickly
                # but we'll wait for the periodic update to avoid instability

        return best_x, best_y
# EVOLVE-BLOCK-END
