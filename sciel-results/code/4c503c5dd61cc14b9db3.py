"""Surrogate-assisted Bayesian optimization for BBOB f9."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


class Optimizer:
    """Bayesian optimization with GP surrogate and Expected Improvement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _gp_predict(self, X_train, y_train, X_test, length_scale=1.0, noise=1e-8):
        """
        Simple GP with RBF kernel. Returns mean and variance at X_test.
        """
        n = X_train.shape[0]
        d = X_train.shape[1]

        # Compute kernel matrix K
        # RBF kernel: K(i,j) = exp(-0.5 * ||xi - xj||^2 / length_scale^2)
        diff = X_train[:, np.newaxis, :] - X_train[np.newaxis, :, :]
        sq_dists = np.sum(diff ** 2, axis=-1)
        K = np.exp(-0.5 * sq_dists / (length_scale ** 2))
        K += noise * np.eye(n)

        # Normalize y for numerical stability
        y_mean = np.mean(y_train)
        y_std = np.std(y_train) + 1e-10
        y_norm = (y_train - y_mean) / y_std

        # Cholesky decomposition
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            # Add jitter and retry
            K += 1e-6 * np.eye(n)
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                # Fallback: return mean and high variance
                return np.full(X_test.shape[0], y_mean), np.full(X_test.shape[0], y_std ** 2)

        # Solve for alpha
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_norm))

        # Predict at X_test
        diff_test = X_test[:, np.newaxis, :] - X_train[np.newaxis, :, :]
        sq_dists_test = np.sum(diff_test ** 2, axis=-1)
        K_star = np.exp(-0.5 * sq_dists_test / (length_scale ** 2))

        # Mean prediction
        mu = K_star @ alpha * y_std + y_mean

        # Variance prediction
        # var = 1 - K_star @ L^{-T} L^{-1} @ K_star.T
        # Use Cholesky: L^{-1} K_star.T
        v = np.linalg.solve(L, K_star.T)
        var = 1.0 - np.sum(v ** 2, axis=0)
        var = np.maximum(var, 1e-10)
        var *= y_std ** 2

        return mu, var

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Initial sampling: use a mix of random points and LHS
        n_init = min(20, budget)
        X = np.zeros((n_init, dim))
        # Latin Hypercube Sampling
        for d in range(dim):
            perm = rng.permutation(n_init)
            X[:, d] = (perm + 0.5) / n_init * 10.0 - 5.0

        y = np.zeros(n_init)
        for i in range(n_init):
            y[i] = func(X[i])

        # Adaptive length scale based on data spread
        # Estimate typical scale from best vs worst
        best_idx = np.argmin(y)
        worst_idx = np.argmax(y)
        dist_best_worst = np.linalg.norm(X[best_idx] - X[worst_idx])
        length_scale = max(0.5, dist_best_worst / 3.0)

        calls_made = n_init
        best_x = X[best_idx].copy()
        best_y = y[best_idx]

        # Bayesian optimization loop
        while calls_made < budget:
            # Generate candidate points for acquisition
            n_candidates = 200
            # Mix of random points and points near current best
            X_rand = rng.uniform(-5, 5, (n_candidates, dim))
            
            # Add some points around the best solution
            n_local = 50
            X_local = best_x + rng.normal(0, length_scale * 0.5, (n_local, dim))
            X_local = np.clip(X_local, -5, 5)
            
            X_cand = np.vstack([X_rand, X_local])
            
            # Predict with GP
            mu, var = self._gp_predict(X, y, X_cand, length_scale)
            std = np.sqrt(var)
            
            # Expected Improvement
            eps = 1e-8
            y_best = np.min(y)
            improvement = y_best - mu
            z = improvement / (std + eps)
            EI = improvement * norm.cdf(z) + std * norm.pdf(z)
            EI = np.where(std > eps, EI, 0.0)
            
            # Select best candidate
            idx = np.argmax(EI)
            x_new = X_cand[idx]
            
            # Evaluate
            f_new = func(x_new)
            calls_made += 1
            
            # Update dataset
            X = np.vstack([X, x_new])
            y = np.append(y, f_new)
            
            # Update best
            if f_new < best_y:
                best_y = f_new
                best_x = x_new.copy()
            
            # Adapt length scale occasionally
            if calls_made % 50 == 0 and len(y) > 10:
                # Re-estimate length scale
                sorted_idx = np.argsort(y)
                top_k = min(5, len(y))
                best_points = X[sorted_idx[:top_k]]
                if top_k > 1:
                    # Typical distance between good points
                    dists = []
                    for i in range(top_k):
                        for j in range(i + 1, top_k):
                            dists.append(np.linalg.norm(best_points[i] - best_points[j]))
                    if dists:
                        length_scale = max(0.2, np.mean(dists) * 0.5)

        return best_x, best_y
# EVOLVE-BLOCK-END
