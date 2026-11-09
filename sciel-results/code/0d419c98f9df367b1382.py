"""Surrogate-assisted Bayesian Optimization with GP and EI acquisition for BBOB f21."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


class Optimizer:
    """Surrogate-assisted Bayesian Optimization with Gaussian Process and EI."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _gp_predict(self, X, X_train, y_train, length_scale, variance, noise):
        """GP mean and variance prediction."""
        n = len(X_train)
        K = self._kernel(X_train, X_train, length_scale, variance) + noise * np.eye(n)
        K_inv = np.linalg.inv(K + 1e-8 * np.eye(n))
        K_star = self._kernel(X, X_train, length_scale, variance)
        mu = K_star @ K_inv @ y_train
        K_ss = self._kernel(X, X, length_scale, variance)
        var = np.diag(K_ss - K_star @ K_inv @ K_star.T)
        var = np.maximum(var, 1e-10)
        return mu, np.sqrt(var)

    def _kernel(self, X1, X2, length_scale, variance):
        """Matern 5/2 kernel with ARD."""
        # X1: (n1, d), X2: (n2, d)
        ls = length_scale
        Z = np.zeros((X1.shape[0], X2.shape[0]))
        for d in range(X1.shape[1]):
            Z += (X1[:, d:d+1] - X2[:, d:d+1].T) ** 2 / (ls[d] ** 2)
        Z = np.sqrt(Z)
        return variance * (1 + np.sqrt(5) * Z + 5 / 3 * Z ** 2) * np.exp(-np.sqrt(5) * Z)

    def _fit_gp(self, X_train, y_train):
        """Fit GP hyperparameters by maximizing log-likelihood."""
        n = len(X_train)
        d = X_train.shape[1]
        # Initial hyperparameters
        length_scale = np.ones(d) * 1.0
        variance = 1.0
        noise = 1e-4

        # Normalize y for numerical stability
        y_mean = np.mean(y_train)
        y_std = max(np.std(y_train), 1e-8)
        y_norm = (y_train - y_mean) / y_std

        best_ll = -np.inf
        best_params = (length_scale.copy(), variance, noise)

        # Simple coordinate ascent for hyperparameters
        for iteration in range(5):
            # Try different length scales
            for ls_mult in [0.5, 1.0, 2.0]:
                ls_try = length_scale * ls_mult
                K = self._kernel(X_train, X_train, ls_try, variance) + noise * np.eye(n)
                try:
                    L = np.linalg.cholesky(K + 1e-8 * np.eye(n))
                    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_norm))
                    ll = -0.5 * y_norm @ alpha - np.sum(np.log(np.diag(L))) - 0.5 * n * np.log(2 * np.pi)
                    if ll > best_ll:
                        best_ll = ll
                        best_params = (ls_try.copy(), variance, noise)
                except:
                    pass

            # Try different variances
            for var_mult in [0.5, 1.0, 2.0]:
                var_try = variance * var_mult
                K = self._kernel(X_train, X_train, best_params[0], var_try) + best_params[2] * np.eye(n)
                try:
                    L = np.linalg.cholesky(K + 1e-8 * np.eye(n))
                    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_norm))
                    ll = -0.5 * y_norm @ alpha - np.sum(np.log(np.diag(L))) - 0.5 * n * np.log(2 * np.pi)
                    if ll > best_ll:
                        best_ll = ll
                        best_params = (best_params[0].copy(), var_try, best_params[2])
                except:
                    pass

            # Try different noise
            for noise_try in [1e-5, 1e-4, 1e-3]:
                K = self._kernel(X_train, X_train, best_params[0], best_params[1]) + noise_try * np.eye(n)
                try:
                    L = np.linalg.cholesky(K + 1e-8 * np.eye(n))
                    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_norm))
                    ll = -0.5 * y_norm @ alpha - np.sum(np.log(np.diag(L))) - 0.5 * n * np.log(2 * np.pi)
                    if ll > best_ll:
                        best_ll = ll
                        best_params = (best_params[0].copy(), best_params[1], noise_try)
                except:
                    pass

        return best_params[0], best_params[1], best_params[2], y_mean, y_std

    def _ei_acquisition(self, x, X_train, y_train, length_scale, variance, noise, y_mean, y_std, best_y):
        """Expected Improvement acquisition function (negative for minimization)."""
        x = x.reshape(1, -1)
        mu, sigma = self._gp_predict(x, X_train, y_train, length_scale, variance, noise)
        mu = mu[0] * y_std + y_mean
        sigma = sigma[0] * y_std
        if sigma < 1e-10:
            return 0.0
        z = (mu - best_y) / sigma
        ei = (best_y - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return -ei  # Negate for minimization

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Phase 1: Initial sampling
        n_init = min(30, budget // 10)
        X_list = []
        y_list = []

        # Random initialization
        for i in range(n_init):
            x = rng.uniform(-5, 5, dim)
            y = func(x)
            X_list.append(x)
            y_list.append(y)

        X = np.array(X_list)
        y = np.array(y_list)
        best_idx = np.argmin(y)
        best_x = X[best_idx].copy()
        best_y = y[best_idx]

        calls_made = n_init

        # Phase 2: GP-based Bayesian Optimization
        # Reserve some budget for final local search
        n_local_search = 50
        gp_budget = budget - n_init - n_local_search

        for gp_iter in range(min(10, gp_budget)):
            if calls_made >= budget:
                break

            # Fit GP
            try:
                length_scale, variance, noise, y_mean, y_std = self._fit_gp(X, y)
            except:
                # Fallback to random
                x_new = rng.uniform(-5, 5, dim)
                y_new = func(x_new)
                X_list.append(x_new)
                y_list.append(y_new)
                calls_made += 1
                if y_new < best_y:
                    best_x = x_new.copy()
                    best_y = y_new
                continue

            # Optimize EI acquisition using multiple random starts
            n_restarts = 20
            best_acq = np.inf
            best_x_new = None

            for r in range(n_restarts):
                if r == 0:
                    x0 = best_x.copy()
                elif r == 1:
                    x0 = rng.uniform(-5, 5, dim)
                else:
                    # Perturb best point
                    x0 = np.clip(best_x + rng.normal(0, 0.5, dim), -5, 5)

                result = minimize(
                    self._ei_acquisition,
                    x0,
                    args=(X, y, length_scale, variance, noise, y_mean, y_std, best_y),
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 50}
                )
                if result.fun < best_acq:
                    best_acq = result.fun
                    best_x_new = np.clip(result.x, -5, 5)

            if best_x_new is None:
                best_x_new = rng.uniform(-5, 5, dim)

            # Evaluate new point
            y_new = func(best_x_new)
            X_list.append(best_x_new)
            y_list.append(y_new)
            calls_made += 1

            if y_new < best_y:
                best_x = best_x_new.copy()
                best_y = y_new

            X = np.array(X_list)
            y = np.array(y_list)

        # Phase 3: Local (1+1)-ES around best point
        x = best_x.copy()
        fx = best_y
        sigma = 0.1

        for i in range(n_local_search):
            if calls_made >= budget:
                break
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls_made += 1
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.2
            else:
                sigma *= 0.85

            if fx < best_y:
                best_x = x.copy()
                best_y = fx

        return best_x, best_y
# EVOLVE-BLOCK-END
