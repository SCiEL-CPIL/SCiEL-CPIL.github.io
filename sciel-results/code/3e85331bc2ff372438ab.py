"""Bayesian-Optimization-style optimizer for BBOB f23 (Katsuura)."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Gaussian-process surrogate with EI acquisition + local polishing."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = np.full(dim, -5.0)
        self.ub = np.full(dim, 5.0)

    def _latin_hypercube(self, n, dim, rng):
        pts = np.empty((n, dim))
        for d in range(dim):
            order = rng.permutation(n)
            pts[order, d] = (np.arange(n) + rng.random(n)) / n
        return self.lb + pts * (self.ub - self.lb)

    def _gp_predict(self, X, X_train, y_train, length_scale, sigma_f, noise):
        # Simple squared-exponential kernel with isotropic length scale
        d2 = np.sum(X**2, axis=1)[:, None] - 2 * X @ X_train.T + np.sum(X_train**2, axis=1)[None, :]
        K = np.exp(-d2 / (2 * length_scale**2))
        Kss = K + noise * np.eye(len(X_train))
        # Add small jitter for numerical stability
        Kss += 1e-8 * np.eye(len(X_train))
        try:
            L = np.linalg.cholesky(Kss)
        except np.linalg.LinAlgError:
            Kss += 1e-6 * np.eye(len(X_train))
            L = np.linalg.cholesky(Kss)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
        mu = K @ alpha
        v = np.linalg.solve(L, K.T)
        var = sigma_f**2 - np.sum(v**2, axis=0)
        var = np.maximum(var, 1e-12)
        return mu, np.sqrt(var)

    def _ei(self, mu, sigma, best_f, xi=0.01):
        z = (best_f - xi - mu) / (sigma + 1e-12)
        from scipy.stats import norm
        ei = (best_f - xi - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return ei

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s
        rng = np.random.default_rng(42)

        dim = self.dim
        lb, ub = self.lb, self.ub

        # Initial Latin Hypercube sampling: 25 points
        n_init = 25
        X = self._latin_hypercube(n_init, dim, rng)
        y = np.array([func(x) for x in X])

        # Track best
        idx_best = np.argmin(y)
        x_best = X[idx_best].copy()
        f_best = y[idx_best]

        calls_made = n_init
        budget_remaining = self.budget - calls_made

        # GP hyperparameters (tuned for 5D Katsuura)
        length_scale = 1.5
        sigma_f = 1.0
        noise = 1e-4

        # Bayesian optimization loop with EI acquisition
        # Use a candidate set approach: generate random candidates, evaluate EI, pick best
        n_candidates = 200
        # Reserve some budget for local polishing
        n_bo = min(budget_remaining, 350)
        n_polish = budget_remaining - n_bo

        for i in range(n_bo):
            # Check deadline
            if time.time() > deadline - 30.0:
                break

            # Generate candidate points
            X_cand = rng.uniform(lb, ub, (n_candidates, dim))
            # Also add some points near current best
            X_cand_near = x_best + rng.normal(0, 0.5, (50, dim))
            X_cand_near = np.clip(X_cand_near, lb, ub)
            X_cand = np.vstack([X_cand, X_cand_near])

            # Predict mean and variance
            mu, sigma = self._gp_predict(X_cand, X, y, length_scale, sigma_f, noise)

            # Expected Improvement
            ei_vals = self._ei(mu, sigma, f_best, xi=0.01)

            # Pick candidate with highest EI
            idx = np.argmax(ei_vals)
            x_new = X_cand[idx].copy()

            # Evaluate
            f_new = func(x_new)
            calls_made += 1

            # Update GP data
            X = np.vstack([X, x_new[None, :]])
            y = np.append(y, f_new)

            # Update best
            if f_new < f_best:
                f_best = f_new
                x_best = x_new.copy()
                # Occasionally re-estimate sigma_f to track scale
                if len(y) % 20 == 0:
                    sigma_f = max(np.std(y), 0.1)

            # Adaptive length scale: shrink as we find better solutions
            if i % 30 == 0 and i > 0:
                length_scale = max(length_scale * 0.9, 0.3)

        # Local polishing with Nelder-Mead from best point
        if calls_made < self.budget and time.time() < deadline - 10.0:
            n_polish_steps = min(self.budget - calls_made, 50)
            for _ in range(n_polish_steps):
                if time.time() > deadline - 5.0:
                    break
                try:
                    res = minimize(
                        lambda x: func(np.clip(x, lb, ub)),
                        x_best,
                        method='Nelder-Mead',
                        options={'maxiter': 30, 'xatol': 1e-6, 'fatol': 1e-8},
                    )
                    if res.fun < f_best:
                        f_best = res.fun
                        x_best = np.clip(res.x, lb, ub).copy()
                except Exception:
                    break

        return x_best, f_best
# EVOLVE-BLOCK-END
