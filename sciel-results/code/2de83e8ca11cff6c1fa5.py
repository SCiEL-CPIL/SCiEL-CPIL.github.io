"""Surrogate-assisted derivative-free optimizer for BBOB f11 (Discus) using GP + EI."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


class Optimizer:
    """
    Gaussian Process surrogate-assisted search with Expected Improvement acquisition.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0

    def _count_call(self, func, x):
        self.calls += 1
        return func(x)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        self.calls = 0
        rng = np.random.default_rng(42)

        # Phase 1: Initial random sampling
        n_init = min(20, budget)
        X = rng.uniform(-5, 5, size=(n_init, dim))
        F = np.array([self._count_call(func, x) for x in X])

        best_idx = np.argmin(F)
        x_best = X[best_idx].copy()
        f_best = F[best_idx]

        # Normalize F for GP (helps with numerical stability)
        f_mean = np.mean(F)
        f_std = np.std(F) + 1e-10
        F_norm = (F - f_mean) / f_std

        def gp_predict(X_new, X_train, Y_train, length_scale=1.0, noise=1e-6):
            """GP prediction: returns mean and variance arrays."""
            n_train = len(X_train)
            n_new = len(X_new)
            # RBF kernel
            def rbf_kernel(A, B, ls):
                # A: (n, d), B: (m, d)
                sq_dist = np.sum(A**2, axis=1)[:, None] + np.sum(B**2, axis=1)[None, :] - 2 * A @ B.T
                sq_dist = np.maximum(sq_dist, 0)
                return np.exp(-sq_dist / (2 * ls**2))

            K = rbf_kernel(X_train, X_train, length_scale) + noise * np.eye(n_train)
            K_s = rbf_kernel(X_new, X_train, length_scale)

            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                K += 1e-8 * np.eye(n_train)
                L = np.linalg.cholesky(K)

            alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_train))
            mu = K_s @ alpha
            v = np.linalg.solve(L, K_s.T)
            var = length_scale**2 - np.sum(v**2, axis=0)
            var = np.maximum(var, 1e-10)
            return mu, var

        def expected_improvement(x, mu, var, f_best_norm):
            """EI acquisition function (minimization)."""
            sigma = np.sqrt(var)
            if sigma < 1e-10:
                return 0.0
            z = (f_best_norm - mu) / sigma
            ei = (f_best_norm - mu) * norm.cdf(z) + sigma * norm.pdf(z)
            return ei

        def acquisition_func(x, predict_func, f_best_norm):
            """Negated EI for minimization."""
            mu, var = predict_func(x.reshape(1, -1))
            return -expected_improvement(x, mu[0], var[0], f_best_norm)

        remaining = budget - self.calls
        n_iter = 0
        length_scale = 1.0

        while remaining > 0 and n_iter < 400:
            # Update normalization
            f_mean = np.mean(F)
            f_std = np.std(F) + 1e-10
            F_norm = (F - f_mean) / f_std
            f_best_norm = (f_best - f_mean) / f_std

            # Try to update length scale via simple grid search on log scale
            # (lightweight hyperparameter tuning)
            best_ls = length_scale
            best_neg_ei = np.inf
            if len(X) >= 10:
                ls_candidates = np.logspace(-1.5, 1.5, 7)
                for ls in ls_candidates:
                    try:
                        mu, var = gp_predict(X, X, F_norm, length_scale=ls, noise=1e-6)
                        # Use negative log-likelihood proxy: sum of -0.5*log(var) - 0.5*(Y-mu)^2/var
                        nll = np.sum(0.5 * np.log(var) + 0.5 * (F_norm - mu)**2 / var)
                        if nll < best_neg_ei:
                            best_neg_ei = nll
                            best_ls = ls
                    except Exception:
                        continue
                length_scale = best_ls

            # Generate candidate via EI maximization from multiple starting points
            predict_func = lambda x: gp_predict(x, X, F_norm, length_scale=length_scale, noise=1e-6)

            # Starting points: best point + a few random points
            start_points = [x_best.copy()]
            n_starts = min(5, remaining - 1)
            for _ in range(n_starts):
                start_points.append(rng.uniform(-5, 5, dim))

            best_x_cand = None
            best_acq_val = -np.inf

            for sp in start_points:
                try:
                    res = minimize(
                        acquisition_func,
                        sp,
                        args=(predict_func, f_best_norm),
                        method='Nelder-Mead',
                        bounds=[(-5, 5)] * dim,
                        options={'maxiter': 25, 'xatol': 1e-4, 'fatol': 1e-8}
                    )
                    if res.x is not None:
                        x_cand = np.clip(res.x, -5, 5)
                        # Check distance to existing points
                        dists = np.linalg.norm(X - x_cand, axis=1)
                        if len(dists) > 0 and np.min(dists) < 1e-4:
                            continue
                        acq_val = -acquisition_func(x_cand, predict_func, f_best_norm)
                        if acq_val > best_acq_val:
                            best_acq_val = acq_val
                            best_x_cand = x_cand
                except Exception:
                    continue

            if best_x_cand is None:
                # Fallback: random point
                best_x_cand = rng.uniform(-5, 5, dim)

            # Evaluate
            f_cand = self._count_call(func, best_x_cand)

            # Update model
            X = np.vstack([X, best_x_cand.reshape(1, -1)])
            F = np.append(F, f_cand)

            if f_cand < f_best:
                f_best = f_cand
                x_best = best_x_cand.copy()

            remaining = budget - self.calls
            n_iter += 1

        return x_best, f_best


# EVOLVE-BLOCK-END
