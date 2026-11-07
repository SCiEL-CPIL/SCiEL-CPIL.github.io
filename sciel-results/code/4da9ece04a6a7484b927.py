"""Bayesian-Optimization style surrogate-assisted optimizer for Rosenbrock (BBOB f8)."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel


class Optimizer:
    """Surrogate-assisted global optimizer with local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        self.calls = 0
        dim = self.dim
        budget = self.budget

        # Internal function wrapper with call counting and deadline check
        def wrapped(x):
            nonlocal self
            if self.calls >= budget:
                return 1e10
            if time.time() > deadline:
                return 1e10
            x = np.asarray(x, dtype=float).ravel()[:dim]
            x = np.clip(x, -5.0, 5.0)
            val = float(func(x))
            self.calls += 1
            return val

        # Initialize with a small random set
        rng = np.random.default_rng(42)
        X = []
        Y = []

        # Initial random points
        n_init = 10
        for _ in range(n_init):
            x = rng.uniform(-5, 5, dim)
            fx = wrapped(x)
            X.append(x)
            Y.append(fx)

        X = np.array(X)
        Y = np.array(Y)

        best_x = X[np.argmin(Y)]
        best_y = Y.min()

        # Kernel: Matern with length scales + white kernel for noise
        length_scale = np.ones(dim) * 1.0
        kernel = ConstantKernel(1.0) * Matern(nu=2.5, length_scale=length_scale, length_scale_bounds=(0.1, 10.0)) + \
                   WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-8, 1e-2))

        # GP-based acquisition loop
        max_gp_iter = 40
        gp_update_interval = 5  # re-fit GP every this many evaluations

        for gp_iter in range(max_gp_iter):
            if self.calls >= budget or time.time() > deadline:
                break

            # Fit GP on current data
            try:
                gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=1,
                                               normalize_y=True, alpha=1e-8)
                gp.fit(X, Y)
            except Exception:
                # Fallback: use current best and do local search
                break

            # Acquisition: Expected Improvement
            # We'll evaluate candidate points from a grid + random + local around best
            candidates = []

            # 1. Random candidates
            for _ in range(20):
                candidates.append(rng.uniform(-5, 5, dim))

            # 2. Points around current best
            for _ in range(10):
                candidates.append(np.clip(best_x + rng.normal(0, 0.5, dim), -5, 5))

            # 3. Grid-like candidates (coarse)
            for i in range(5):
                for j in range(5):
                    if dim == 5:
                        c = np.array([
                            -5 + 2.5 * i / 4,
                            -5 + 2.5 * j / 4,
                            rng.uniform(-5, 5),
                            rng.uniform(-5, 5),
                            rng.uniform(-5, 5)
                        ])
                    else:
                        c = rng.uniform(-5, 5, dim)
                    candidates.append(c)

            candidates = np.array(candidates)

            # Compute EI for each candidate
            mu, sigma = gp.predict(candidates, return_std=True)
            sigma = np.maximum(sigma, 1e-8)
            best_y_gp = Y.min()
            sigma_f = (best_y_gp - mu) / sigma
            from scipy.stats import norm
            EI = (best_y_gp - mu) * norm.cdf(sigma_f) + sigma * norm.pdf(sigma_f)
            EI = np.maximum(EI, 0)

            # Select top candidates
            top_k = 5
            top_idx = np.argsort(-EI)[:top_k]

            # Evaluate selected candidates
            for idx in top_idx:
                if self.calls >= budget or time.time() > deadline:
                    break
                x = np.clip(candidates[idx], -5, 5)
                fx = wrapped(x)
                X = np.vstack([X, x])
                Y = np.append(Y, fx)
                if fx < best_y:
                    best_y = fx
                    best_x = x.copy()

            # Periodically do a local refinement from the best point
            if gp_iter % 5 == 4:
                if self.calls < budget and time.time() < deadline:
                    try:
                        res = minimize(
                            lambda x: wrapped(x),
                            best_x,
                            method='L-BFGS-B',
                            bounds=[(-5, 5)] * dim,
                            options={'maxiter': 20, 'maxfun': 50}
                        )
                        if res.fun < best_y:
                            best_y = res.fun
                            best_x = np.clip(res.x, -5, 5)
                    except Exception:
                        pass

        # Final local refinement if budget remains
        if self.calls < budget and time.time() < deadline:
            try:
                res = minimize(
                    lambda x: wrapped(x),
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 50, 'maxfun': 100}
                )
                if res.fun < best_y:
                    best_y = res.fun
                    best_x = np.clip(res.x, -5, 5)
            except Exception:
                pass

        return best_x, best_y
# EVOLVE-BLOCK-END
