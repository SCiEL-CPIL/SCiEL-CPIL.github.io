"""Surrogate-assisted derivative-free optimizer for BBOB f11 (Discus)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.stats import qmc


class Optimizer:
    """Bayesian Optimization with GP surrogate for low-dimensional continuous optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Initial design: Latin hypercube + center + random
        n_init = min(20, budget // 2)
        if n_init < 5:
            n_init = 5

        # Latin hypercube sampling
        sampler = qmc.LatinHypercube(d=dim, seed=42)
        lhc_samples = sampler.random(n=n_init)
        # Map [0,1] to [-5,5]
        X_init = lhc_samples * 10.0 - 5.0

        # Add center point
        X_init = np.vstack([X_init, np.zeros(dim)])

        # Evaluate initial points
        X = X_init
        Y = np.array([func(x) for x in X])

        best_x = X[np.argmin(Y)]
        best_y = Y.min()

        # GP kernel: Matern with length scales + white kernel for noise
        length_scale = np.ones(dim) * 2.0
        kernel = ConstantKernel(1.0) * Matern(length_scale=length_scale, nu=2.5) + WhiteKernel(noise_level=1e-6)

        n_evals = len(Y)

        # Main loop
        while n_evals < budget:
            # Train GP
            gp = GaussianProcessRegressor(
                kernel=kernel,
                alpha=1e-10,
                normalize_y=True,
                n_restarts_optimizer=5,
                random_state=42
            )
            gp.fit(X, Y)

            # Generate candidate points for acquisition
            # Use a grid of random points
            n_candidates = 2000
            X_cand = rng.uniform(-5, 5, size=(n_candidates, dim))

            # Predict mean and std
            mu, sigma = gp.predict(X_cand, return_std=True)

            # Expected Improvement acquisition
            # EI(x) = (mu_best - mu(x)) * Phi(z) + sigma(x) * phi(z)
            # where z = (mu_best - mu(x)) / sigma(x)
            from scipy.stats import norm

            # Avoid division by zero
            sigma_safe = np.where(sigma < 1e-10, 1e-10, sigma)
            z = (best_y - mu) / sigma_safe
            EI = (best_y - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
            EI = np.where(sigma < 1e-10, 0.0, EI)

            # Select best candidate
            idx = np.argmax(EI)
            x_new = X_cand[idx]

            # Evaluate
            y_new = func(x_new)
            n_evals += 1

            # Update best
            if y_new < best_y:
                best_y = y_new
                best_x = x_new.copy()

            # Add to dataset
            X = np.vstack([X, x_new])
            Y = np.append(Y, y_new)

            # Periodically run a local refinement using scipy L-BFGS-B
            # from the best point found so far
            if n_evals % 50 == 0 and n_evals < budget - 10:
                try:
                    # Use a simple numerical gradient approach via finite differences
                    # but since we can't compute gradients of the black box easily,
                    # we'll do a few Nelder-Mead steps around best_x
                    from scipy.optimize import minimize as sp_minimize

                    def wrapped_func(x):
                        return func(np.clip(x, -5, 5))

                    res = sp_minimize(
                        wrapped_func,
                        best_x,
                        method='Nelder-Mead',
                        options={'maxiter': 20, 'xatol': 1e-4, 'fatol': 1e-6}
                    )
                    if res.fun < best_y:
                        best_y = res.fun
                        best_x = np.clip(res.x, -5, 5)
                        # Add this point to dataset
                        X = np.vstack([X, best_x])
                        Y = np.append(Y, best_y)
                        n_evals += 1
                        if n_evals >= budget:
                            break
                except Exception:
                    pass

        return best_x, best_y
# EVOLVE-BLOCK-END
