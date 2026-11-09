"""Optimizer for BBOB f17 (Schaffers F7) using CMA-ES with proper covariance adaptation."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer with proper covariance matrix adaptation."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # safety margin before 300s limit

        # CMA-ES parameters
        lambda_ = max(4, int(4 + 3 * np.log(dim)))  # population size
        mu = lambda_ // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        # Step size adaptation
        c_sigma = (mu_eff + 2) / ((dim + 1) ** 2 + mu_eff)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma

        # Covariance adaptation
        c_c = (4 + mu_eff / dim) / ((dim + 4) ** 2 + mu_eff)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        # Initial state
        rng = np.random.default_rng(12345)
        x_mean = rng.uniform(-3, 3, dim)
        sigma = 2.0
        C = np.eye(dim)
        C_inv_sqrt = np.eye(dim)  # We'll use Cholesky for sampling
        C_cholesky = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 + 1 / (4 * dim) - 1 / (21 * dim ** 2))

        # Sort indices
        indices = np.argsort(weights)[::-1]

        # Initial population
        best_x = None
        best_f = np.inf
        calls = 0

        for generation in range(budget // lambda_ + 1):
            if time.time() > deadline:
                break

            # Generate population
            pop = np.zeros((lambda_, dim))
            for i in range(lambda_):
                z = rng.standard_normal(dim)
                pop[i] = x_mean + sigma * (C_cholesky @ z)
                pop[i] = np.clip(pop[i], -5, 5)

            # Evaluate population
            fvals = np.zeros(lambda_)
            for i in range(lambda_):
                if calls >= budget:
                    break
                fvals[i] = func(pop[i])
                calls += 1
                if fvals[i] < best_f:
                    best_f = fvals[i]
                    best_x = pop[i].copy()

            if calls >= budget:
                break

            # Sort by fitness
            order = np.argsort(fvals)
            pop_sorted = pop[order]
            fvals_sorted = fvals[order]

            # Update best
            if fvals_sorted[0] < best_f:
                best_f = fvals_sorted[0]
                best_x = pop_sorted[0].copy()

            # Weighted mean of best mu
            x_new = np.sum(weights[:, np.newaxis] * pop_sorted[:mu], axis=0)

            # Update evolution path for sigma
            y = (x_new - x_mean) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y

            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 1e6)

            # Update evolution path for covariance
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * y

            # Update covariance matrix
            delta_heav = 1 + (1 - c_c) ** 2 * (1 - c_c) / c_c if False else 1 + (1 - c_c) * (1 - c_c) * (1 - c_c) / c_c
            # Proper delta_heav
            delta_heav = 1 + (1 - c_c) * (1 - c_c) / c_c

            C = (1 + c_c * (2 - c_c) - c_1 - c_mu) * C \
                + c_1 * (np.outer(p_c, p_c) - C) \
                + c_mu * np.sum(weights[:, np.newaxis, np.newaxis] * np.outer(pop_sorted[:mu] - x_mean, (pop_sorted[:mu] - x_mean).T), axis=0) / sigma**2

            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2

            # Compute Cholesky factorization
            try:
                C_cholesky = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Add small regularization
                C += 1e-8 * np.eye(dim)
                C = (C + C.T) / 2
                try:
                    C_cholesky = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C_cholesky = np.eye(dim)

            # Update mean
            x_mean = x_new
            x_mean = np.clip(x_mean, -5, 5)

            # Check convergence
            if sigma < 1e-12:
                break

        # Final polish with Nelder-Mead if we have budget left
        if calls < budget and best_x is not None:
            remaining = budget - calls
            if remaining > 10:
                best_x = self._nelder_mead_polish(func, best_x, best_f, remaining, deadline)
                calls = budget  # approximate

        return best_x, best_f

    def _nelder_mead_polish(self, func, x0, f0, budget, deadline):
        """Nelder-Mead polish starting from best point."""
        dim = self.dim
        calls = 0
        eps = 1e-4

        # Build initial simplex
        simplex = np.zeros((dim + 1, dim))
        fvals = np.zeros(dim + 1)
        simplex[0] = x0.copy()
        fvals[0] = f0
        calls += 1

        for i in range(dim):
            simplex[i + 1] = x0.copy()
            simplex[i + 1][i] += eps
            simplex[i + 1] = np.clip(simplex[i + 1], -5, 5)
            fvals[i + 1] = func(simplex[i + 1])
            calls += 1

        alpha, gamma, rho = 1.0, 2.0, 0.5

        while calls < budget and time.time() < deadline:
            # Sort simplex
            order = np.argsort(fvals)
            simplex = simplex[order]
            fvals = fvals[order]

            # Centroid of best points
            x_centroid = np.mean(simplex[:dim], axis=0)

            # Reflection
            x_r = x_centroid + alpha * (x_centroid - simplex[-1])
            x_r = np.clip(x_r, -5, 5)
            f_r = func(x_r)
            calls += 1

            if f_r < fvals[0]:
                # Expansion
                x_e = x_centroid + gamma * (x_r - x_centroid)
                x_e = np.clip(x_e, -5, 5)
                f_e = func(x_e)
                calls += 1
                if f_e < f_r:
                    simplex[-1] = x_e
                    fvals[-1] = f_e
                else:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
            elif f_r < fvals[-2]:
                simplex[-1] = x_r
                fvals[-1] = f_r
            else:
                # Contraction
                x_c = x_centroid + rho * (simplex[-1] - x_centroid)
                x_c = np.clip(x_c, -5, 5)
                f_c = func(x_c)
                calls += 1
                if f_c < fvals[-1]:
                    simplex[-1] = x_c
                    fvals[-1] = f_c
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        fvals[i] = func(simplex[i])
                        calls += 1
                        if calls >= budget:
                            break
                    if calls >= budget:
                        break

            # Check convergence
            if np.max(fvals) - np.min(fvals) < 1e-12:
                break

        # Return best point
        idx = np.argmin(fvals)
        return simplex[idx], fvals[idx]
# EVOLVE-BLOCK-END
