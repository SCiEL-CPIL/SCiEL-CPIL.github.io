"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with time guard for BBOB f6."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280.0

        rng = np.random.default_rng(12345)

        # CMA-ES parameters
        lambda_ = 10
        mu = 5
        w = np.zeros(mu)
        cums = np.sum(w[:mu])
        w[:mu] = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        mueff = 1.0 / np.sum(w**2)
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
        sigma = 2.0
        x = rng.uniform(-3, 3, dim)
        fx = func(x)
        calls = 1
        best_x = x.copy()
        best_fx = fx

        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        C = np.eye(dim)
        step_size = np.sqrt(2) / np.sqrt(dim)

        # Initial phase: random restarts to find a good starting point
        for _ in range(min(20, budget - calls)):
            if time.time() > deadline:
                break
            x_try = rng.uniform(-5, 5, dim)
            fx_try = func(x_try)
            calls += 1
            if fx_try < best_fx:
                best_x, best_fx = x_try, fx_try
                x = x_try
                fx = fx_try

        if calls >= budget:
            return best_x, best_fx

        # CMA-ES main loop
        while calls < budget:
            if time.time() > deadline:
                break

            # Generate population
            ys = np.zeros((lambda_, dim))
            fvals = np.zeros(lambda_)
            for i in range(lambda_):
                if time.time() > deadline:
                    break
                # Sample from multivariate normal using Cholesky
                z = rng.standard_normal(dim)
                y = x + sigma * (C @ z)
                y = np.clip(y, -5, 5)
                ys[i] = y
                fvals[i] = func(y)
                calls += 1

            # Sort by fitness
            idx = np.argsort(fvals)
            ys = ys[idx]
            fvals = fvals[idx]

            # Update best
            if fvals[0] < best_fx:
                best_x = ys[0].copy()
                best_fx = fvals[0]

            # Update mean
            new_x = np.sum(ys[:mu] * w[:, None], axis=0)
            
            # Update path
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mueff) / sigma * (C @ (new_x - x))
            hsig = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs) ** (2 * (calls - budget)))) < (1.4 + 2 / (dim + 1)) * chi_n
            p_c = (1 - cc) * p_c + hsig * np.sqrt(cc * (2 - cc) * mueff) * (new_x - x) / sigma

            # Update covariance
            delta_hsig = (1 - hsig) * cc * (2 - cc)
            C = (1 - c1 - cmu + delta_hsig) * C \
                + c1 * (np.outer(p_c + cc * (x - new_x), p_c + cc * (x - new_x)) - C) \
                + cmu * np.sum(w[i] * np.outer(ys[i] - x, ys[i] - x) for i in range(mu))

            x = new_x
            fx = fvals[0]

            # Adapt sigma
            sigma *= np.exp(cs / damps * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 100.0)

            # If stuck, try a restart
            if sigma < 1e-8 or np.linalg.norm(ys[:mu] - x) < 1e-10:
                x = np.clip(best_x + 1.0 * rng.standard_normal(dim), -5, 5)
                fx = func(x)
                calls += 1
                if fx < best_fx:
                    best_x, best_fx = x.copy(), fx
                sigma = 1.0
                p_sigma = np.zeros(dim)
                p_c = np.zeros(dim)
                C = np.eye(dim)

        return best_x, best_fx
# EVOLVE-BLOCK-END
