"""Optimizer for BBOB-style f17 (Schaffers F7) with hidden shift/rotation/offset."""
import numpy as np
import time


class Optimizer:
    """CMA-ES with deadline guard and adaptive restart."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # safety margin before 330s limit
        dim = self.dim
        budget = self.budget

        # CMA-ES parameters
        # lambda = 4 (2*dim+1 = 11 is standard but 4 is faster for dim=5)
        # For dim=5, let's use lambda=4 or 6
        lambda_es = 4  # population size

        # Standard CMA-ES parameters
        mu = lambda_es // 2  # number of weighted individuals
        # Weights: proportional weights
        w = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mueff = 1.0 / np.sum(w ** 2)
        c_sigma = (mueff + 2) / ((dim + 1.3) ** 2 + mueff)
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, mueff / (dim + 2) * (0.5 + min(1, (mueff - 2) / (dim + mueff))))
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma

        # Initial sigma
        sigma = 1.0

        # Phase 1: Broad exploration with random restarts
        # Phase 2: CMA-ES refinement

        best_x = None
        best_f = np.inf
        calls_made = 0

        # Try multiple random starting points with CMA-ES
        rng = np.random.default_rng(42)

        # First, do a quick random search to find a good starting point
        n_random = 20
        for i in range(n_random):
            x = rng.uniform(-4, 4, dim)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            if calls_made >= budget or time.time() > deadline:
                break

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls_made += 1

        # Now run CMA-ES from the best point found
        remaining = budget - calls_made
        if remaining > 20:
            best_x, best_f = self._run_cmaes(
                func, best_x, best_f, sigma, lambda_es,
                w, mueff, c_sigma, c_c, c_1, c_mu, d_sigma,
                remaining, deadline, calls_made
            )

        return best_x, best_f

    def _run_cmaes(self, func, x0, f0, sigma, lambda_es, w, mueff,
                   c_sigma, c_c, c_1, c_mu, d_sigma,
                   budget, deadline, calls_made):
        dim = self.dim
        rng = np.random.default_rng(123)

        # State
        xmean = x0.copy()
        sigma = sigma
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        best_x = x0.copy()
        best_f = f0
        calls = calls_made

        # Precompute
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        for gen in range(budget // lambda_es + 1):
            if time.time() > deadline or calls >= budget:
                break

            # Generate population
            D = np.linalg.cholesky(C)
            B = D.T  # For sampling
            # Actually, C = B * B.T, so sample as xmean + sigma * B @ z
            # Let's use: z ~ N(0, I), y = B @ z, x = xmean + sigma * y

            population = np.zeros((lambda_es, dim))
            fvals = np.zeros(lambda_es)

            for i in range(lambda_es):
                if time.time() > deadline or calls >= budget:
                    break
                z = rng.standard_normal(dim)
                y = D @ z
                x = np.clip(xmean + sigma * y, -5, 5)
                population[i] = x
                fvals[i] = func(x)
                calls += 1
                if fvals[i] < best_f:
                    best_f = fvals[i]
                    best_x = x.copy()

            # Sort by fitness
            order = np.argsort(fvals)
            x_sorted = population[order]
            f_sorted = fvals[order]

            # Update mean
            xmean_new = np.sum(w[:, np.newaxis] * x_sorted[:mu], axis=0)
            xmean_new = np.clip(xmean_new, -5, 5)

            # Update step-size control
            ys = (D @ (xmean_new - xmean)) / sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * ys
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            sigma = max(sigma, 1e-8)
            sigma = min(sigma, 10.0)

            # Update evolution path
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mueff) * (xmean_new - xmean) / sigma

            # Update covariance
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1) / lambda_es))) < (1.4 + 2 / (dim + 1)) * chi_n
            delta_hsig = 1.0 if hsig else 0.0

            outer = np.outer(pc, pc)
            C = (1 + c_1 * delta_hsig - c_1 - c_mu) * C
            C += c_1 * (outer - (1 - delta_hsig) * C)
            C += c_mu * np.sum(w[:, np.newaxis, np.newaxis] * np.outer(
                (x_sorted[:mu] - xmean) / sigma,
                (x_sorted[:mu] - xmean) / sigma
            ), axis=0)

            # Ensure symmetry
            C = (C + C.T) / 2

            # Regularize if needed
            eigvals = np.linalg.eigvalsh(C)
            if eigvals.min() < 1e-12:
                C += 1e-12 * np.eye(dim)

            xmean = xmean_new

            # Termination check
            if np.linalg.norm(xmean - best_x) < 1e-7 and gen > 10:
                break

        return best_x, best_f
