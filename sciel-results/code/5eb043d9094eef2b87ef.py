"""Optimizer for BBOB-style f15 (Rastrigin rotated) using CMA-ES."""
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer with large population for global diversity."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Time guard: 330s limit, use 320s as safety margin
        start_time = time.time()
        time_limit = 320.0

        # CMA-ES parameters
        # lambda = 10 for dim=5, but we can use a larger population
        # Standard: lambda = 4 + floor(3*log(dim))
        lam = max(10, 4 + 3 * int(np.log(dim)))
        mu = lam // 2  # number of parents

        # Initial population
        x0 = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)
        ps = np.zeros(dim)
        pc = np.zeros(dim)

        # CMA-ES parameters
        cmu = 2.0 / ((dim + 1.7) ** 2 + lam)
        cc = (4 + dim / lam) / ((dim + 2) ** 2 + lam)
        cs = (1 + 2 * max(0, np.sqrt((mu - 1) / (dim + 1)) - 1)) / ((dim + 2) + lam)
        c1 = (2 - cmu + cc) / ((dim + 1.3) ** 2 + lam)
        cmu = 2 * (mu - 1 + 1.0 / mu) / ((dim + 2) ** 2 + lam)
        dsigma = 1 + 2 * max(0, np.sqrt((mu - 1) / (dim + 1)) - 1) + cc

        # Cholesky of C
        cholC = np.linalg.cholesky(C)

        # Initial population evaluation
        best_x = None
        best_f = np.inf
        calls = 0

        # Generate initial population
        pop = []
        for i in range(lam):
            xi = x0 + sigma * (cholC @ rng.standard_normal(dim))
            xi = np.clip(xi, -5, 5)
            fi = func(xi)
            calls += 1
            pop.append((fi, xi))
            if fi < best_f:
                best_f = fi
                best_x = xi.copy()

        # Sort by fitness
        pop.sort(key=lambda t: t[0])

        # CMA-ES loop
        while calls < budget:
            # Check time
            if time.time() - start_time > time_limit:
                break

            # Update mean
            w = np.zeros(lam)
            w[:mu] = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w /= w.sum()
            mueff = 1.0 / np.sum(w ** 2)

            xmean_old = x0.copy()
            x0 = np.sum([w[i] * pop[i][1] for i in range(mu)], axis=0)

            # Update step size control variables
            # ps update
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (cholC.T @ (x0 - xmean_old)) / sigma
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (calls // lam + 1))) <
                    (1.4 + 2 / (dim + 1)) * np.trace(C) / dim)

            # pc update
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (x0 - xmean_old) / sigma

            # Update covariance matrix
            rank1 = np.outer(pc, pc)
            delta_h = (1.0 if hsig else 0.0) * (2 - cc)
            rankmu = np.zeros((dim, dim))
            for i in range(mu):
                zi = (pop[i][1] - xmean_old) / sigma
                rankmu += w[i] * np.outer(zi, zi)
            C = (1 + c1 * delta_h - c1 - cmu) * C + c1 * rank1 + cmu * rankmu

            # Symmetrize and ensure positive definite
            C = (C + C.T) / 2.0
            # Add small regularization if needed
            try:
                cholC = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C += 1e-4 * np.eye(dim)
                C = (C + C.T) / 2.0
                cholC = np.linalg.cholesky(C)

            # Update sigma
            sigma *= np.exp(cs / dsigma * (np.linalg.norm(ps) / np.sqrt(2 * dim) - 1))

            # Generate new population
            pop = []
            for i in range(lam):
                if calls >= budget:
                    break
                # Check time
                if time.time() - start_time > time_limit:
                    break
                xi = x0 + sigma * (cholC @ rng.standard_normal(dim))
                xi = np.clip(xi, -5, 5)
                fi = func(xi)
                calls += 1
                pop.append((fi, xi))
                if fi < best_f:
                    best_f = fi
                    best_x = xi.copy()

            if calls >= budget or time.time() - start_time > time_limit:
                break

            # Sort by fitness
            pop.sort(key=lambda t: t[0])

        return best_x, best_f
