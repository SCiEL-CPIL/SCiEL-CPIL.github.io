"""Rewritten optimizer: CMA-ES style population-based search with deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES variant for the BBOB f5 (linear slope) task."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280  # safety margin under 300s limit

        # CMA-ES parameters
        lam = 4 * int(1 + 3 * dim)  # population size
        # Weights
        mu = lam // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Initialize
        rng = np.random.default_rng(42)
        mean = np.zeros(dim)
        sigma = 2.0
        B = np.eye(dim)
        D = np.ones(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)

        # First evaluation
        mean = rng.uniform(-5, 5, dim)
        best_x = mean.copy()
        best_f = func(mean)
        calls = 1

        # Generate initial population
        def generate_population():
            nonlocal B, D, sigma
            z = rng.standard_normal((lam, dim))
            # Transform using B @ diag(D)
            x = mean + sigma * (z @ (B * D).T)
            return x, z

        pop, z = generate_population()
        fvals = np.array([func(x) for x in pop])
        calls += lam

        # Track best
        idx_best = np.argmin(fvals)
        if fvals[idx_best] < best_f:
            best_f = fvals[idx_best]
            best_x = pop[idx_best].copy()

        # Sorting indices
        order = np.argsort(fvals)

        # Main loop
        while calls < budget:
            if time.time() > deadline:
                break

            # Update mean
            w = weights[:mu]
            y_w = (pop[order[:mu]] - mean) / sigma
            new_mean = mean + np.sum(w[:, None] * y_w, axis=0)

            # Update step size control
            y_sigma = (B @ (D * ((B.T @ (new_mean - mean)) / sigma)))
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma / chi_n
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls + 1)))) < (1.4 + 2 / (dim + 1)) * chi_n

            # Update covariance matrix
            pc = (1 - c_c) * pc + hsig * c_c * y_w
            art = pc @ pc.T
            art[np.diag_indices(dim)] += c_c
            C = (1 - c_1 - c_mu) * C + c_1 * (art + (c_c * (2 - c_c) - 1) * np.eye(dim)) + \
                c_mu * np.sum(w[:, None, None] * np.outer(y_w, y_w), axis=0)

            # Eigen-decomposition (stabilize)
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-20, None)
                D = np.sqrt(eigvals)
                B = eigvecs
            except np.linalg.LinAlgError:
                pass  # keep old B, D

            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 100.0)
            mean = new_mean

            # Generate next population
            pop, z = generate_population()

            # Evaluate
            for i in range(lam):
                if calls >= budget or time.time() > deadline:
                    break
                fvals[i] = func(pop[i])
                calls += 1
                if fvals[i] < best_f:
                    best_f = fvals[i]
                    best_x = pop[i].copy()

            # Update order
            order = np.argsort(fvals)

        return best_x, best_f
# EVOLVE-BLOCK-END
