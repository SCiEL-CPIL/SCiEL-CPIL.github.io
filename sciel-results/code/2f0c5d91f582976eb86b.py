"""CMA-ES optimizer for BBOB-style f6 (Attractive sector)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES (Covariance Matrix Adaptation Evolution Strategy)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290  # safety margin before 300s limit

        # Standard CMA-ES parameters
        lam = 4 + int(3 * np.log(1 + 2 * dim))
        mu = lam // 2

        # Weights
        w = np.zeros(mu)
        for i in range(mu):
            w[i] = np.log(mu + 0.5) - np.log(i + 1)
        w /= w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)

        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (dim + 1)) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        chi = np.sqrt(dim * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2)))

        # State
        x = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)

        # Initial evaluation
        fx = func(x)
        best_x = x.copy()
        best_f = fx
        calls = 1

        # Cholesky factor for sampling
        L = np.linalg.cholesky(C)

        while calls < budget:
            if time.time() > deadline:
                break

            # Generate population
            pops = []
            for i in range(lam):
                z = np.random.standard_normal(dim)
                y = x + sigma * (L @ z)
                y = np.clip(y, -5, 5)
                pops.append(y)

            # Evaluate population
            pop_f = np.array([func(y) for y in pops])
            calls += lam

            # Sort by fitness
            order = np.argsort(pop_f)
            selected = [pops[i] for i in order[:mu]]
            selected_f = pop_f[order[:mu]]

            # Update best
            if selected_f[0] < best_f:
                best_f = selected_f[0]
                best_x = selected[0].copy()

            # Mean update
            x_new = np.zeros(dim)
            for i in range(mu):
                x_new += w[i] * selected[i]

            # Update ps
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (L.T @ (x_new - x) / sigma)

            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi - 1))
            sigma = np.clip(sigma, 1e-8, 1e8)

            # Update pc
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls // lam + 1)))) < (1.4 + 2 / (dim + 1)) * chi
            pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_new - x) / sigma
            x = x_new

            # Update C
            delta_hsig = (1 - hsig) * c_c * (2 - c_c)
            C = (1 + c_1 + c_mu - c_1 - delta_hsig) * C
            for i in range(mu):
                y_i = (selected[i] - x) / sigma
                C += c_mu * w[i] * np.outer(y_i, y_i)
            C += delta_hsig * np.outer(pc, pc)

            # Ensure symmetry
            C = (C + C.T) / 2

            # Update Cholesky
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C += 1e-10 * np.eye(dim)
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    L = np.eye(dim)
                    C = np.eye(dim)

        return best_x, best_f
# EVOLVE-BLOCK-END
