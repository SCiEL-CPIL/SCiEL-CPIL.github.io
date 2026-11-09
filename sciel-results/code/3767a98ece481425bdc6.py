"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Population-based CMA-ES with deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 275.0  # safety margin

        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(0)

        # CMA-ES parameters
        lam = max(4, int(4 + 3 * np.log(n)))
        c_sigma = 1.0 / (4 * n + 4)
        c_c = (4 + n / 4) / (n + 4.3)
        c_1 = 2 / ((n + 1.3) ** 2 + n)
        c_mu = min(1 - c_1, 2 * ((int(0.5 * lam) + 0.5) / (n + 2) - 1))

        # Weights
        weights = np.log(lam / 2 + 0.5) - np.log(np.arange(1, lam + 1))
        weights = np.maximum(weights, 0)
        if np.sum(weights) == 0:
            weights = np.ones(lam) / lam
        else:
            weights /= np.sum(weights)
        mu = int(np.sum(weights > 0))
        mu_eff = 1.0 / np.sum(weights[:mu] ** 2)

        # Initialize
        x_mean = np.zeros(n)
        sigma = 2.0
        C = np.eye(n)
        B = np.eye(n)
        D = np.ones(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        # Helper to sample
        def sample():
            z = rng.standard_normal(n)
            return x_mean + sigma * (B @ (D * z))

        # Evaluate initial population
        pops = []
        for i in range(lam):
            x = sample()
            x = np.clip(x, -5, 5)
            f = func(x)
            pops.append((f, x))

        pops.sort(key=lambda t: t[0])
        best_x = pops[0][1].copy()
        best_f = pops[0][0]
        calls_made = lam

        # Main loop
        for gen in range(1, 1000):
            if calls_made >= budget:
                break
            if time.time() > deadline - 5.0:
                break

            # Update mean
            new_mean = np.zeros(n)
            for i in range(mu):
                new_mean += weights[i] * pops[i][1]

            # Update step size
            diff = (new_mean - x_mean) / sigma
            old_norm = np.linalg.norm(diff)
            if old_norm > 1e-12:
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (B @ (D * diff / old_norm))
            hs = np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1))) / chi_n
            if hs < 1.4 + 2 / (n + 1):
                sigma *= np.exp(c_sigma * (hs / chi_n - 1))

            # Update covariance
            pc = (1 - c_c) * pc + c_c * diff
            rank_one = np.outer(pc, pc)
            rank_mu = np.zeros((n, n))
            for i in range(mu):
                di = (pops[i][1] - x_mean) / sigma
                rank_mu += (weights[i] / c_1) * np.outer(B @ (D * di), B @ (D * di))
            delta_hs = (1 - c_c) ** 2 * (np.linalg.norm(pc) ** 2 / n - 1)
            C = (1 + c_1 * (c_mu - c_1) + c_mu) * C - c_1 * rank_one - c_1 * c_mu * rank_mu
            C += (c_1 * c_mu * delta_hs + (2 * c_mu * c_c - c_1 * c_c) * (np.linalg.norm(pc) ** 2 / n)) * np.eye(n)

            # Symmetrize and re-orthogonalize
            C = (C + C.T) / 2
            try:
                B, D2 = np.linalg.eigh(C)
                D = np.sqrt(np.maximum(D2, 1e-12))
                B = B @ np.diag(1.0 / np.maximum(D, 1e-12))
            except:
                pass

            x_mean = new_mean

            # Generate and evaluate new population
            new_pops = []
            for i in range(lam):
                if calls_made >= budget:
                    break
                if time.time() > deadline - 2.0:
                    break
                x = sample()
                x = np.clip(x, -5, 5)
                f = func(x)
                calls_made += 1
                new_pops.append((f, x))
                if f < best_f:
                    best_f = f
                    best_x = x.copy()

            if not new_pops:
                break
            new_pops.sort(key=lambda t: t[0])
            pops = new_pops

        return best_x, best_f
# EVOLVE-BLOCK-END
