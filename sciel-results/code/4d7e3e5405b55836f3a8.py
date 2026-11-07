"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for BBOB f7 (Step ellipsoidal)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s kill

        d = self.dim
        budget = self.budget

        # Count function calls
        calls = 0

        def f(x):
            nonlocal calls
            calls += 1
            return func(np.asarray(x, dtype=float))

        # CMA-ES parameters
        lam = max(4, int(round((4 + int(np.log(d))) * d)))  # population size
        mu = lam // 2  # recombination weight
        # Weights for recombination
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)
        # Step-size adaptation parameters
        cc = (4 + mueff / d) / (d + 4 + 2 * mueff / d)
        cs = (mueff + 2) / (d + mueff + 5)
        # Covariance matrix adaptation
        c1 = (2 - 1 / (d + 1.8) + 2 * min(1, (mueff - 2 + 1 / d) / (d + 3))) / (
            (d + 1.8) ** 2 + 2
        )
        cmu = min(
            1 - c1,
            2 * (mueff - 2 + 1 / d) / ((d + 2) ** 2 + mueff),
        )
        # Damping for step-size adaptation
        dsigma = 1.0 + 2 * max(0, np.sqrt((mueff - 1) / (d + 1)) - 1) + cs
        chi_n = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d ** 2))

        # Initialize
        rng = np.random.default_rng(42)
        # Start near center with some randomization
        xmean = np.zeros(d)
        sigma = 2.0

        # Evolution paths
        ps = np.zeros(d)
        pc = np.zeros(d)

        # Covariance matrix
        C = np.eye(d)
        Cinv = np.eye(d)
        # Precompute C^1/2 and Cinv^1/2
        eigvals, eigvecs = np.linalg.eigh(C)
        Cinv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
        C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

        # Evaluate initial point
        best_x = xmean.copy()
        best_f = f(best_x)

        # Main loop
        while calls < budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Generate population
            y = np.array([C_sqrt @ (sigma * rng.standard_normal(d)) for _ in range(lam)])
            x_pop = xmean[None, :] + y
            # Clip to box
            x_pop = np.clip(x_pop, -5, 5)

            # Evaluate population
            f_vals = np.array([f(x_pop[i]) for i in range(lam)])

            # Update best
            idx_best = np.argmin(f_vals)
            if f_vals[idx_best] < best_f:
                best_f = f_vals[idx_best]
                best_x = x_pop[idx_best].copy()

            # Sort by fitness
            order = np.argsort(f_vals)
            x_sorted = x_pop[order]
            f_sorted = f_vals[order]

            # Recombination
            xnew = sum(weights[i] * x_sorted[i] for i in range(mu))

            # Update evolution paths
            yinv = Cinv_sqrt @ (xnew - xmean) / sigma
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * yinv
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (calls + 1) / (d * cs)))) < (1.4 + 2 / (d + 1)) * chi_n
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * yinv

            # Update covariance matrix
            delta_hs = (1 - hsig) * (1 + cc) - 1
            rank_one = np.outer(yinv, yinv)
            rank_mu = np.zeros((d, d))
            for i in range(mu):
                yinv_i = Cinv_sqrt @ (x_sorted[i] - xmean) / sigma
                rank_mu += weights[i] * np.outer(yinv_i, yinv_i)
            C = (1 + c1 * delta_hs + c1 + cmu - c1) * C + c1 * (np.outer(pc, pc) + delta_hs - 1) + cmu * rank_mu

            # Step-size adaptation
            sigma *= np.exp(cs / dsigma * (np.linalg.norm(ps) / chi_n - 1))

            # Update mean
            xmean = xnew

            # Update Cinv_sqrt periodically (every 100 iterations or so)
            if calls % 100 == 0 or calls < lam:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-20)
                Cinv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
                C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

        return best_x, best_f
# EVOLVE-BLOCK-END
