"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive step size, covariance adaptation, and restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        rng = np.random.default_rng(42)

        # Global best tracking
        global_best_x = None
        global_best_f = np.inf

        # CMA-ES parameters
        dim = self.dim
        # Population size: lambda = 4 + floor(3 * log(dim))
        lam = 4 + int(3 * np.log(dim))
        # Selection size: mu = floor(lam / 2)
        mu = lam // 2
        # Weights
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        # Equivalent sample size
        mu_eff = 1.0 / np.sum(weights ** 2)
        # Learning rates
        cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        cs = (mu_eff + 2) / (dim + mu_eff + 5)
        c1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        # Damping for step size
        damps = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + cs

        def run_cmaes(x0, sigma0, budget):
            nonlocal global_best_x, global_best_f

            x = x0.copy()
            sigma = sigma0
            # Covariance matrix
            C = np.eye(dim)
            # Evolution path for covariance
            pc = np.zeros(dim)
            # Evolution path for step size
            ps = np.zeros(dim)
            # Inverse square root of C
            C_eigvals, C_eigvecs = np.linalg.eigh(C)
            C_eigvals = np.clip(C_eigvals, 1e-20, None)
            C_sqrt_inv = C_eigvecs @ np.diag(1.0 / np.sqrt(C_eigvals)) @ C_eigvecs.T

            fx = func(x)
            if fx < global_best_f:
                global_best_f = fx
                global_best_x = x.copy()

            # Best lambda/2 solutions
            best_x = x.copy()
            best_f = fx

            for gen in range(budget):
                # Check deadline
                if time.time() > deadline:
                    break

                # Generate offspring
                ys = []
                fys = []
                for i in range(lam):
                    z = rng.standard_normal(dim)
                    y = x + sigma * (C_sqrt_inv @ z)
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    ys.append(y)
                    fys.append(fy)

                fys = np.array(fys)
                ys = np.array(ys)

                # Sort by fitness
                order = np.argsort(fys)
                ys_sorted = ys[order]
                fys_sorted = fys[order]

                # Update global best
                if fys_sorted[0] < global_best_f:
                    global_best_f = fys_sorted[0]
                    global_best_x = ys_sorted[0].copy()

                # Update best
                best_x = ys_sorted[0].copy()
                best_f = fys_sorted[0]

                # Weighted mean of best mu solutions
                ymean = np.zeros(dim)
                for i in range(mu):
                    ymean += weights[i] * ys_sorted[i]

                # Update evolution path for step size
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (C_sqrt_inv.T @ (x - ymean) / sigma)
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1)))) <= (1.4 + 2 / (dim + 1)) * np.sqrt(dim + 1 - mu_eff) / np.sqrt(mu_eff)

                # Update step size
                sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / np.sqrt(mu_eff) - 1))
                sigma = np.clip(sigma, 1e-10, 100)

                # Update evolution path for covariance
                pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mu_eff) * (x - ymean) / sigma

                # Update covariance matrix
                delta = hsig * (1 + cc) * (2 - cc) - 1
                delta_outer = np.outer(pc, pc)
                rank1 = (1 - c1) * C + c1 * (delta_outer + delta * np.eye(dim))

                rankmu = (1 - c1 - cmu) * C
                for i in range(mu):
                    zi = C_sqrt_inv.T @ (ys_sorted[i] - x) / sigma
                    rankmu += cmu * weights[i] * np.outer(zi, zi)

                C = rank1 + rankmu

                # Ensure C is symmetric positive definite
                C = (C + C.T) / 2
                C_eigvals, C_eigvecs = np.linalg.eigh(C)
                C_eigvals = np.clip(C_eigvals, 1e-20, None)
                C = C_eigvecs @ np.diag(C_eigvals) @ C_eigvecs.T
                C_sqrt_inv = C_eigvecs @ np.diag(1.0 / np.sqrt(C_eigvals)) @ C_eigvecs.T

                # Update mean
                x = ymean
                x = np.clip(x, -5, 5)

                # Check convergence
                if sigma < 1e-10:
                    break

            return x, best_f

        # Multiple restarts
        n_restarts = 3
        calls_used = 0
        calls_per_restart = self.budget // n_restarts

        for r in range(n_restarts):
            if time.time() > deadline:
                break

            # Initialize
            if r == 0:
                x0 = np.zeros(dim)
            else:
                x0 = rng.uniform(-5, 5, dim)

            sigma0 = 2.0
            budget_this = calls_per_restart if r < n_restarts - 1 else (self.budget - calls_used)

            run_cmaes(x0, sigma0, budget_this)
            calls_used += budget_this

        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
