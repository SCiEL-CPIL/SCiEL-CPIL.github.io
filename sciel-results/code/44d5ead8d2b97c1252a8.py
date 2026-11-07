"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES with deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = 320.0  # Safety margin for 330s limit
        budget = self.budget
        dim = self.dim
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_made = 0

        def check_deadline():
            return (time.time() - start_time) > (deadline - 5.0)

        def do_cmaes(x0, f0, remaining_budget, sigma_init):
            nonlocal best_x, best_f, calls_made
            if remaining_budget <= 0 or check_deadline():
                return

            n = dim
            # CMA-ES parameters
            mu = max(1, n // 2)  # number of parents
            # Weights
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights**2)
            c_sigma = (mueff + 2) / (n + mueff + 5)
            c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            c_1 = 2 / ((n + 1.3)**2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2)**2 + mueff))
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))

            x_mean = x0.copy()
            sigma = sigma_init
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            d_amps = 1.0 / (chi_n * c_sigma)

            # Precompute
            B, D = np.linalg.eigh(C)
            B_inv = np.linalg.inv(B)
            D_inv = 1.0 / np.sqrt(D)
            D_sqrt = np.sqrt(D)

            # Initial population
            pop_size = 2 * mu
            if remaining_budget < pop_size + 1:
                # Fallback to simple hill climbing
                for _ in range(remaining_budget):
                    if check_deadline():
                        return
                    y = np.clip(x_mean + sigma * rng.standard_normal(n), -5, 5)
                    fy = func(y)
                    calls_made += 1
                    if fy < best_f:
                        best_f = fy
                        best_x = y.copy()
                    if fy < f0:
                        x_mean, f0 = y, fy
                        sigma *= 1.2
                    else:
                        sigma *= 0.9
                return

            # Generate initial population
            Y = x_mean + sigma * (B @ (D_inv * (B.T @ rng.standard_normal((pop_size, n))))).T
            Y = np.clip(Y, -5, 5)
            F = np.array([func(y) for y in Y])
            calls_made += pop_size

            # Update best
            idx_best = np.argmin(F)
            if F[idx_best] < best_f:
                best_f = F[idx_best]
                best_x = Y[idx_best].copy()

            # Main loop
            while (budget - calls_made) >= pop_size + 1 and not check_deadline():
                # Sort by fitness
                order = np.argsort(F)
                Y_sorted = Y[order]
                F_sorted = F[order]

                # Update mean
                x_mean_old = x_mean.copy()
                x_mean = np.sum(weights[:, None] * Y_sorted[:mu], axis=0)

                # Update covariance matrix adaptation
                # Compute evolution paths
                y_vec = (x_mean - x_mean_old) / (sigma * chi_n)
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * y_vec
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls_made - pop_size - 1) / pop_size)) <
                        (1.4 + 2 / (n + 1)) * chi_n)

                pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mueff) * y_vec

                # Update C
                delta_hsig = (1 - hsig) * c_c * (2 - c_c)
                C = (1 + c_1 + c_mu - c_1 - c_mu + c_1 * (n + 2) - c_mu) * C  # This is wrong, let me redo
                # Correct update:
                C = (1 - c_1 - c_mu) * C
                C += c_1 * (np.outer(pc, pc) + delta_hsig * C - np.eye(n))
                C += c_mu * np.sum(weights[:, None, None] * np.outer(Y_sorted[:mu] - x_mean_old, Y_sorted[:mu] - x_mean_old), axis=0) / sigma**2

                # Ensure symmetry
                C = (C + C.T) / 2

                # Update sigma
                sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1))

                # Recompute eigen decomposition
                B, D = np.linalg.eigh(C)
                D = np.clip(D, 1e-10, None)
                D_inv = 1.0 / np.sqrt(D)
                D_sqrt = np.sqrt(D)

                # Generate new population
                Z = rng.standard_normal((pop_size, n))
                Y = x_mean + sigma * (B @ (D_inv * (B.T @ Z))).T
                Y = np.clip(Y, -5, 5)
                F = np.array([func(y) for y in Y])
                calls_made += pop_size

                # Update best
                idx_best = np.argmin(F)
                if F[idx_best] < best_f:
                    best_f = F[idx_best]
                    best_x = Y[idx_best].copy()

        # Strategy: multiple restarts with different initializations
        # First, a quick random search to find a good starting point
        num_initial = min(50, budget)
        for i in range(num_initial):
            if check_deadline() or budget - calls_made < 2:
                break
            x = rng.uniform(-4, 4, dim)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            # Should not happen, but just in case
            x = rng.uniform(-5, 5, dim)
            best_x = x.copy()
            best_f = func(x)
            calls_made += 1

        # Now run CMA-ES from the best point found
        remaining = budget - calls_made
        if remaining > 10:
            # Use a reasonable initial sigma
            sigma_init = 1.0
            do_cmaes(best_x, best_f, remaining, sigma_init)

        # If we still have budget, do some local refinement with (1+1)-ES
        remaining = budget - calls_made
        if remaining > 5 and best_x is not None:
            x = best_x.copy()
            fx = best_f
            sigma = 0.5
            for _ in range(remaining):
                if check_deadline():
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_made += 1
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.3
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.85

        return best_x, best_f
# EVOLVE-BLOCK-END
