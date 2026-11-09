"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid CMA-ES / (1+1)-ES with adaptive population and stall detection."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 270  # 30s safety margin before 300s limit
        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget

        # Phase 1: Initial sampling to find a good starting point
        best_x = None
        best_fx = np.inf
        n_init = min(10, budget)
        for i in range(n_init):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

        x = best_x.copy()
        fx = best_fx

        # CMA-ES state
        mean = x.copy()
        sigma = 1.0
        C = np.eye(dim)
        C_inv_sqrt = np.eye(dim)
        # Step size for CMA-ES
        c_sigma = (1 + 2 * max(0, (np.sqrt(dim) - 1) / (dim + 2))) / (dim + 2 + 2 * max(0, (np.sqrt(dim) - 1) / (dim + 2)))
        c_c = (4 + dim / 4) / (dim + 4 + 2 * c_sigma)
        c_1 = 2 / ((dim + 1.3) ** 2 + dim)
        c_mu = min(1 - c_1, 2 * ((dim + 1) / 2 - 1) / ((dim + 2) ** 2 + dim))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        h_sigma = (1 + c_sigma + 2 / (dim + 1)) * chi_n

        # Population size: start larger, shrink as we converge
        # For dim=5, start with lambda=10, shrink to 5
        lambda_pop = min(10, max(5, 2 * dim))
        mu = lambda_pop // 2
        weights = np.log(lambda_pop + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1 / np.sum(weights ** 2)

        stall_count = 0
        max_stall = 20
        calls_used = n_init
        converged = False

        # Main loop: CMA-ES phase then (1+1)-ES polish
        while calls_used < budget:
            if time.time() > deadline:
                break

            # Check if we should transition to (1+1)-ES polish
            # Transition when sigma gets small or after enough iterations
            if sigma < 0.01 or calls_used > budget * 0.7:
                converged = True
                break

            # Adaptive population: shrink as sigma decreases
            current_lambda = max(5, int(lambda_pop * max(0.5, min(1.0, sigma / 0.5))))
            if current_lambda < 3:
                current_lambda = 3

            # Generate population
            population = []
            for i in range(current_lambda):
                if time.time() > deadline:
                    break
                # CMA-ES step
                z = rng.standard_normal(dim)
                # Apply CMA-ES transformation: x = mean + sigma * C^{1/2} z
                # For simplicity with small dim, use Cholesky
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    L = np.eye(dim)
                step = sigma * L @ z
                y = np.clip(mean + step, -5, 5)
                fy = func(y)
                calls_used += 1
                population.append((fy, y.copy()))

            if not population:
                break

            # Sort by fitness
            population.sort(key=lambda t: t[0])

            # Update mean
            new_mean = np.zeros(dim)
            for i in range(mu):
                new_mean += weights[i] * population[i][1]
            mean = new_mean

            # Update best
            if population[0][0] < best_fx:
                best_fx = population[0][0]
                best_x = population[0][1].copy()

            # Update sigma (CSA)
            y_old = (mean - x) / (sigma * chi_n) if sigma > 1e-15 else np.zeros(dim)
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_old
            sigma *= np.exp(c_sigma / c_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-8, 10.0)

            # Update C matrix
            y_c = (mean - x) / (sigma * chi_n) if sigma > 1e-15 else np.zeros(dim)
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_c

            h_1 = (np.linalg.norm(p_c) / np.sqrt(1 - (1 - c_c) ** (calls_used + 1)) / chi_n) < (1 + c_c + 2 / (dim + 1)) * 1.4
            h_2 = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (calls_used + 1)) / chi_n) < (1 + c_sigma + 2 / (dim + 1)) * 1.4

            # Rank-mu update
            outer = np.zeros((dim, dim))
            for i in range(mu):
                z_i = (population[i][1] - x) / sigma if sigma > 1e-15 else np.zeros(dim)
                outer += weights[i] * np.outer(z_i, z_i)

            C = (1 + c_1 * (2 - c_1) - c_mu) * C + c_1 * np.outer(p_c, p_c) + c_mu * outer
            # Symmetrize
            C = (C + C.T) / 2

            # Ensure positive semi-definite
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-10)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            x = mean.copy()
            fx = best_fx

            # Stall detection
            if population[0][0] >= best_fx - 1e-10:
                stall_count += 1
            else:
                stall_count = 0

            if stall_count >= max_stall:
                stall_count = 0
                # Restart with perturbation
                sigma = max(sigma * 2.0, 0.5)
                mean = best_x + sigma * rng.standard_normal(dim) * 0.5
                mean = np.clip(mean, -5, 5)
                C = np.eye(dim)
                p_sigma = np.zeros(dim)
                p_c = np.zeros(dim)
                x = mean.copy()
                fx = func(x)
                calls_used += 1
                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()

        # Phase 2: (1+1)-ES polish if converged or time remains
        if not converged and calls_used < budget:
            pass  # Already in main loop

        # Final polish with (1+1)-ES using adapted step size
        x = best_x.copy()
        fx = best_fx
        sigma = max(sigma * 0.5, 0.01) if converged else max(sigma, 0.1)
        polish_stall = 0
        max_polish_stall = 15

        while calls_used < budget:
            if time.time() > deadline:
                break

            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls_used += 1

            if fy < fx:
                x, fx = y, fy
                sigma *= 1.2
                polish_stall = 0
                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()
            else:
                sigma *= 0.85
                polish_stall += 1

            if polish_stall >= max_polish_stall:
                polish_stall = 0
                sigma = max(sigma * 0.5, 0.005)
                # Try perturbation around best
                for _ in range(3):
                    if time.time() > deadline:
                        break
                    y2 = np.clip(best_x + sigma * rng.standard_normal(dim), -5, 5)
                    fy2 = func(y2)
                    calls_used += 1
                    if fy2 < fx:
                        x, fx = y2, fy2
                        break
                    if fy2 < best_fx:
                        best_fx = fy2
                        best_x = y2.copy()

                if sigma < 0.001:
                    break

        return best_x, best_fx
# EVOLVE-BLOCK-END
