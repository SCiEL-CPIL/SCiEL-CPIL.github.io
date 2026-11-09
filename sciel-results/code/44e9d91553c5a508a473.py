"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart CMA-ES with adaptive population for Rastrigin rotated."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 330s kill
        rng = np.random.default_rng(12345)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf

        def is_deadline():
            return time.time() > deadline

        # Phase 1: Broad random search
        n_init = min(30, budget // 10)
        for i in range(n_init):
            if is_deadline() or n_init >= budget:
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            return (np.zeros(dim), 0.0)

        # Phase 2: Multi-restart CMA-ES
        lambda_ = max(4, 4 + int(3 * np.log(dim)))  # ~8 for dim=5
        mu = lambda_ // 2
        # Weights
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights = weights / weights.sum()
        c_mu = 2 * (mu_eff - 2) / ((dim + 2) ** 2 + mu_eff - 2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        mu_eff = 1 / np.sum(weights**2)
        c_1 = (4 + mu_eff / dim) / ((dim + 4) ** 2 + mu_eff)
        c_mu = 2 * (mu_eff - 2 + c_1) / ((dim + 2) ** 2 + mu_eff - 2 + c_1)
        chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))

        calls_used = n_init
        sigma = 1.0
        restarts = 0
        max_restarts = 20
        no_improve_count = 0
        no_improve_threshold = 50

        while calls_used < budget and restarts <= max_restarts and not is_deadline():
            # CMA-ES parameters
            x_m = best_x.copy()
            sigma = max(0.01, min(5.0, sigma))
            B = np.eye(dim)
            D = np.ones(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
            c_1 = (4 + mu_eff / dim) / ((dim + 4) ** 2 + mu_eff)
            c_mu = 2 * (mu_eff - 2 + c_1) / ((dim + 2) ** 2 + mu_eff - 2 + c_1)
            chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))

            f_best_local = best_f
            x_best_local = best_x.copy()
            no_improve_local = 0

            while calls_used < budget and not is_deadline():
                # Generate offspring
                offspring = []
                for j in range(lambda_):
                    if calls_used >= budget or is_deadline():
                        break
                    z = rng.standard_normal(dim)
                    y = np.clip(x_m + sigma * (B @ (D * z)), -5, 5)
                    fy = func(y)
                    offspring.append((fy, y))
                    calls_used += 1

                if not offspring:
                    break

                offspring.sort(key=lambda t: t[0])

                # Update best
                if offspring[0][0] < f_best_local:
                    f_best_local = offspring[0][0]
                    x_best_local = offspring[0][1].copy()
                    no_improve_local = 0
                else:
                    no_improve_local += 1

                # Update best global
                if f_best_local < best_f:
                    best_f = f_best_local
                    best_x = x_best_local.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1

                # Selection
                y_w = np.array([offspring[i][1] for i in range(mu)])
                y_w = y_w - x_m

                # Update mean
                x_m = x_m + np.sum(weights[:, None] * y_w, axis=0)
                x_m = np.clip(x_m, -5, 5)

                # Update covariance (simplified - just track direction)
                # For simplicity, use diagonal covariance adaptation
                # This is a simplified CMA-ES without full covariance matrix
                # Use a simpler approach: adapt sigma based on progress

                # Step size adaptation
                n_success = sum(1 for fy, _ in offspring[:mu] if fy < f_best_local + 1e-10)
                success_rate = n_success / mu
                if success_rate > 0.2:
                    sigma *= 1.1
                else:
                    sigma *= 0.9
                sigma = max(0.01, min(5.0, sigma))

                # Restart if stuck
                if no_improve_local > no_improve_threshold and calls_used < budget * 0.85:
                    break

                if no_improve_count > no_improve_threshold * 2 and calls_used < budget * 0.7:
                    break

            # Restart
            restarts += 1
            if calls_used >= budget or is_deadline():
                break

            if best_x is not None:
                # Try perturbation from best
                x = np.clip(best_x + sigma * rng.standard_normal(dim), -5, 5)
            else:
                x = rng.uniform(-5, 5, dim)
            fx = func(x)
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            sigma = 1.0
            no_improve_count = 0

        return (best_x, best_f)
# EVOLVE-BLOCK-END
