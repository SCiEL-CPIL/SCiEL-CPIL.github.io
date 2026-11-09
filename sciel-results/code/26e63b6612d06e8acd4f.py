"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid CMA-ES with (1+1)-ES refinement for Katsuura f23."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 330s kill

        rng = np.random.default_rng(42)

        # Global best tracking
        best_x = np.zeros(n)
        best_f = float('inf')
        total_calls = 0

        def cma_es(x0, sigma0, max_calls, rng_local):
            """Run CMA-ES from x0 with initial sigma0, using up to max_calls evaluations."""
            nonlocal best_x, best_f, total_calls

            # CMA-ES parameters
            x = x0.copy()
            sigma = sigma0
            lambda_ = 4 + int(3 * np.log(n))
            mu = lambda_ // 2
            weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / weights.sum()
            c_sigma = (mueff + 2) / (n + mueff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma
            hsig_threshold = 1.4 + 2 / (n + 1)
            c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            c_1 = 2 / ((n + 1.3) ** 2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            local_calls = 0
            local_best_f = float('inf')
            local_best_x = x.copy()

            # Evaluate initial point
            f_init = func(x)
            local_calls += 1
            total_calls += 1
            local_best_f = f_init
            local_best_x = x.copy()
            if f_init < best_f:
                best_f = f_init
                best_x = x.copy()

            while local_calls < max_calls and time.time() < deadline:
                # Generate population
                try:
                    B = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    B = np.eye(n)

                offspring = []
                for i in range(lambda_):
                    z = rng_local.standard_normal(n)
                    y = x + sigma * B @ z
                    y = np.clip(y, -5, 5)
                    offspring.append(y)

                # Evaluate offspring
                for i in range(lambda_):
                    if local_calls >= max_calls or time.time() > deadline:
                        break
                    fy = func(offspring[i])
                    local_calls += 1
                    total_calls += 1
                    if fy < local_best_f:
                        local_best_f = fy
                        local_best_x = offspring[i].copy()
                    if fy < best_f:
                        best_f = fy
                        best_x = offspring[i].copy()
                    offspring[i] = (offspring[i], fy)

                if local_calls >= max_calls or time.time() > deadline:
                    break

                # Sort by fitness
                offspring.sort(key=lambda t: t[1])
                selected = [offspring[i][0] for i in range(mu)]

                # Update mean
                m_old = x.copy()
                x_new = sum(weights[i] * selected[i] for i in range(mu))

                # Update evolution paths
                diff = x_new - m_old
                norm_diff = np.linalg.norm(diff)
                if norm_diff > 1e-12:
                    z = diff / (sigma * norm_diff)
                    try:
                        C_inv_B = np.linalg.inv(B)
                        z_transformed = C_inv_B @ z
                    except np.linalg.LinAlgError:
                        z_transformed = z
                    ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * z_transformed
                    hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (local_calls - (local_calls - lambda_))) / chi_n < hsig_threshold
                    if hsig:
                        pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mueff) * diff / sigma
                    else:
                        pc = (1 - c_c) * pc
                else:
                    hsig = True

                # Update covariance
                delta_hsig = (1.0 if hsig else 0.0) - (1 - c_sigma)
                rank_one = np.outer(ps, ps)
                rank_mu = sum(weights[i] * np.outer(selected[i] - m_old, selected[i] - m_old) for i in range(mu))
                C = (1 + c_c * delta_hsig - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) + (1 if not hsig else 0) * np.eye(n)) + c_mu * rank_mu

                # Update sigma
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
                sigma = np.clip(sigma, 1e-10, 10.0)

                x = x_new

            return local_best_x, local_best_f

        def one_plus_one_es(x0, fx0, max_calls, rng_local):
            """(1+1)-ES with 1/5 success rule for local refinement."""
            nonlocal best_x, best_f, total_calls

            x = x0.copy()
            fx = fx0
            sigma = 1.0

            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            for _ in range(max_calls):
                if total_calls >= budget or time.time() > deadline:
                    break

                y = x + sigma * rng_local.standard_normal(n)
                y = np.clip(y, -5, 5)
                fy = func(y)
                total_calls += 1

                if fy < fx:
                    x = y
                    fx = fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma *= 1.5
                else:
                    sigma *= 0.87

                if sigma < 1e-8:
                    sigma = 1e-8
                if sigma > 10.0:
                    sigma = 10.0

            return x, fx

        # Phase 1: Global exploration with multiple CMA-ES restarts
        # Start with a large sigma to explore the full domain
        n_restarts = 4
        calls_per_restart = (budget - 1) // n_restarts

        for r in range(n_restarts):
            if total_calls >= budget or time.time() > deadline:
                break

            if r == 0:
                x_start = np.zeros(n)
                sigma_start = 4.0
            else:
                x_start = rng.uniform(-4, 4, n)
                sigma_start = 3.0

            remaining = budget - total_calls
            if remaining <= 0:
                break

            max_calls_this = min(calls_per_restart, remaining)
            _, _ = cma_es(x_start, sigma_start, max_calls_this, rng)

        # Phase 2: Local refinement with (1+1)-ES around the best point
        remaining = budget - total_calls
        if remaining > 20 and time.time() < deadline:
            # Run (1+1)-ES from the best point found
            max_calls_refine = min(remaining, 100)
            _, _ = one_plus_one_es(best_x.copy(), best_f, max_calls_refine, rng)

        # Phase 3: Final fine-tuning with very small perturbations
        remaining = budget - total_calls
        if remaining > 5 and time.time() < deadline:
            x_refine = best_x.copy()
            sigma_refine = 0.1
            max_calls_final = min(remaining, 50)

            for _ in range(max_calls_final):
                if total_calls >= budget or time.time() > deadline:
                    break
                y = x_refine + sigma_refine * rng.standard_normal(n)
                y = np.clip(y, -5, 5)
                fy = func(y)
                total_calls += 1
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                    x_refine = y.copy()
                    sigma_refine *= 0.9
                else:
                    sigma_refine *= 0.95
                    sigma_refine = max(sigma_refine, 0.001)

        return best_x, best_f
# EVOLVE-BLOCK-END
