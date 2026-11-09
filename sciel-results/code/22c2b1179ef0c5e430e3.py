"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart CMA-ES for robust global optimization on rotated multimodal functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        total_budget = self.budget

        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit

        rng = np.random.default_rng(0)

        best_x = None
        best_f = float('inf')
        calls_made = 0

        # Phase 1: Initial random sampling to find a good starting point
        init_budget = max(20, int(total_budget * 0.10))
        init_points = rng.uniform(-5, 5, (init_budget, n))
        for i in range(init_budget):
            if calls_made >= total_budget or time.time() > deadline:
                break
            fval = func(init_points[i])
            calls_made += 1
            if fval < best_f:
                best_f = fval
                best_x = init_points[i].copy()

        # Phase 2: Multi-restart CMA-ES
        # Each restart gets a portion of the remaining budget
        remaining = total_budget - calls_made
        if remaining <= 0:
            return best_x, best_f

        # Determine number of restarts based on budget
        num_restarts = max(2, int(np.sqrt(remaining / 50)))
        budget_per_restart = remaining // num_restarts

        for restart in range(num_restarts):
            if calls_made >= total_budget or time.time() > deadline:
                break

            # For first restart, use best point found; for others, sample near best
            if restart == 0:
                x0 = best_x.copy() if best_x is not None else rng.uniform(-5, 5, n)
            else:
                # Sample near the best point with some perturbation
                sigma_init = 2.0 / (restart + 1)
                x0 = best_x + sigma_init * rng.standard_normal(n)
                x0 = np.clip(x0, -5, 5)

            # CMA-ES parameters
            pop_size = max(4, int(4 + 3 * np.log(n)))
            sigma = 1.0
            x_mean = x0.copy()

            # Evaluate initial population
            pop = x_mean + sigma * rng.standard_normal((pop_size, n))
            pop = np.clip(pop, -5, 5)
            fvals = np.array([func(p) for p in pop])
            calls_made += pop_size

            # Sort by fitness
            order = np.argsort(fvals)
            pop = pop[order]
            fvals = fvals[order]

            # Update best
            if fvals[0] < best_f:
                best_f = fvals[0]
                best_x = pop[0].copy()

            # CMA-ES parameters
            mu = pop_size // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mu_eff = 1.0 / np.sum(weights ** 2)
            c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
            c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))

            # Initialize state
            C = np.eye(n)
            C_sqrt_inv = np.eye(n)
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            # Run CMA-ES until budget exhausted
            while calls_made < total_budget and time.time() < deadline:
                if calls_made + pop_size > total_budget:
                    # Evaluate only as many as we can
                    eval_count = total_budget - calls_made
                else:
                    eval_count = pop_size

                if eval_count <= 0:
                    break

                # Generate new population
                pop = x_mean + sigma * rng.standard_normal((eval_count, n))
                pop = np.clip(pop, -5, 5)
                fvals = np.array([func(p) for p in pop])
                calls_made += eval_count

                # Sort by fitness
                order = np.argsort(fvals)
                pop = pop[order]
                fvals = fvals[order]

                # Update best
                if fvals[0] < best_f:
                    best_f = fvals[0]
                    best_x = pop[0].copy()

                # Use top mu for weight update
                mu_use = min(mu, eval_count)
                if mu_use < 2:
                    break

                # Update mean
                x_new_mean = np.sum(weights[:mu_use, np.newaxis] * pop[:mu_use], axis=0)

                # Update step-size control
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_sqrt_inv @ (x_new_mean - x_mean) / sigma
                if np.linalg.norm(p_sigma) / chi_n < (1.4 + 1 / mu_eff) * np.sqrt(1 - (1 - c_sigma) ** (2 * (calls_made - init_budget))):
                    sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))

                # Update covariance matrix
                diff = x_new_mean - x_mean
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * diff / sigma
                delta_h = 1 if np.linalg.norm(p_c / np.sqrt(1 - (1 - c_c) ** 2)) / chi_n < (1.4 + 1 / mu_eff) * np.sqrt(1 - (1 - c_c) ** (2 * (calls_made - init_budget))) else 0

                C = (1 + c_1 * delta_h - c_1 - c_mu) * C + c_1 * np.outer(p_c, p_c) + c_mu * np.sum(weights[:mu_use, np.newaxis, np.newaxis] * np.outer(pop[:mu_use] - x_mean, pop[:mu_use] - x_mean), axis=0) / sigma ** 2

                # Ensure C is positive definite and compute sqrt inverse
                C = (C + C.T) / 2
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-20)
                    C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                    C_sqrt_inv = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
                except np.linalg.LinAlgError:
                    C = np.eye(n)
                    C_sqrt_inv = np.eye(n)

                x_mean = x_new_mean

                # Check for convergence
                if sigma < 1e-12:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
