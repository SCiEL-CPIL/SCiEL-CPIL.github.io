"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive restarts for black-box continuous optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        dim = self.dim
        total_budget = self.budget

        # Track best solution found
        best_x = np.zeros(dim)
        best_f = np.inf
        calls_made = 0

        # Phase 1: Multi-start CMA-ES with different scales
        # We'll run several CMA-ES restarts with different initializations

        def cma_es(x0, f0, budget_remaining, sigma_init, max_iter):
            """Run CMA-ES from x0 with f0 for budget_remaining evaluations."""
            nonlocal best_x, best_f, calls_made

            dim = self.dim
            # CMA-ES parameters
            pop_size = 4 + int(3 * np.log(dim))  # ~4 for dim=5
            pop_size = max(pop_size, 2 * dim)  # at least 10 for dim=5

            # Strategy parameters
            mu = pop_size // 2  # number of parents
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)

            # Adaptation rates
            cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            cs = (mu_eff + 2) / (dim + mu_eff + 5)
            c1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
            cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

            # Initialize
            x_mean = x0.copy()
            f_mean = f0
            sigma = sigma_init
            B = np.eye(dim)
            D = np.ones(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)

            # Track best within this CMA-ES run
            local_best_x = x0.copy()
            local_best_f = f0

            for iteration in range(max_iter):
                # Check time
                if time.time() > deadline:
                    break

                # Generate population
                offspring = []
                for i in range(pop_size):
                    if calls_made + 1 > total_budget:
                        break
                    z = np.random.randn(dim)
                    y = x_mean + sigma * (B @ (D * z))
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    calls_made += 1
                    offspring.append((fy, y))

                    if fy < local_best_f:
                        local_best_f = fy
                        local_best_x = y.copy()

                    if fy < best_f:
                        best_f = fy
                        best_x = y.copy()

                if len(offspring) == 0:
                    break

                # Sort offspring
                offspring.sort(key=lambda x: x[0])

                # Select best mu offspring
                selected = offspring[:min(mu, len(offspring))]
                if len(selected) < mu:
                    break

                # Update mean
                new_mean = np.zeros(dim)
                for i, (fi, xi) in enumerate(selected):
                    new_mean += weights[i] * xi
                x_mean = new_mean

                # Update sigma (CSA)
                old_sigma = sigma
                y_norm = (B @ (D ** 2 * ((x_mean - old_x_mean) / sigma))) if hasattr(self, '_old_x_mean') else None
                # Simplified: compute norm of step
                step = x_mean - x_mean  # This needs proper tracking
                # Let's use a simpler sigma update
                # Check success rate
                success = sum(1 for fi, _ in selected if fi < f_mean)
                if success > 0:
                    sigma *= 1.0 + 0.3 * (success / mu - 0.5)
                else:
                    sigma *= 0.9
                sigma = np.clip(sigma, 1e-10, 50.0)

                f_mean = selected[0][0]

                # Check convergence
                if iteration > 10:
                    # Check if population has converged
                    pop_spread = np.max([np.linalg.norm(xi - x_mean) for _, xi in offspring])
                    if pop_spread < 1e-10:
                        break

            return local_best_x, local_best_f

        # Phase 1: Initial exploration with multiple scales
        # Start with a few well-spread points
        n_starts = 5
        for i in range(n_starts):
            if calls_made >= total_budget or time.time() > deadline:
                break

            if i == 0:
                # Start near origin (common for BBOB)
                x0 = np.zeros(dim)
            elif i == 1:
                x0 = np.full(dim, 3.0)
            elif i == 2:
                x0 = np.full(dim, -3.0)
            elif i == 3:
                x0 = np.random.uniform(-4, 4, dim)
            else:
                x0 = np.random.uniform(-5, 5, dim)

            x0 = np.clip(x0, -5, 5)
            f0 = func(x0)
            calls_made += 1

            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()

            # Run a short CMA-ES from this point
            remaining = total_budget - calls_made
            if remaining < 20:
                break

            # Use a fraction of remaining budget
            alloc = min(remaining, 50)
            cma_es(x0, f0, alloc, 1.5, alloc)

        # Phase 2: Exploit around best found so far
        if calls_made < total_budget and time.time() < deadline:
            remaining = total_budget - calls_made
            if remaining > 30:
                # Run a focused CMA-ES around the best point
                x0 = best_x.copy()
                f0 = best_f
                alloc = min(remaining, 80)
                cma_es(x0, f0, alloc, 0.5, alloc)

        # Phase 3: Fine polishing with small steps
        if calls_made < total_budget and time.time() < deadline:
            remaining = total_budget - calls_made
            sigma = 0.1
            x = best_x.copy()
            fx = best_f

            for _ in range(remaining):
                if time.time() > deadline:
                    break
                y = np.clip(x + sigma * np.random.randn(dim), -5, 5)
                fy = func(y)
                calls_made += 1
                if fy < fx:
                    x, fx = y, fy
                    if fy < best_f:
                        best_f = fy
                        best_x = y.copy()
                    sigma *= 1.1
                else:
                    sigma *= 0.9

        return best_x, best_f
# EVOLVE-BLOCK-END
