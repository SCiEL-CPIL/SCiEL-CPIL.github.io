"""CMA-ES optimizer with restarts, corrected update rules, and final polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with multiple restarts, corrected covariance update, and final polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Global best
        best_x = np.full(dim, 0.0)
        best_f = np.inf
        calls_used = 0

        def evaluate(x):
            nonlocal calls_used
            x_clipped = np.clip(x, -5, 5)
            fx = float(func(x_clipped))
            calls_used += 1
            return fx

        def cma_es(x_mean, sigma, budget_for_this_restart):
            """Run CMA-ES from given mean and sigma. Returns (best_x, best_f, calls_used)."""
            nonlocal best_x, best_f, calls_used, deadline, rng

            n = dim
            # Population size and selection
            lambda_ = max(4 * n, 10)  # larger population for better gradient
            mu = lambda_ // 2  # half the population as parents
            # Logarithmic weights
            weights = np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)

            # Strategy parameters
            c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
            c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1.0 / mu_eff) / ((n + 2) ** 2 + mu_eff))
            chi_n = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n ** 2))

            # State
            sigma = sigma
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)
            C = np.eye(n)
            B, D = np.linalg.eigh(C)
            D = np.sqrt(np.maximum(D, 1e-12))

            local_best_x = x_mean.copy()
            local_best_f = np.inf

            max_iters = budget_for_this_restart // lambda_
            for it in range(max_iters):
                if calls_used >= budget:
                    break
                if time.time() > deadline:
                    break

                # Generate population
                population = []
                for i in range(lambda_):
                    z = rng.standard_normal(n)
                    y = B @ (D * z)
                    xi = x_mean + sigma * y
                    fi = evaluate(xi)
                    population.append((fi, xi, y))

                if calls_used >= budget:
                    break
                if time.time() > deadline:
                    break

                # Sort by fitness
                population.sort(key=lambda t: t[0])

                # Track local best
                if population[0][0] < local_best_f:
                    local_best_f = population[0][0]
                    local_best_x = population[0][1].copy()
                if population[0][0] < best_f:
                    best_f = population[0][0]
                    best_x = population[0][1].copy()

                # Extract mu best steps
                y_sorted = [population[i][2] for i in range(mu)]
                x_sorted = [population[i][1] for i in range(mu)]

                # Update mean: weighted average of mu best
                x_mean = sum(w * xi for w, xi in zip(weights, x_sorted))

                # Update sigma path
                y_w_mean = sum(w * yi for w, yi in zip(weights, y_sorted))
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / chi_n * B.T @ y_w_mean

                # Update C path
                h_sigma = 1.0 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (it + 1))) < (1.4 + 2.0 / (n + 1)) * chi_n else 0.0
                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * B.T @ y_w_mean

                # Update covariance
                rank_one = np.outer(p_c, p_c) - (1 - c_c) * C
                rank_mu = sum(w * np.outer(yi, yi) for w, yi in zip(weights, y_sorted))
                C = (1 + c_1 * c_mu - c_1 - c_1) * C + c_1 * rank_one + c_mu * rank_mu

                # Ensure C is symmetric and positive definite
                C = 0.5 * (C + C.T)
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T

                # Update eigen decomposition
                B, D = np.linalg.eigh(C)
                D = np.sqrt(np.maximum(D, 1e-12))

                # Adapt sigma
                sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
                sigma = np.clip(sigma, 1e-10, 10.0)

                # Convergence check
                if sigma < 1e-10:
                    break

            return local_best_x, local_best_f

        # Restart strategy: broad to fine
        restarts = [
            (int(budget * 0.45), 4.0),   # large sigma, broad exploration
            (int(budget * 0.25), 2.0),
            (int(budget * 0.15), 1.0),
            (int(budget * 0.10), 0.5),
            (int(budget * 0.05), 0.2),
        ]

        for restart_idx, (restart_budget, sigma0) in enumerate(restarts):
            if calls_used >= budget:
                break
            if time.time() > deadline:
                break

            # Initialize: first restart broad, later restarts near best
            if restart_idx == 0:
                x_init = rng.uniform(-4, 4, dim)
            else:
                x_init = best_x + rng.standard_normal(dim) * 0.3
                x_init = np.clip(x_init, -5, 5)

            cma_es(x_init, sigma0, restart_budget)

        # Final polish: simple local search with decreasing step size
        if calls_used < budget and time.time() < deadline:
            polish_calls = min(budget - calls_used, 30)
            step_sizes = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]
            for ss in step_sizes:
                if calls_used >= budget or time.time() > deadline:
                    break
                for coord in range(dim):
                    if calls_used >= budget or time.time() > deadline:
                        break
                    for direction in [1, -1]:
                        if calls_used >= budget or time.time() > deadline:
                            break
                        x_trial = best_x.copy()
                        x_trial[coord] += direction * ss
                        f_trial = evaluate(x_trial)
                        if f_trial < best_f:
                            best_f = f_trial
                            best_x = x_trial.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
