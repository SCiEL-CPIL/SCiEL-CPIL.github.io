"""Optimized for BBOB f15 (Rastrigin rotated) with CMA-ES + restarts + polish."""
import time
import numpy as np


class Optimizer:
    """CMA-ES with multiple restarts and final polish for rotated Rastrigin."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        total_budget = self.budget
        dim = self.dim
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf

        def cma_es(x0, sigma0, budget, func, rng, deadline):
            """Run CMA-ES from x0 with initial sigma0 for given budget."""
            n = dim
            # Population size
            lambda_ = max(4, int(4 + 3 * np.log(n)))
            # Weights
            mu = lambda_ // 2
            w = np.log(np.arange(1, mu + 1))
            w /= w.sum()
            mu_eff = 1.0 / np.sum(w ** 2)
            c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
            c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            x = x0.copy()
            sigma = sigma0
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            d_orth = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_mu

            # Initial population
            calls = 0
            pop = []
            for i in range(lambda_):
                z = rng.standard_normal(n)
                xi = x + sigma * z
                xi = np.clip(xi, -5, 5)
                fi = func(xi)
                calls += 1
                pop.append((fi, xi, z))

            best_f_local = min(p[0] for p in pop)
            best_x_local = [p[1] for p in pop if p[0] == best_f_local][0]

            while calls < budget:
                if time.time() > deadline:
                    break

                # Sort by fitness
                pop.sort(key=lambda p: p[0])

                # Update mean
                x_old = x.copy()
                x = sum(w[i] * pop[i][1] for i in range(mu))

                # Update sigma
                if np.linalg.norm(C) > 0:
                    # Approximate: use identity for C initially, update C below
                    pass

                # Update evolution paths
                z_w = sum(w[i] * pop[i][2] for i in range(mu))
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * z_w
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls // lambda_ + 1)))) < (1.4 + 2 / (n + 1)) * chi_n
                pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mu_eff) * (x - x_old) / sigma

                # Update covariance
                delta_d = hsig * (1 - c_c) - (1 - c_c)
                C = (1 + c_1 * delta_d - c_1 - c_mu) * C
                for i in range(mu):
                    y_i = (pop[i][1] - x_old) / sigma
                    C += c_mu * w[i] * np.outer(y_i, y_i)

                # Ensure C is symmetric positive definite
                C = (C + C.T) / 2

                # Update sigma
                sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1))
                sigma = max(1e-10, min(sigma, 100.0))

                # Generate new population
                pop = []
                for i in range(lambda_):
                    z = rng.standard_normal(n)
                    # Transform z using Cholesky of C
                    try:
                        L = np.linalg.cholesky(C + 1e-10 * np.eye(n))
                        y = L @ z
                    except np.linalg.LinAlgError:
                        y = z
                    xi = x + sigma * y
                    xi = np.clip(xi, -5, 5)
                    fi = func(xi)
                    calls += 1
                    pop.append((fi, xi, z))

                # Track best
                local_best_f = min(p[0] for p in pop)
                if local_best_f < best_f_local:
                    best_f_local = local_best_f
                    best_x_local = [p[1] for p in pop if p[0] == local_best_f][0]

                # Check for convergence
                if np.linalg.norm(x - x_old) < 1e-10 * max(1, np.linalg.norm(x)) and sigma < 1e-6:
                    break

            return best_x_local, best_f_local, calls

        # Multiple restarts with decreasing budgets
        # Strategy: 5 restarts with budgets roughly proportional
        # First restart gets largest budget for exploration
        n_restarts = 5
        # Budget allocation: first restart gets 40%, then decreasing
        budgets = []
        remaining = total_budget
        for i in range(n_restarts):
            if i == n_restarts - 1:
                b = remaining
            else:
                b = int(remaining * (0.4 if i == 0 else 0.3))
                remaining -= b
            budgets.append(b)

        # Reserve some budget for final polish
        polish_budget = 30
        total_search = total_budget - polish_budget
        # Scale budgets to fit total_search
        scale = total_search / sum(budgets)
        budgets = [max(10, int(b * scale)) for b in budgets]
        # Ensure sum doesn't exceed total_search
        while sum(budgets) > total_search:
            idx = np.argmax(budgets)
            budgets[idx] -= 1
        while sum(budgets) < total_search:
            idx = np.argmin(budgets)
            budgets[idx] += 1

        for r in range(n_restarts):
            if time.time() > deadline:
                break

            # Different starting points for each restart
            if r == 0:
                x0 = rng.uniform(-5, 5, dim)
                sigma0 = 3.0
            elif r == 1:
                x0 = rng.uniform(-4, 4, dim)
                sigma0 = 2.0
            elif r == 2:
                x0 = rng.uniform(-3, 3, dim)
                sigma0 = 1.5
            elif r == 3:
                x0 = rng.uniform(-2, 2, dim)
                sigma0 = 1.0
            else:
                # Start near best found so far
                if best_x is not None:
                    x0 = best_x + rng.uniform(-1, 1, dim)
                else:
                    x0 = rng.uniform(-5, 5, dim)
                sigma0 = 0.8

            x0 = np.clip(x0, -5, 5)

            x_best_r, f_best_r, calls_used = cma_es(
                x0, sigma0, budgets[r], func, rng, deadline
            )

            if f_best_r < best_f:
                best_f = f_best_r
                best_x = x_best_r.copy()

        # Final polish: local search around best point
        if best_x is not None and time.time() < deadline:
            # Use a fine-grained local search
            polish_budget_actual = min(polish_budget, total_budget - sum(budgets))
            if polish_budget_actual > 0:
                x = best_x.copy()
                fx = best_f
                sigma = 0.3
                for _ in range(polish_budget_actual):
                    if time.time() > deadline:
                        break
                    y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                    fy = func(y)
                    if fy < fx:
                        x, fx = y, fy
                        sigma *= 1.2
                    else:
                        sigma *= 0.85
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)

        return best_x, best_f
