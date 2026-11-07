"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive restarts and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _run_cma(self, func, x0, budget_remaining, rng, deadline):
        """Run CMA-ES from x0 for up to budget_remaining evaluations."""
        dim = self.dim
        n = dim
        lam = max(4 + int(3 * np.log(n)), 8)
        sigma = 1.0
        x = x0.copy()
        fx = func(x)
        budget_used = 1

        # CMA-ES parameters
        c_sigma = (mu_eff + 1) / ((n + 2) * (mu_eff + 2))
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1) - 1)) + c_sigma
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        # Initialize
        pc = np.zeros(n)
        ps = np.zeros(n)
        B = np.eye(n)
        D = np.ones(n)
        damps = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1) - 1)) + c_sigma

        # Selection
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1 / np.sum(weights ** 2)

        # Initial population
        y = [x + sigma * rng.standard_normal(n) for _ in range(lam)]
        y = [np.clip(yp, -5, 5) for yp in y]
        fy = [func(yp) for yp in y]
        budget_used += lam

        # Sort
        order = np.argsort(fy)
        y = [y[i] for i in order[:mu]]
        fy = [fy[i] for i in order[:mu]]

        # Main loop
        while budget_used < budget_remaining:
            # Check deadline
            if time.time() > deadline - 2.0:
                break

            # Update covariance matrix
            y_mean = np.sum([w * yp for w, yp in zip(weights, y)], axis=0)
            pc = (1 - c_c) * pc + c_c * (y_mean / sigma)
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_mean / sigma

            # Hessian correction
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (budget_used - 1) / (n * c_sigma)))) < (1.4 + 2 / (n + 1)) * chi_n
            delta_hsig = 1.0 if not hsig else 0.0

            # Update B and D
            C = np.outer(B, B) * np.outer(D, D)
            C = (1 + c_1 * delta_hsig * (np.outer(pc, pc) - C - 2 * c_c * C + c_c ** 2 * C) - c_1 * C + c_mu * sum(w * np.outer(yp - x, yp - x) for w, yp in zip(weights, y)))
            C = 0.5 * (C + C.T)

            # Update B and D
            evals = np.linalg.eigh(C)
            D = np.sqrt(np.maximum(evals[0], 1e-20))
            B = evals[1]

            # Update step size
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            sigma = min(max(sigma, 1e-8), 1e3)

            # Update mean
            x = x + sum(w * yp for w, yp in zip(weights, y))

            # Generate new population
            y = [x + sigma * (B @ (D * rng.standard_normal(n))) for _ in range(lam)]
            y = [np.clip(yp, -5, 5) for yp in y]

            # Check budget
            if budget_used + lam > budget_remaining:
                # Only evaluate remaining budget
                remaining = budget_remaining - budget_used
                y = y[:remaining]
                fy = [func(yp) for yp in y]
                budget_used = budget_remaining
                # Sort and update
                order = np.argsort(fy)
                if len(order) >= mu:
                    y = [y[i] for i in order[:mu]]
                    fy = [fy[i] for i in order[:mu]]
                break

            fy = [func(yp) for yp in y]
            budget_used += lam

            # Sort
            order = np.argsort(fy)
            y = [y[i] for i in order[:mu]]
            fy = [fy[i] for i in order[:mu]]

        return x, fx, budget_used

    def __call__(self, func):
        deadline = time.time() + 290  # Safety margin before 300s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf

        # Try multiple restarts with different initial points
        n_restarts = 3
        budget_per_restart = budget // n_restarts

        for r in range(n_restarts):
            if time.time() > deadline - 5.0:
                break

            if r == 0:
                x0 = np.zeros(dim)
            elif r == 1:
                x0 = rng.uniform(-3, 3, dim)
            else:
                x0 = rng.uniform(-4, 4, dim)

            # Quick initial evaluation
            fx0 = func(x0)
            if fx0 < best_f:
                best_f = fx0
                best_x = x0.copy()

            # Run CMA-ES
            x_opt, f_opt, used = self._run_cma(func, x0, budget_per_restart, rng, deadline)
            if f_opt < best_f:
                best_f = f_opt
                best_x = x_opt.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
