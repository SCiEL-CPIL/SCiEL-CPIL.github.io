"""Portfolio-based continuous optimizer for BBOB-style f17 (Schaffers F7)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-restart CMA-ES with deadline guard.
    Schaffers F7 has many local minima; CMA-ES with a population and
    multiple restarts explores the landscape more effectively.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cma_es(self, func, x0, sigma0, budget, rng, deadline):
        """Run CMA-ES from x0 with initial step size sigma0 for up to budget evaluations."""
        n = self.dim
        x = np.array(x0, dtype=float)
        sigma = sigma0
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)

        # CMA-ES parameters
        mu = max(1, n // 2)  # number of selected individuals
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1.0 / (w @ w)
        c_sigma = (1 + 2 / max(n - 1, 1) + mu_eff / 2) / (n + 4 + mu_eff / 2)
        d_sigma = 1 + 2 * max((n - 1) / (n + 2) - 1, 0) + min(mu_eff / 2 - 1, 0.5)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))

        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        x_best = x.copy()
        f_best = func(x)
        calls = 1

        while calls < budget:
            if time.time() > deadline:
                break

            # Generate population
            B = rng.standard_normal((mu, n))
            # Apply C: C = D D^T, so C^{1/2} v = D (D^T v)
            # We maintain C as a matrix; use eigen-decomposition for square root
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-10, None)
            D = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

            pop = np.array([x + sigma * (D @ b) for b in B])
            pop = np.clip(pop, -5, 5)

            # Evaluate
            fvals = []
            for xi in pop:
                if calls >= budget or time.time() > deadline:
                    break
                fi = func(xi)
                fvals.append(fi)
                calls += 1
                if fi < f_best:
                    f_best = fi
                    x_best = xi.copy()

            if len(fvals) < mu:
                break

            # Sort by fitness
            order = np.argsort(fvals)
            fvals = np.array(fvals)
            B = B[order]
            fvals = fvals[order]

            # Update mean
            y_w = w @ B
            x_new = x + sigma * (D @ y_w)
            x = np.clip(x_new, -5, 5)

            # Update path and covariance
            y_invC = D @ y_w
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (D.T @ y_invC) / chi_n
            hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls - budget + 1))) < (1.4 + 2 / (n + 1)) * chi_n
            pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_invC

            delta_hsig = (1 - hsig) * c_c * (2 - c_c)
            rank_mu = np.outer(w @ B, w @ B).T  # sum w_i w_j y_i y_j^T
            rank_1 = np.outer(pc, pc)
            C = (1 + c_1 * delta_hsig - c_1 - c_mu) * C + c_1 * rank_1 + c_mu * rank_mu

            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 100)

            # Ensure C is symmetric positive definite
            C = (C + C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-10, None)
            C = (eigvecs * eigvals) @ eigvecs.T

        return x_best, f_best, calls

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: stop 10s before the 330s limit
        deadline = start_time + 320.0

        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        # Strategy: split budget among multiple restarts
        # Use CMA-ES with small population for each restart
        # Number of restarts: adaptive based on time
        # Each CMA-ES run uses a fraction of the budget

        # Start with a broad exploration phase, then refine
        # Use 5 restarts initially, adjust based on time

        best_x = None
        best_f = np.inf
        total_calls = 0

        # Phase 1: Multiple diverse starts
        n_restarts = 8
        per_restart_budget = max(1, (budget - n_restarts) // n_restarts)

        for r in range(n_restarts):
            if time.time() > deadline:
                break
            if total_calls >= budget:
                break

            # Diverse starting points
            if r == 0:
                x0 = np.zeros(n)
            elif r == 1:
                x0 = np.array([3.0, -3.0, 3.0, -3.0, 3.0])
            elif r == 2:
                x0 = np.array([-3.0, 3.0, -3.0, 3.0, -3.0])
            elif r == 3:
                x0 = rng.uniform(-4, 4, n)
            elif r == 4:
                x0 = rng.uniform(-4, 4, n)
            elif r == 5:
                x0 = rng.uniform(-4, 4, n)
            elif r == 6:
                x0 = np.array([2.5, 2.5, 2.5, 2.5, 2.5])
            else:
                x0 = rng.uniform(-4, 4, n)

            sigma0 = 1.5 if r < 3 else 2.0

            x, f, calls_used = self._cma_es(func, x0, sigma0, per_restart_budget, rng, deadline)
            total_calls += calls_used

            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 2: If time remains, do a final refinement from the best point
        if total_calls < budget and time.time() < deadline:
            remaining = min(budget - total_calls, 100)
            if remaining > 10 and best_x is not None:
                x, f, calls_used = self._cma_es(func, best_x, 0.5, remaining, rng, deadline)
                total_calls += calls_used
                if f < best_f:
                    best_f = f
                    best_x = x.copy()

        if best_x is None:
            # Fallback: single evaluation
            best_x = np.zeros(n)
            best_f = func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
