"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES based optimizer for BBOB f23 (Katsuura) with multi-restart strategy."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self._call_count = 0

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        self._call_count = 0

        # Strict budget enforcement
        class BudgetExceeded(Exception):
            pass

        def counting_func(x):
            self._call_count += 1
            if self._call_count > budget:
                raise BudgetExceeded()
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            return func(x)

        best_x = np.zeros(dim)
        best_f = np.inf

        # Phase 1: Global exploration with CMA-ES
        # Use ~70% of budget for global search
        global_budget = int(0.7 * budget)
        remaining_for_global = global_budget

        # Multiple restarts with different initializations
        n_restarts = 3
        budget_per_restart = remaining_for_global // n_restarts

        for restart in range(n_restarts):
            if self._call_count >= budget:
                break

            # Initialize CMA-ES state
            x0 = np.random.uniform(-3.0, 3.0, dim)  # Start in central region
            sigma = 2.0  # Initial step size

            # CMA-ES parameters
            popsize = max(4, int(4 + 3 * np.log(dim)))  # Standard CMA-ES population size
            mu = popsize // 2  # Number of parents
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mu_eff = 1.0 / np.sum(weights ** 2)

            # Learning rates
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2.0 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

            # State variables
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            C = np.eye(dim)
            chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))

            # Generate initial population
            pop_x = np.zeros((popsize, dim))
            pop_f = np.zeros(popsize)

            for i in range(popsize):
                if self._call_count >= budget:
                    break
                z = np.random.randn(dim)
                x = x0 + sigma * z
                x = np.clip(x, -5.0, 5.0)
                pop_x[i] = x
                try:
                    pop_f[i] = counting_func(x)
                    if pop_f[i] < best_f:
                        best_f = pop_f[i]
                        best_x = x.copy()
                except BudgetExceeded:
                    break

            if self._call_count >= budget:
                break

            # Sort by fitness
            order = np.argsort(pop_f)
            pop_x = pop_x[order]
            pop_f = pop_f[order]

            # Mean update
            x0 = np.sum(weights[:, np.newaxis] * pop_x[:mu], axis=0)

            # Update sigma
            y = (x0 - x0) / (sigma + 1e-30)  # This is zero, but we need the previous mean
            # Actually, we need to track the mean before update
            # Let me restructure this

            # Restart CMA-ES with proper tracking
            x_mean = x0.copy()
            for iteration in range(100):  # Max iterations per restart
                if self._call_count >= budget:
                    break

                # Generate population
                for i in range(popsize):
                    if self._call_count >= budget:
                        break
                    z = np.random.randn(dim)
                    x = x_mean + sigma * z
                    x = np.clip(x, -5.0, 5.0)
                    try:
                        f = counting_func(x)
                        if f < best_f:
                            best_f = f
                            best_x = x.copy()
                        pop_x[i] = x
                        pop_f[i] = f
                    except BudgetExceeded:
                        break

                if self._call_count >= budget:
                    break

                # Sort
                order = np.argsort(pop_f)
                pop_x = pop_x[order]
                pop_f = pop_f[order]

                # Update mean
                x_old = x_mean.copy()
                x_mean = np.sum(weights[:, np.newaxis] * pop_x[:mu], axis=0)

                # Update step size
                y = (x_mean - x_old) / (sigma + 1e-30)
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y
                sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
                sigma = max(sigma, 1e-10)

                # Update covariance
                h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (iteration + 1)))) < (1.4 + 2 / (dim + 1)) * chi_n
                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y
                delta_h = (1 - h_sigma) * c_c * (2 - c_c)
                C = (1 + c_1 + c_mu - c_1 - c_c * (1 + 2 * delta_h)) * C
                C += c_1 * (np.outer(p_c, p_c) - C)
                C += c_mu * np.sum(weights[:, np.newaxis, np.newaxis] * np.outer(pop_x[:mu] - x_old, pop_x[:mu] - x_old), axis=0)

                # Symmetrize
                C = (C + C.T) / 2

                # Check for convergence
                if np.all(np.abs(x_mean - x_old) < 1e-10) or sigma < 1e-12:
                    break

            # After CMA-ES, refine the best point with local search
            if self._call_count < budget:
                local_budget = budget - self._call_count
                if local_budget > 10:
                    # Simple local refinement using coordinate descent
                    x_current = best_x.copy()
                    f_current = best_f
                    step = 0.1
                    for _ in range(50):
                        if self._call_count >= budget:
                            break
                        improved = False
                        for d in range(dim):
                            for direction in [1, -1]:
                                x_try = x_current.copy()
                                x_try[d] += direction * step
                                x_try = np.clip(x_try, -5.0, 5.0)
                                try:
                                    f_try = counting_func(x_try)
                                    if f_try < f_current:
                                        x_current = x_try
                                        f_current = f_try
                                        improved = True
                                        if f_current < best_f:
                                            best_f = f_current
                                            best_x = x_current.copy()
                                except BudgetExceeded:
                                    break
                            if self._call_count >= budget:
                                break
                        if not improved:
                            step *= 0.5
                            if step < 1e-6:
                                break
                        else:
                            step = max(step * 0.9, 1e-6)

        return best_x, best_f

# EVOLVE-BLOCK-END
