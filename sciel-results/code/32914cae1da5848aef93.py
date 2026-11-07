"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Population-based CMA-ES with restarts for Büche-Rastrigin."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_made = 0

        def elapsed_ok():
            return time.time() < deadline

        # Multi-restart strategy: try several initializations
        # For Büche-Rastrigin (has local minima), restarts help escape
        n_restarts = 5
        budget_per_restart = budget // n_restarts
        if budget_per_restart < 50:
            n_restarts = 1
            budget_per_restart = budget

        for restart in range(n_restarts):
            if calls_made >= budget or not elapsed_ok():
                break

            # Initialize center
            if restart == 0:
                center = np.zeros(dim)
            else:
                center = rng.uniform(-4, 4, dim)

            # CMA-ES parameters
            sigma = 2.0
            pop_size = max(4, dim + int(3 * np.log(dim)))
            pop_size = min(pop_size, 20)  # cap for budget efficiency

            # Covariance matrix
            C = np.eye(dim)
            # Evolution paths
            ps = np.zeros(dim)
            hc = (1 + 1 / (pop_size - 1)) * (pop_size + 2) / (pop_size - 1)
            cc = (4 + pop_size / dim) / (pop_size + 4 + 2 / dim)
            cmu = min(0.2, (2 - (dim + 2) / (pop_size + 2) ** 2) / ((pop_size + 2) + 2))
            cs = (pop_size + 2) / (pop_size + 5)
            hs = (1 + 1 / (pop_size - 1)) * (pop_size + 2) / (pop_size - 1)

            # Cholesky of C
            try:
                C_eigval, C_eigvec = np.linalg.eigh(C)
                C_eigval = np.clip(C_eigval, 1e-12, None)
                C_sqrt = C_eigvec @ np.diag(np.sqrt(C_eigval)) @ C_eigvec.T
            except:
                C_sqrt = np.eye(dim)

            x = center.copy()
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            # Main loop
            for gen in range(budget_per_restart - 1):
                if calls_made >= budget or not elapsed_ok():
                    break

                # Generate population
                W = rng.standard_normal((pop_size, dim))
                offspring = np.array([
                    np.clip(x + sigma * (C_sqrt @ w), -5, 5) for w in W
                ])

                # Evaluate
                offspring_f = np.array([func(offspring[i]) for i in range(pop_size)])
                calls_made += pop_size

                # Update best
                for i in range(pop_size):
                    if offspring_f[i] < best_f:
                        best_f = offspring_f[i]
                        best_x = offspring[i].copy()

                # Sort
                order = np.argsort(offspring_f)
                lam = pop_size // 2  # number of parents

                # Update mean
                weights = np.ones(lam)
                weights /= weights.sum()
                new_mean = np.sum(weights[:, None] * offspring[order[:lam]], axis=0)

                # Update covariance
                diff = (new_mean - x) / sigma
                C = (1 - cmu) * C + cmu * np.outer(diff, diff)
                # Regularize
                C = (C + C.T) / 2
                C += 1e-12 * np.eye(dim)

                # Update sigma
                ps = (1 - cs) * ps + cs * diff
                sigma *= np.exp(cs / (pop_size + 1) * (np.linalg.norm(ps) / np.sqrt(pop_size) - 1))
                sigma = np.clip(sigma, 1e-6, 10.0)

                x = new_mean

                # Update Cholesky periodically
                if gen % 5 == 0:
                    try:
                        C_eigval, C_eigvec = np.linalg.eigh(C)
                        C_eigval = np.clip(C_eigval, 1e-12, None)
                        C_sqrt = C_eigvec @ np.diag(np.sqrt(C_eigval)) @ C_eigvec.T
                    except:
                        C_sqrt = np.eye(dim)

        # Final polish with (1+1)-ES around best point
        if best_x is not None and calls_made < budget and elapsed_ok():
            x = best_x.copy()
            fx = best_f
            sigma = 0.5
            remaining = budget - calls_made
            for _ in range(remaining):
                if not elapsed_ok():
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_made += 1
                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma *= 1.3
                else:
                    sigma *= 0.85
                sigma = np.clip(sigma, 1e-6, 5.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
