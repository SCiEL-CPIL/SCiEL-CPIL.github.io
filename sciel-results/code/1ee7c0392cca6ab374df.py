"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES with time guard for BBOB f6."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s kill
        dim = self.dim
        budget = self.budget

        # Track best solution and value globally
        best_x = None
        best_f = np.inf
        calls_made = 0

        def eval_point(x):
            nonlocal calls_made, best_x, best_f
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Initial exploration: sample several random points
        rng = np.random.default_rng(42)
        n_init = 20
        for i in range(n_init):
            x = rng.uniform(-5, 5, dim)
            fx = eval_point(x)
            if time.time() > deadline:
                break

        # Multi-start CMA-ES
        # Use the best few initial points as seeds for local refinement
        # We'll run several CMA-ES instances from different starting points

        # Collect initial points sorted by fitness
        # Since we only track best, let's do a small grid/random set
        # Actually, let's just use the best point and a few others

        # Better approach: run CMA-ES from the best initial point,
        # and if it stagnates, restart from a new random point

        # CMA-ES parameters
        def cma_es(x0, max_evals, seed):
            nonlocal calls_made, best_x, best_f, deadline

            rng_local = np.random.default_rng(seed)
            n = dim
            x = x0.copy()
            f_x = eval_point(x)
            if time.time() > deadline:
                return x, f_x

            # CMA-ES parameters
            sigma = 1.0
            # Population size
            lambda_ = 4 + int(3 * np.log(n))
            # Step size control
            cs = 1 / (n + 4)
            ds = 1 + 2 * max(0, np.sqrt((lambda_ / 2 - 1) / (n + 1)) - 1)
            # Evolution path
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)
            # Covariance matrix
            C = np.eye(n)
            # Inverse square root of C (for efficient sampling)
            C_sqrt_inv = np.linalg.inv(np.linalg.cholesky(C).T)
            # Weights
            mu = lambda_ // 2
            weights = np.log(lambda_ / 2 + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1 / np.sum(weights ** 2)
            # Update rules
            c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
            c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            c_cov = (2 + 2 * mu_eff / n) / (n + 1 + 3 * mu_eff / n)
            # Chi distribution
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            evals = 1
            while evals < max_evals and calls_made < budget:
                if time.time() > deadline:
                    break

                # Sample population
                ys = []
                for i in range(lambda_):
                    z = rng_local.standard_normal(n)
                    y = x + sigma * C_sqrt_inv @ z
                    ys.append(y)
                ys = np.array(ys)

                # Evaluate
                fitnesses = np.array([eval_point(y) for y in ys])
                evals += lambda_

                # Sort
                order = np.argsort(fitnesses)
                ys = ys[order]
                fitnesses = fitnesses[order]

                # Mean
                x_new = np.sum(weights[:, None] * ys[:mu], axis=0)

                # Update CMA-ES parameters
                y_vec = (x_new - x) / sigma
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_vec
                h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (evals // lambda_)))) < (1.4 + 2 / (n + 1)) * chi_n
                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_vec

                delta_h = (1 - h_sigma) * c_c * (2 - c_c)
                C = (1 + c_cov * delta_h - c_cov) * C + c_cov * np.outer(y_vec, y_vec) * mu_eff + c_cov * (1 - h_sigma) * (np.outer(p_c, p_c) - C) * mu_eff

                # Ensure positive definite
                try:
                    L = np.linalg.cholesky(C)
                    C_sqrt_inv = np.linalg.inv(L.T)
                except np.linalg.LinAlgError:
                    # Regularize
                    C = C + 1e-8 * np.eye(n)
                    L = np.linalg.cholesky(C)
                    C_sqrt_inv = np.linalg.inv(L.T)

                # Update sigma
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))

                # Clip to box
                x_new = np.clip(x_new, -5, 5)
                x = x_new

                # Check for stagnation
                if evals % (lambda_ * 10) == 0:
                    # If no improvement in recent generations, break
                    if np.all(fitnesses[:mu] >= np.min(fitnesses) - 1e-10):
                        pass  # Continue, but we'll check externally

            return x, f_x

        # Run CMA-ES from best initial point
        # First, find a good starting point
        # Use the best_x we have
        if best_x is not None:
            remaining = budget - calls_made
            if remaining > 0:
                # Allocate budget to CMA-ES
                x_opt, f_opt = cma_es(best_x, remaining, seed=123)

        # If we have budget left, try another restart from a different region
        if calls_made < budget and time.time() < deadline:
            # Try a few more random restarts
            for restart_seed in [7, 13, 21]:
                if calls_made >= budget or time.time() > deadline:
                    break
                x0 = rng.uniform(-5, 5, dim)
                remaining = budget - calls_made
                if remaining < 50:
                    break
                x_opt, f_opt = cma_es(x0, remaining, seed=restart_seed)

        return best_x, best_f
# EVOLVE-BLOCK-END
