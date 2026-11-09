"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart CMA-ES with random initialization phase for f22."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 320.0  # Safety margin before 330s kill
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(0)

        def check_time():
            return (time.time() - start_time) < time_limit

        # ---------------- Phase 1: Random sampling ----------------
        # Spend ~50 evaluations doing pure random search to get a decent
        # starting point and avoid the worst local minima.
        n_random = min(50, budget)
        x_best = rng.uniform(-5, 5, dim)
        f_best = func(x_best)
        calls = 1

        for _ in range(n_random - 1):
            if not check_time():
                break
            x = rng.uniform(-5, 5, dim)
            f = func(x)
            calls += 1
            if f < f_best:
                f_best, x_best = f, x

        # ---------------- Phase 2: Multi-restart CMA-ES ----------------
        remaining = budget - calls
        if remaining <= 0:
            return x_best, f_best

        # Number of restarts: at least 3, at most 8, based on remaining budget
        # f22 is multimodal, so multiple restarts are beneficial.
        n_restarts = max(3, min(8, remaining // 50))
        budget_per_restart = max(30, remaining // n_restarts)

        for r in range(n_restarts):
            if not check_time():
                break
            if calls >= budget:
                break

            # Initialize CMA-ES near the current best with some perturbation
            # For the first restart, start near x_best; for subsequent ones,
            # use a fresh random point or significant perturbation to explore other basins
            if r == 0:
                x_mean = x_best.copy()
            else:
                # Perturb the best point significantly to escape the current basin
                # Alternatively, pick a new random point if we want more exploration
                # Let's stick to perturbation of best to maintain some continuity
                # but with a larger step to jump basins
                x_mean = np.clip(x_best + 2.0 * rng.standard_normal(dim), -5, 5)

            sigma = 1.0  # Initial step size
            lambda_ = max(4, dim + int(3 * np.log(dim)))  # Population size
            mu = lambda_ // 2  # Number of parents

            # Weights for recombination
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights ** 2)

            # CMA-ES parameters
            c_sigma = (mueff + 2) / ((dim + 2) ** 2 + mueff)
            d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
            c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 2 / mueff) / ((dim + 2) ** 2 + mueff))
            chiN = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

            # State variables
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            C = np.eye(dim)
            C_sqrt = np.eye(dim)
            inv_C_sqrt = np.eye(dim)
            step_size = sigma
            hsig = (1 + (1 + 3 / dim) * c_sigma) * chiN

            # Cholesky update helper
            def update_C_sqrt():
                nonlocal C_sqrt, inv_C_sqrt
                try:
                    C_sqrt = np.linalg.cholesky(C)
                    inv_C_sqrt = np.linalg.inv(C_sqrt)
                except np.linalg.LinAlgError:
                    # Fallback: reset to identity if matrix becomes singular
                    C_sqrt = np.eye(dim)
                    inv_C_sqrt = np.eye(dim)
                    # Also reset C to identity to stabilize
                    C = np.eye(dim)

            # Run CMA-ES for this restart
            max_iter = budget_per_restart // lambda_ + 1
            for it in range(max_iter):
                if not check_time():
                    break
                if calls >= budget:
                    break

                # Generate and evaluate the entire population
                offspring = []
                for i in range(lambda_):
                    if not check_time() or calls >= budget:
                        break
                    z = rng.standard_normal(dim)
                    x_child = x_mean + step_size * (C_sqrt @ z)
                    x_child = np.clip(x_child, -5, 5)
                    f_child = func(x_child)
                    calls += 1
                    offspring.append((f_child, x_child))

                    if f_child < f_best:
                        f_best, x_best = f_child, x_child

                if len(offspring) < lambda_:
                    break

                # Sort by fitness
                offspring.sort(key=lambda t: t[0])
                
                # Select parents
                x_parents = np.array([offspring[i][1] for i in range(mu)])
                
                # Compute new mean
                x_new_mean = np.sum(weights[:, None] * x_parents, axis=0)
                
                # Compute new covariance matrix
                d = x_new_mean - x_mean
                y = x_parents - x_mean
                y_weighted = (weights[:, None] * y.T).T
                
                # Rank-mu update
                C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(d, d) + (C - np.diag(np.diag(C)))) + c_mu * (y_weighted @ y_weighted.T)
                
                # Ensure C is symmetric
                C = (C + C.T) / 2
                
                # Update evolution paths
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (inv_C_sqrt @ d) / step_size
                
                hsig = (1 + (1 + 3 / dim) * c_sigma) * chiN
                hsig_condition = np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (it + 1))) < (1.4 + 2 / (dim + 1)) * hsig
                
                if hsig_condition:
                    pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mueff) * d / step_size
                else:
                    pc = (1 - c_c) * pc
                
                # Rank-one update for covariance (already included in the formula above? 
                # Standard CMA-ES: C = (1 - c1 - cmu)C + c1(pc pc^T + (1-hsig)cc(2-cc)C) + cmu * sum...
                # The formula used above: C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(d, d) + (C - np.diag(np.diag(C)))) + c_mu * (y_weighted @ y_weighted.T)
                # This is a simplified version. Let's use the standard robust update.
                
                # Correct Standard CMA-ES Covariance Update:
                # C = (1 - c1 - cmu) * C 
                #     + c1 * (pc @ pc.T + (1 - hsig) * cc * (2 - cc) * C) 
                #     + cmu * (y_weighted @ y_weighted.T)
                
                # Let's re-calculate C properly
                C = (1 - c_1 - c_mu) * C
                C += c_1 * (np.outer(pc, pc) + (1 - hsig_condition) * c_c * (2 - c_c) * C)
                C += c_mu * (y_weighted @ y_weighted.T)
                
                # Symmetrize
                C = (C + C.T) / 2

                # Update step size (sigma)
                step_size *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chiN - 1))
                
                # Clamp sigma to prevent explosion or collapse
                step_size = np.clip(step_size, 1e-12, 1e12)

                # Update mean
                x_mean = x_new_mean

                # Update Cholesky decomposition
                update_C_sqrt()

                # Early stopping criteria: if sigma is very small or no improvement in last few generations
                # We'll just rely on the budget and time limit for simplicity and robustness

        return x_best, f_best
# EVOLVE-BLOCK-END
