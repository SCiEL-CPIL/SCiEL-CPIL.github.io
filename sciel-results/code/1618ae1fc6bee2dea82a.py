"""Rewritten optimizer for BBOB-style f11 (Discus)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES with restarts for a box-constrained black-box problem."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        total_budget = self.budget
        deadline = time.time() + 280.0  # safety margin before 300s kill

        # Track global best
        global_best_x = None
        global_best_f = np.inf

        # CMA-ES parameters
        lambda_pop = max(4, int(np.ceil(3 + 3 * np.log(dim))))  # population size
        # Standard CMA-ES step sizes
        c_sigma = (lambda_pop + 2) / (dim + lambda_pop + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((lambda_pop - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + lambda_pop / dim) / (dim + 4 + 2 * lambda_pop / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + lambda_pop)
        c_mu = min(1 - c_1, 2 * ((mu_eff + 0.5) / ((dim + 2) ** 2 + lambda_pop)) - c_1)
        mu_eff = lambda_pop / 2.0
        chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))

        def cma_es(x0, fx0, budget, rng):
            """Run CMA-ES from x0 with fx0 for up to budget function evaluations."""
            nonlocal global_best_x, global_best_f

            n = dim
            lam = lambda_pop
            mu = lam // 2  # number of recombination weights
            # Weights
            w = np.log(np.arange(1, mu + 1)) - np.log(mu)
            w /= w.sum()
            mu_eff = 1.0 / np.sum(w ** 2)

            # Recompute c_1, c_mu with actual mu_eff
            c_1 = 2 / ((n + 1.3) ** 2 + lam)
            c_mu = min(1 - c_1, 2 * ((mu_eff + 0.5) / ((n + 2) ** 2 + lam)) - c_1)

            # Initial parameters
            sigma = 1.0
            pc = np.zeros(n)
            ps = np.zeros(n)
            C = np.eye(n)
            # Precompute sqrt of C (identity initially)
            B = np.eye(n)
            D = np.ones(n)
            # Update frequency for C
            nu = 1.0 / (c_1 + c_mu)

            # Population
            x = x0.copy()
            fx = fx0

            # Track local best
            local_best_x = x0.copy()
            local_best_f = fx0

            # Precompute some constants
            alpha_sigma = (mu_eff + 2) / (n + mu_eff)

            # Main loop
            calls = 1  # already evaluated x0
            while calls < budget:
                # Check deadline
                if time.time() > deadline:
                    break

                # Generate population
                offspring = np.empty((lam, n))
                # Update C via eigen decomposition periodically
                # For simplicity, use Cholesky or just B*D
                # We'll use the standard CMA-ES update

                for i in range(lam):
                    z = rng.standard_normal(n)
                    offspring[i] = x + sigma * (B @ (D * z))

                # Clip to box
                offspring = np.clip(offspring, -5, 5)

                # Evaluate
                fitness = np.empty(lam)
                for i in range(lam):
                    if calls >= budget or time.time() > deadline:
                        break
                    fitness[i] = func(offspring[i])
                    calls += 1

                # Sort by fitness
                order = np.argsort(fitness[:calls - 1]) if calls - 1 < lam else np.argsort(fitness)
                n_eval = min(lam, calls - 1)
                order = np.argsort(fitness[:n_eval])

                # Update mean
                x_new = np.sum(w[:, None] * offspring[order[:mu]], axis=0)
                x_new = np.clip(x_new, -5, 5)

                # Update sigma
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (B.T @ (D * ((x_new - x) / sigma)))
                sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1))
                sigma = max(sigma, 1e-12)

                # Update evolution path
                pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * (B.T @ (D * ((x_new - x) / sigma)))

                # Update covariance matrix
                # hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls + 1) / (lam * c_sigma))) < (1 + 2 / (n + 1)) * chi_n)
                # For simplicity, skip hsig correction
                delta_hsig = 0.0
                hsig = 1.0  # simplified

                rank_mu = np.sum(w[:, None] * (w[:, None] * (offspring[order[:mu]] - x_new).T), axis=1) @ (offspring[order[:mu]] - x_new).T
                rank_one = (1 + 2 * (hsig - 1)) * np.outer(pc, pc)
                C = (1 + c_1 * (rank_one - C) + c_mu * (rank_mu - C)) * C

                # Ensure symmetry
                C = (C + C.T) / 2

                # Eigendecomposition (periodically)
                # Do it every nu steps for efficiency, but for small n it's fine
                evals, evecs = np.linalg.eigh(C)
                evals = np.maximum(evals, 1e-20)
                D = np.sqrt(evals)
                B = evecs

                # Update position
                x = x_new

                # Track local best
                best_idx = order[0]
                if fitness[best_idx] < local_best_f:
                    local_best_f = fitness[best_idx]
                    local_best_x = offspring[best_idx].copy()

                # Update global best
                if local_best_f < global_best_f:
                    global_best_f = local_best_f
                    global_best_x = local_best_x.copy()

            return local_best_x, local_best_f

        # Multi-start strategy
        rng = np.random.default_rng(42)

        # First start: origin
        x0 = np.zeros(dim)
        fx0 = func(x0)
        if fx0 < global_best_f:
            global_best_f = fx0
            global_best_x = x0.copy()

        # Budget allocation for first run
        first_budget = min(total_budget, 150)
        _, _ = cma_es(x0, fx0, first_budget, rng)

        # Check if we've used up budget
        calls_used = first_budget
        if calls_used >= total_budget or time.time() > deadline:
            return global_best_x, global_best_f

        # Additional restarts from promising locations
        # Use a few random starts and one at the current best
        remaining = total_budget - calls_used
        num_restarts = min(5, remaining // 30)  # at least 30 calls per restart

        for r in range(num_restarts):
            if time.time() > deadline:
                break
            if remaining <= 0:
                break

            # Choose starting point
            if r == 0:
                # Start from current global best with small perturbation
                x_start = global_best_x + 0.1 * rng.standard_normal(dim)
                x_start = np.clip(x_start, -5, 5)
            else:
                # Random start
                x_start = rng.uniform(-5, 5, dim)

            fx_start = func(x_start)
            remaining -= 1
            if fx_start < global_best_f:
                global_best_f = fx_start
                global_best_x = x_start.copy()

            # Budget for this restart
            restart_budget = min(remaining, max(30, remaining // (num_restarts - r)))
            _, _ = cma_es(x_start, fx_start, restart_budget, rng)
            remaining -= (restart_budget - 1)  # subtract calls used (excluding the initial eval)

        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
