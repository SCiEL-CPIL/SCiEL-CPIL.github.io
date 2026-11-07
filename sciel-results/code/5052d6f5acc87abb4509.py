"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES with stagnation-based restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = np.full(dim, -5.0)
        self.ub = np.full(dim, 5.0)
        self.total_calls = 0

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        rng = np.random.default_rng(0)

        best_x = None
        best_f = np.inf
        calls = 0

        def clip(x):
            return np.clip(x, self.lb, self.ub)

        def try_call(x):
            nonlocal calls
            if calls >= budget:
                return None
            calls += 1
            try:
                return func(x)
            except Exception:
                return None

        # Multi-start CMA-ES
        # Each restart: run CMA-ES from a random start until stagnation or budget/deadline
        max_restarts = 20
        # Stagnation: no improvement in best_f over this many evaluations
        stagnation_window = 60
        # Min evaluations per restart before allowing a restart
        min_evals_per_restart = 40

        for restart in range(max_restarts):
            if calls >= budget or time.time() > deadline:
                break

            # Initialize CMA-ES state
            x_mean = rng.uniform(-4, 4, dim)  # minimizer lies in [-4,4]^5
            sigma = 1.5
            # CMA-ES parameters
            popsize = 4 + int(3 * np.log(dim))  # lambda
            popsize = max(popsize, 5)
            # Weights
            mu = popsize // 2
            weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / weights.sum()
            cc = (4.0 + mueff / dim) / (dim + 4.0 + 2.0 * mueff / dim)
            cs = (mueff + 2.0) / (dim + mueff + 5.0)
            c1 = (2.0 - mueff * (mueff - 2.0) / (dim * dim - 1.0)) / ((dim + 1.3) ** 2 + mueff)
            cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((dim + 2.0) ** 2 + mueff))
            # Covariance matrix
            C = np.eye(dim)
            # Evolution paths
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            # Cholesky of C
            try:
                C_sqrt = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C_sqrt = np.eye(dim)

            local_best_f = best_f
            local_stagnation = 0
            evals_this_restart = 0

            while calls < budget and time.time() < deadline:
                # Check deadline with safety margin
                if time.time() > deadline:
                    break

                # Generate population
                # If sigma too small, we may be stuck
                if sigma < 1e-12:
                    break

                # Sample offspring
                offspring = []
                offspring_f = []
                for i in range(popsize):
                    if calls >= budget or time.time() > deadline:
                        break
                    z = rng.standard_normal(dim)
                    y = x_mean + sigma * (C_sqrt @ z)
                    y = clip(y)
                    fy = try_call(y)
                    if fy is None:
                        continue
                    offspring.append(y)
                    offspring_f.append(fy)

                if len(offspring) == 0:
                    break

                evals_this_restart += len(offspring)

                # Sort by fitness
                order = np.argsort(offspring_f)
                sorted_offspring = [offspring[i] for i in order]
                sorted_f = [offspring_f[i] for i in order]

                # Update best
                if sorted_f[0] < local_best_f:
                    local_best_f = sorted_f[0]
                    local_stagnation = 0
                    if local_best_f < best_f:
                        best_f = local_best_f
                        best_x = sorted_offspring[0].copy()
                else:
                    local_stagnation += 1

                # Check stagnation
                if local_stagnation > stagnation_window and evals_this_restart > min_evals_per_restart:
                    break

                # Update CMA-ES parameters
                # Weighted mean
                x_new_mean = np.zeros(dim)
                for i in range(mu):
                    x_new_mean += weights[i] * sorted_offspring[i]

                # Update covariance
                # Evolve ps
                inv_sqrt_C = np.linalg.inv(C_sqrt)
                diff_mean = x_new_mean - x_mean
                ps = (1.0 - cs) * ps + np.sqrt(cs * (2.0 - cs) * mueff) * (inv_sqrt_C @ diff_mean / sigma)
                hsig = (np.linalg.norm(ps) / np.sqrt(1.0 - (1.0 - cs) ** (2 * (evals_this_restart + 1)))) \
                    < (1.4 + 2.0 / (dim + 1.0)) * cc
                # Update pc
                pc = (1.0 - cc) * pc + hsig * cc * diff_mean / sigma
                # Rank-one and rank-mu updates
                outer = np.outer(pc, pc)
                delta_h = (1.0 - hsig) * cc * (2.0 - cc)
                C = (1.0 + c1 * delta_h - c1) * C + c1 * outer
                # Rank-mu update
                rank_mu = np.zeros((dim, dim))
                for i in range(mu):
                    zi = inv_sqrt_C @ (sorted_offspring[i] - x_mean) / sigma
                    rank_mu += weights[i] * np.outer(zi, zi)
                C += cmu * rank_mu
                # Ensure symmetry and positive semi-definite
                C = (C + C.T) / 2.0
                # Adjust sigma
                chi_n = np.sqrt(dim) * (1.0 - 1.0 / (2.0 * dim) + 1.0 / (12.0 * dim ** 2))
                sigma = sigma * np.exp(cs / chi_n * (np.linalg.norm(ps) / chi_n - 1.0))
                sigma = min(sigma, 10.0)
                # Update mean
                x_mean = x_new_mean
                # Update Cholesky
                try:
                    C_sqrt = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # Regularize
                    C += 1e-6 * np.eye(dim)
                    try:
                        C_sqrt = np.linalg.cholesky(C)
                    except np.linalg.LinAlgError:
                        C = np.eye(dim)
                        C_sqrt = np.eye(dim)
                        pc = np.zeros(dim)
                        ps = np.zeros(dim)
                        sigma = 1.0

            # If we broke out of the inner loop due to stagnation or budget, continue to next restart
            if calls >= budget or time.time() > deadline:
                break

        # Ensure we have a valid result
        if best_x is None:
            # Fallback: single random point
            x = rng.uniform(-5, 5, dim)
            fx = try_call(x)
            if fx is not None:
                best_x = x
                best_f = fx
            else:
                best_x = np.zeros(dim)
                best_f = 0.0

        return best_x, best_f
# EVOLVE-BLOCK-END
