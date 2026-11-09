"""Model-and-optimize optimizer for BBOB f11 (Discus)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer with restarts for BBOB f11 (Discus)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cmaes(self, func, x0, sigma0, budget, rng, evals_ref, deadline):
        """Run CMA-ES from initial point x0 with step size sigma0.
        Returns (best_x, best_f, evals_used)."""
        dim = self.dim
        N = max(4, int(4 + 3 * np.log(dim)))  # population size
        mu = N // 2  # number of parents
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)

        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        # Initialize
        x_mean = np.array(x0, dtype=float)
        sigma = sigma0
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        cs = 1.0 / np.sqrt(dim)
        norm_s = np.sqrt(dim)

        best_x = x_mean.copy()
        best_f = func(np.clip(best_x, -5, 5))
        evals = 1

        while evals < budget:
            if time.time() > deadline:
                break

            # Generate population
            offspring = []
            offspring_f = []
            
            # Compute Cholesky factorization once per generation
            try:
                B = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C is not positive definite, regularize
                C = C + 1e-6 * np.eye(dim)
                try:
                    B = np.linalg.cholesky(C)
                except:
                    break

            for i in range(N):
                if evals >= budget:
                    break
                z = rng.standard_normal(dim)
                y = x_mean + sigma * (B @ z)
                y = np.clip(y, -5, 5)
                fy = func(y)
                evals += 1
                offspring.append(y)
                offspring_f.append(fy)

            if len(offspring) < mu:
                break

            # Sort by fitness
            sorted_idx = np.argsort(offspring_f)
            sorted_offspring = [offspring[i] for i in sorted_idx[:mu]]
            sorted_offspring_f = [offspring_f[i] for i in sorted_idx[:mu]]

            # Update best
            if sorted_offspring_f[0] < best_f:
                best_f = sorted_offspring_f[0]
                best_x = sorted_offspring[0].copy()

            # Weighted mean of selected
            x_new = np.sum([weights[i] * sorted_offspring[i] for i in range(mu)], axis=0)

            # Update evolution paths
            diff = (x_new - x_mean) / sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * diff
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (evals + 1) // N))) / norm_s
            hsig = min(hsig, 1.4 + 2 / (dim + 1))
            pc = (1 - c_c) * pc + hsig * c_c * diff
            C = (1 - c_1 - c_mu) * C + c_1 * np.outer(pc, pc) + c_mu * sum(
                weights[i] * np.outer(sorted_offspring[i] - x_mean, sorted_offspring[i] - x_mean) / sigma ** 2
                for i in range(mu)
            )

            # Symmetrize C
            C = (C + C.T) / 2

            # Update step size
            sigma *= np.exp(c_sigma / norm_s * (np.linalg.norm(ps) / norm_s - 1))

            x_mean = x_new

            # Check for stagnation
            if sigma < 1e-10:
                break

        return best_x, best_f, evals

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 320  # safety margin before 330s kill

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Phase 1: Initial exploration to find a good starting point
        n_init = min(10, budget)
        init_points = [np.zeros(dim)]
        for _ in range(n_init - 1):
            init_points.append(rng.uniform(-4, 4, dim))

        init_vals = [func(np.clip(p, -5, 5)) for p in init_points]
        evals = n_init

        best_idx = int(np.argmin(init_vals))
        best_x = init_points[best_idx].copy()
        best_f = init_vals[best_idx]

        # Estimate initial sigma from spread of good points
        sorted_idx = np.argsort(init_vals)
        top_k = min(5, len(init_vals))
        top_points = np.array([init_points[i] for i in sorted_idx[:top_k]])
        if len(top_points) > 1:
            distances = np.linalg.norm(top_points[1:] - top_points[0], axis=1)
            sigma0 = max(0.1, np.mean(distances) * 0.5)
        else:
            sigma0 = 1.0

        # Phase 2: Single CMA-ES run with remaining budget
        remaining_budget = budget - evals
        if remaining_budget > 0:
            new_best_x, new_best_f, evals_used = self._cmaes(
                func, best_x, sigma0, remaining_budget, rng, evals, deadline
            )
            evals += evals_used

            if new_best_f < best_f:
                best_f = new_best_f
                best_x = new_best_x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
