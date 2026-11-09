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
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2.0 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        # Step size control
        chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))

        x_mean = np.array(x0, dtype=float)
        sigma = sigma0
        C = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        best_x = x_mean.copy()
        best_f = float('inf')
        evals = 0

        no_improve = 0
        stall_threshold = 20  # generations without improvement before restart

        while evals < budget and time.time() < deadline:
            # Generate population
            population = []
            for i in range(N):
                if evals >= budget:
                    break
                z = rng.standard_normal(dim)
                y = x_mean + sigma * (C @ z)
                y = np.clip(y, -5, 5)
                fy = func(y)
                evals += 1
                population.append((y, fy))

            if len(population) < mu:
                break

            # Sort by fitness
            population.sort(key=lambda t: t[1])
            offspring = population[:mu]

            # Weighted mean of selected
            new_x_mean = np.sum([weights[i] * offspring[i][0] for i in range(mu)], axis=0)

            # Update step size (CSA)
            old_sigma = sigma
            sigma = sigma * np.exp(c_sigma / chi_n * (np.linalg.norm(C @ (new_x_mean - x_mean)) / sigma - chi_n))
            sigma = np.clip(sigma, 1e-10, 1e3)

            # Update evolution path p_sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (new_x_mean - x_mean)

            # Update evolution path p_c
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (new_x_mean - x_mean) / sigma

            # Update covariance matrix C
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (evals // N + 1))) <
                       (1.4 + 2 / (dim + 1)) * chi_n) if evals > 0 else 1
            delta_h_sigma = (1 - h_sigma) * c_c * (2 - c_c)

            C = (1 + c_1 + c_mu - c_1 - delta_h_sigma) * C
            C += c_1 * (np.outer(p_c, p_c) - C)
            for i in range(mu):
                y_i = (offspring[i][0] - x_mean) / sigma
                C += c_mu * weights[i] * (np.outer(y_i, y_i) - C)

            # Ensure C is positive definite
            C = (C + C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            C = (C + C.T) / 2

            x_mean = new_x_mean

            # Track best
            for y, fy in population:
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()

            # Check for stall
            if best_f == float('inf'):
                break
            no_improve += 1
            if no_improve >= stall_threshold:
                break

        return best_x, best_f, evals

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 310  # safety margin before 330s kill

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Phase 1: Quick random search to find a good starting point
        n_init = min(20, budget // 10)
        init_points = []
        init_vals = []
        for _ in range(n_init):
            x = rng.uniform(-4, 4, dim)
            fx = func(x)
            init_points.append(x)
            init_vals.append(fx)

        if not init_vals:
            return np.zeros(dim), 0.0

        best_idx = int(np.argmin(init_vals))
        best_x = np.array(init_points[best_idx])
        best_f = init_vals[best_idx]
        evals = n_init

        # Estimate initial sigma from spread of good points
        sorted_idx = np.argsort(init_vals)
        top_k = min(5, len(init_vals))
        top_points = np.array([init_points[i] for i in sorted_idx[:top_k]])
        if len(top_points) > 1:
            distances = np.linalg.norm(top_points[1:] - top_points[0], axis=1)
            sigma0 = max(0.2, np.mean(distances) * 0.3)
        else:
            sigma0 = 1.0

        # Phase 2: CMA-ES with restarts
        total_budget_remaining = budget - evals
        max_restarts = 3
        restarts_used = 0

        while evals < budget and time.time() < deadline:
            # Allocate budget for this CMA-ES run
            if restarts_used < max_restarts:
                run_budget = min(total_budget_remaining, max(10, (budget - evals) // (max_restarts - restarts_used + 1)))
            else:
                run_budget = budget - evals

            x0 = best_x.copy()
            sigma0_current = sigma0

            # Run CMA-ES
            new_best_x, new_best_f, evals_used = self._cmaes(
                func, x0, sigma0_current, run_budget, rng, evals, deadline
            )

            evals += evals_used

            if new_best_f < best_f:
                best_x = new_best_x
                best_f = new_best_f
                # Shrink sigma for next restart
                sigma0 *= 0.5
            else:
                # No improvement; try a different starting point
                if restarts_used < max_restarts:
                    # Pick a random point from initial exploration or near current best
                    if rng.random() < 0.5 and init_points:
                        idx = int(rng.integers(len(init_points)))
                        x0 = np.array(init_points[idx])
                    else:
                        x0 = best_x + 0.5 * sigma0 * rng.standard_normal(dim)
                        x0 = np.clip(x0, -5, 5)
                    sigma0 = max(0.1, sigma0 * 1.5)
                else:
                    break

            restarts_used += 1

            # Check if we've made significant progress
            if best_f < -50:  # likely close to optimum for f11
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
