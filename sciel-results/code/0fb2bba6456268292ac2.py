"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES with adaptive population size and restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # safety margin before 300s limit

        best_x = None
        best_f = np.inf

        # Phase 1: Broad exploration with multiple restarts
        # Use a larger population for global exploration
        # Phase 2: Refinement around the best found point

        def cma_es(x0, sigma0, budget_phase, rng, deadline):
            """Run CMA-ES from x0 with initial sigma for budget_phase evaluations."""
            dim = len(x0)
            pop_size = max(4, int(4 + 3 * np.log(dim)))
            # CMA-ES parameters
            mu = pop_size // 2  # number of parents
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)

            # Learning rates
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2.0 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

            # State
            x_mean = x0.copy().astype(float)
            sigma = sigma0
            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (20 * dim ** 2))

            evaluations = 0
            best_local_x = x_mean.copy()
            best_local_f = func(x_mean)
            evaluations += 1

            if best_local_f < best_f:
                best_f = best_local_f
                best_x = x_mean.copy()

            while evaluations < budget_phase:
                if time.time() > deadline:
                    break

                # Generate population
                offspring = []
                for i in range(pop_size):
                    B = np.linalg.cholesky(C)
                    z = rng.standard_normal(dim)
                    y = x_mean + sigma * B @ z
                    y = np.clip(y, -5, 5)
                    offspring.append(y)

                # Evaluate
                offspring_f = [func(y) for y in offspring]
                evaluations += pop_size

                # Update best
                for i in range(pop_size):
                    if offspring_f[i] < best_local_f:
                        best_local_f = offspring_f[i]
                        best_local_x = offspring[i].copy()
                        if best_local_f < best_f:
                            best_f = best_local_f
                            best_x = offspring[i].copy()

                # Sort offspring
                idx = np.argsort(offspring_f)
                offspring_sorted = [offspring[i] for i in idx]
                offspring_f_sorted = [offspring_f[i] for i in idx]

                # Update mean
                x_old = x_mean.copy()
                x_mean = sum(weights[i] * offspring_sorted[i] for i in range(mu))

                # Update covariance matrix
                # ps update
                inv_sqrtC = np.linalg.inv(np.linalg.cholesky(C))
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * inv_sqrtC @ (x_mean - x_old) / sigma
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (evaluations + 1) / (dim * c_sigma)))) < (1.4 + 2 / (dim + 1)) * chi_n

                # pc update
                pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mu_eff) * inv_sqrtC @ (x_mean - x_old) / sigma

                # C update
                delta_d = (1 + 2 * (hsig - 1) * c_c) if hsig else 1
                C = (1 - c_1 - c_mu) * C
                C += c_1 * (delta_d * C + (pc + pc) - np.outer(pc, pc))
                for i in range(mu):
                    v = (offspring_sorted[i] - x_old) / sigma
                    C += c_mu * weights[i] * np.outer(v, v)

                # Ensure symmetry
                C = (C + C.T) / 2

                # Update sigma
                sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1))
                sigma = max(sigma, 1e-10)
                sigma = min(sigma, 20.0)

                # Termination check
                if sigma < 1e-12 or np.all(np.abs(x_mean - x_old) < 1e-12):
                    break

            return best_local_x, best_local_f, evaluations

        # Multi-start strategy
        # First, do a coarse grid/random sampling to find good starting points
        n_starts = 5
        starts = []
        f_starts = []

        # Some structured starts
        starts.append(np.zeros(dim))
        starts.append(rng.uniform(-4, 4, dim))
        starts.append(rng.uniform(-3, 3, dim))
        starts.append(np.array([4.0, -4.0, 0.0, 4.0, -4.0]))
        starts.append(np.array([-4.0, 4.0, 0.0, -4.0, 4.0]))

        # Evaluate starts
        for s in starts:
            s = np.clip(s, -5, 5)
            fs = func(s)
            f_starts.append(fs)
            if fs < best_f:
                best_f = fs
                best_x = s.copy()

        # Sort starts by fitness
        start_order = np.argsort(f_starts)

        # Allocate budget to each start, giving more to better starts
        remaining_budget = budget - len(starts)
        # Weighted allocation
        weights_start = np.array([1.0 / (1 + i) for i in range(n_starts)])
        weights_start = weights_start / weights_start.sum()
        budgets = (remaining_budget * weights_start).astype(int)
        # Ensure at least some budget for each
        budgets = np.maximum(budgets, 10)
        # Adjust to fit remaining budget
        total_alloc = budgets.sum()
        if total_alloc > remaining_budget:
            scale = remaining_budget / total_alloc
            budgets = (budgets * scale).astype(int)
            budgets = np.maximum(budgets, 5)

        # Run CMA-ES from each start
        for i in start_order:
            if time.time() > deadline:
                break
            x0 = starts[i].copy()
            sigma0 = 1.5
            b = budgets[i]
            if b <= 0:
                continue
            _, _, _ = cma_es(x0, sigma0, b, rng, deadline)

        # Final refinement: run a local search around the best point
        if best_x is not None and time.time() < deadline:
            # Use Nelder-Mead or similar for final polish
            from scipy.optimize import minimize
            try:
                res = minimize(func, best_x, method='Nelder-Mead',
                               options={'maxiter': 50, 'xatol': 1e-6, 'fatol': 1e-8,
                                        'adaptive': True})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
