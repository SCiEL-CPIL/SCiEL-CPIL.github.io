"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """CMA-ES style optimizer with multi-start Nelder-Mead refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def _cma_es(self, func, x0, sigma0, max_evals, rng):
        """Simple CMA-ES implementation."""
        dim = self.dim
        n = dim
        popsize = max(4, int(4 + 3 * np.log(n)))  # lambda
        weights = np.log(popsize / 2) - np.arange(1, popsize + 1)
        weights /= weights.sum()
        mu = popsize // 2  # number of parents
        w_mu = weights[:mu]
        w_mu = w_mu / w_mu.sum()
        mu_eff = 1.0 / np.sum(w_mu ** 2)
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma

        x = x0.copy()
        sigma = sigma0
        B = np.eye(n)
        D = np.ones(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        cs = (n + 1) / 2
        hs = (1.1 + mu_eff / (n + 1)) * np.sqrt(
            (1 - np.prod(1 - c_c)) / (1 - np.prod((1 - c_c) ** (cs + 1)))
        )
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        evals_used = 0
        best_x = x0.copy()
        best_f = func(x0)
        evals_used += 1

        while evals_used < max_evals:
            # Generate population
            pop = []
            for i in range(popsize):
                z = rng.standard_normal(n)
                xi = x + sigma * (B @ (D * z))
                pop.append(xi)

            # Evaluate
            pop_f = []
            for xi in pop:
                if evals_used >= max_evals:
                    break
                fi = func(xi)
                pop_f.append(fi)
                evals_used += 1
                if fi < best_f:
                    best_f = fi
                    best_x = xi.copy()

            if len(pop_f) < popsize:
                break

            # Sort by fitness
            indices = np.argsort(pop_f)
            pop_sorted = [pop[i] for i in indices]
            pop_f_sorted = [pop_f[i] for i in indices]

            # Update mean
            x_old = x.copy()
            for i in range(mu):
                x += w_mu[i] * (pop_sorted[i] - x)

            # Update covariance matrix (rank-mu update)
            y = (x - x_old) / (sigma * np.sqrt(-2 * c_sigma * (1 - np.exp(-2 * c_sigma)))) if sigma > 0 else np.zeros(n)
            h_sigma = 1.0 if np.linalg.norm(y) / np.sqrt(1 - (1 - c_sigma) ** (evals_used / (n + 1))) < (1.4 + 2 / (n + 1)) * chi_n else 0.0
            ps = (1 - c_sigma) * ps + h_sigma * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y
            pc = (1 - c_c) * pc + h_sigma * c_c * (x - x_old) / sigma

            # Update B and D
            for i in range(mu):
                diff = (pop_sorted[i] - x_old) / sigma
                B, D = np.linalg.eigh(
                    B @ (np.outer(D, D) + (w_mu[i] / (1 - c_sigma)) * np.outer(diff, diff) - 
                        (c_c * (2 - c_c) / (1 - c_c)) * np.outer(pc, pc) * (1 - h_sigma) + 
                        h_sigma * c_sigma * (2 - c_sigma) * mu_eff * np.eye(n)) @ B.T
                )
            D = np.sqrt(np.clip(np.diag(D), 1e-10, None))

            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            sigma = max(sigma, 1e-10)

        return best_x, best_f

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 285  # safety margin before 300s kill

        # Track best
        best_x = None
        best_f = np.inf
        call_count = 0

        def tracked_func(x):
            nonlocal call_count, best_x, best_f
            if time.time() > deadline:
                raise TimeoutError()
            call_count += 1
            if call_count > budget:
                raise ValueError("Budget exceeded")
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        rng = np.random.default_rng(42)

        # Phase 1: CMA-ES with initial sigma covering the search space
        # Start from origin with large sigma
        try:
            x0 = np.zeros(dim)
            sigma0 = 4.0  # covers most of [-5,5]^5
            max_evals_phase1 = int(budget * 0.6)
            res_x, res_f = self._cma_es(tracked_func, x0, sigma0, max_evals_phase1, rng)
            if res_f < best_f:
                best_f = res_f
                best_x = res_x.copy()
        except (TimeoutError, ValueError):
            pass

        # Phase 2: Multi-start Nelder-Mead refinement
        if best_x is None:
            best_x = np.zeros(dim)

        if time.time() < deadline and call_count < budget:
            # Start from best and several perturbed points
            start_points = [best_x.copy()]
            for scale in [2.0, 1.0, 0.5, 0.2]:
                perturbation = rng.normal(0, scale, dim)
                sp = np.clip(best_x + perturbation, -5, 5)
                start_points.append(sp)

            for sp in start_points:
                if time.time() > deadline:
                    break
                if call_count >= budget:
                    break
                try:
                    nm_result = minimize(
                        tracked_func,
                        sp,
                        method='Nelder-Mead',
                        options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
                    )
                    if nm_result.fun < best_f:
                        best_f = nm_result.fun
                        best_x = nm_result.x.copy()
                except (TimeoutError, ValueError):
                    pass

        return best_x, best_f
# EVOLVE-BLOCK-END
