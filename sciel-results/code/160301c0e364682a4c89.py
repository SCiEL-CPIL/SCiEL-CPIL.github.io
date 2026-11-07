"""Optimizer for BBOB f7 (Step ellipsoidal) using CMA-ES global search + BFGS refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """
    Hybrid Optimizer: CMA-ES for global search, BFGS for local refinement.
    Designed for BBOB f7 (Step Ellipsoidal) in low dimensions.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: 330s kill, aim for 320s
        deadline = start_time + 320.0

        dim = self.dim
        budget = self.budget

        best_x = np.zeros(dim)
        best_f = np.inf
        calls = 0
        rng = np.random.default_rng(42)

        def _eval(x):
            """Wrapper to count calls and track best."""
            nonlocal calls, best_x, best_f
            if time.time() > deadline:
                return np.inf
            if calls >= budget:
                return np.inf

            # Ensure x is a valid numpy array of the correct shape
            x = np.asarray(x, dtype=np.float64).flatten()
            if x.shape != (dim,):
                return np.inf

            fx = func(x)
            calls += 1

            if np.isfinite(fx) and fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # 1. Initial Evaluation
        _eval(best_x)

        if calls >= budget or time.time() > deadline:
            return best_x, best_f

        # 2. CMA-ES Phase
        # Allocate ~70% of the budget to global search
        cma_budget_target = int(budget * 0.7)

        # CMA-ES Parameters
        lam = 12  # Population size
        mu = lam // 2
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1 / np.sum(weights ** 2)

        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = (2 * (mu_eff - 2 + 2 / mu_eff)) / ((dim + 2) ** 2 + mu_eff)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma

        # Initial state
        mean = best_x.copy()
        sigma = 2.0
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        C = np.eye(dim)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        while calls < cma_budget_target and time.time() < deadline:
            # Generate population
            pop = []
            fvals = []

            for _ in range(lam):
                z = rng.standard_normal(dim)
                # Use eigen decomposition for sampling from C (dim=5 is fast)
                val, vec = np.linalg.eigh(C)
                val = np.clip(val, 1e-10, None)
                sqrt_val = np.sqrt(val)
                x_new = mean + sigma * (vec @ (z * sqrt_val))
                x_new = np.clip(x_new, -5, 5)

                fx = _eval(x_new)
                pop.append(x_new)
                fvals.append(fx)

                if calls >= cma_budget_target or time.time() > deadline:
                    break

            if calls >= cma_budget_target or time.time() > deadline:
                break

            # If not all population members were evaluated, fill with inf or stop
            if len(pop) < lam:
                # Pad with worst possible value to maintain array sizes if needed,
                # but simpler to just break if we ran out of budget mid-generation
                break

            fvals = np.array(fvals)
            order = np.argsort(fvals)
            sorted_pop = np.array([pop[i] for i in order])
            sorted_fvals = fvals[order]

            # Update mean
            new_mean = np.sum(weights[:, np.newaxis] * sorted_pop[:mu], axis=0)

            # Update evolution paths
            val, vec = np.linalg.eigh(C)
            val = np.clip(val, 1e-10, None)
            inv_sqrt_C = vec @ np.diag(1.0 / np.sqrt(val)) @ vec.T

            diff = (new_mean - mean) / sigma
            pc = (1 - c_c) * pc + c_c * np.sqrt(mu_eff) * (inv_sqrt_C @ diff)
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (inv_sqrt_C @ diff)

            # Update sigma
            norm_ps = np.linalg.norm(ps)
            delta_sigma = (norm_ps / chi_n - 1) * d_sigma
            sigma *= np.exp(c_sigma * delta_sigma)
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Update C
            y = (sorted_pop[:mu] - mean) / sigma
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                rank_mu += weights[i] * np.outer(y[i], y[i])

            decay = 1 - c_1 - c_mu
            rank_one = np.outer(pc, pc) + (delta_sigma * c_sigma * (2 - c_sigma)) * C
            C = decay * C + c_1 * rank_one + c_mu * rank_mu
            C = (C + C.T) / 2

            mean = new_mean

            # Stop if convergence is reached or sigma is tiny
            if sigma < 1e-8:
                break

        # 3. Local Refinement Phase (BFGS)
        remaining_budget = budget - calls
        if remaining_budget > 10 and time.time() < deadline - 5:
            try:
                # Use the best point found so far as the starting point for local search
                x0_local = best_x.copy()

                # BFGS is derivative-free and efficient for low dimensions
                # maxfev limits the number of function evaluations
                result = minimize(
                    _eval,
                    x0_local,
                    method='BFGS',
                    options={
                        'maxiter': int(remaining_budget / 2),  # Approximate iterations
                        'maxfev': remaining_budget,
                        'gtol': 1e-8,
                        'disp': False
                    }
                )

                # Check if the result improved on best_f
                # Note: _eval updates best_x and best_f globally
                # If _eval hit the deadline or budget, it returns inf, but best_x is still valid
            except Exception:
                pass

        return best_x, best_f

# EVOLVE-BLOCK-END
