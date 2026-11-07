"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Enhanced CMA-ES with restarts and local refinement for rotated Rosenbrock in 5D."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t0 = time.time()
        deadline = t0 + 310.0  # safety margin before 330s limit

        rng = np.random.default_rng(42)

        best_x = None
        best_f = float('inf')
        calls = 0

        def eval_point(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return None
            x_clipped = np.clip(x, -5, 5)
            f = func(x_clipped)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x_clipped.copy()
            return f

        def cma_es(m_init, sigma_init, max_calls, rng_local):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return

            # CMA-ES parameters for 5D
            lam = 10  # population size
            sigma = sigma_init
            m = m_init.copy()
            C = np.eye(dim)
            ps = np.zeros(dim)
            pc = np.zeros(dim)

            # Parameters
            d_sigma = (np.sqrt(lam + 1) - 1) / (dim + 2) + 2 / (dim + lam - 2)
            mu = lam // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1 / np.sum(weights ** 2)
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / dim) / ((dim + 2) ** 2 + mu_eff))

            # Eigen-decomposition
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-20)
            C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

            local_best_f = best_f
            local_best_x = best_x.copy() if best_x is not None else m.copy()
            stall_count = 0
            stall_threshold = 20  # generations without improvement

            gen = 0
            while calls < budget and (budget - calls) >= lam:
                if time.time() > deadline:
                    break

                # Generate population
                population = []
                fitnesses = []
                for i in range(lam):
                    z = rng_local.standard_normal(dim)
                    y = m + sigma * (C_sqrt @ z)
                    y = np.clip(y, -5, 5)
                    fy = eval_point(y)
                    if fy is None:
                        break
                    population.append(y)
                    fitnesses.append(fy)
                    if fy < local_best_f:
                        local_best_f = fy
                        local_best_x = y.copy()
                        stall_count = 0

                if len(population) < 2:
                    break

                # Sort by fitness
                idx = np.argsort(fitnesses)
                population = [population[i] for i in idx]
                fitnesses = [fitnesses[i] for i in idx]

                # Update mean
                new_m = np.sum(weights[:, None] * np.array(population[:mu]), axis=0)

                # Check for stall
                if local_best_f >= best_f - 1e-6:
                    stall_count += 1
                else:
                    stall_count = 0

                # Update evolution paths
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_inv_sqrt @ (new_m - m) / sigma
                h_sigma = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1)))) / np.sqrt(mu_eff)
                if h_sigma < 1.4 + 2 / (dim + 1):
                    pc = (1 - c_c) * pc + c_c * (new_m - m) / sigma
                    C = (1 + c_c * (2 - c_c) * mu_eff - c_1) * C + c_1 * (pc @ pc - C)
                    C += c_mu * np.sum(weights[:mu, None] * np.array(population[:mu] - m)[:, None] @ (population[:mu] - m), axis=0) / sigma ** 2
                else:
                    C = (1 - c_1) * C + c_1 * (C + pc @ pc)

                # Update sigma
                sigma *= np.exp(d_sigma * (np.linalg.norm(ps) / np.sqrt(mu_eff) - 1))
                sigma = np.clip(sigma, 1e-10, 10.0)

                # Update eigen-decomposition periodically
                if gen % 5 == 0:
                    C = 0.5 * (C + C.T)
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-20)
                    C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
                    C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

                m = new_m
                gen += 1

                # Restart if stalled
                if stall_count > stall_threshold and (budget - calls) > 3 * lam:
                    # Perturb the mean
                    m = local_best_x + 0.5 * rng_local.standard_normal(dim)
                    m = np.clip(m, -5, 5)
                    sigma = max(0.5, sigma * 0.5)
                    C = np.eye(dim)
                    ps = np.zeros(dim)
                    pc = np.zeros(dim)
                    eigvals, eigvecs = np.linalg.eigh(C)
                    C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
                    C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
                    stall_count = 0

        # Phase 1: Global search with multiple restarts
        # Start with several diverse initial points
        n_starts = 5
        starts_per = max(1, budget // (n_starts + 1))

        # First, do a quick random sampling to find good regions
        n_sample = min(20, budget // 5)
        for _ in range(n_sample):
            if calls >= budget:
                break
            x = rng.uniform(-4, 4, size=dim)
            eval_point(x)

        # Now do CMA-ES from the best point found so far
        if best_x is not None:
            cma_es(best_x.copy(), 1.0, budget, rng)

        # Phase 2: Local refinement with scipy
        if best_x is not None and calls < budget - 10:
            try:
                def wrapper(x):
                    if calls >= budget:
                        return best_f
                    x_clipped = np.clip(x, -5, 5)
                    f = func(x_clipped)
                    calls += 1
                    return f

                # Use L-BFGS-B for local refinement
                res = minimize(
                    wrapper,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': max(1, (budget - calls) // 2), 'ftol': 1e-12, 'gtol': 1e-12}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5, 5)
            except Exception:
                pass

        # Phase 3: More CMA-ES refinement if budget remains
        if best_x is not None and calls < budget - 5 * 10:
            cma_es(best_x.copy(), 0.1, budget, rng)

        # Final local refinement
        if best_x is not None and calls < budget - 5:
            try:
                def wrapper(x):
                    if calls >= budget:
                        return best_f
                    x_clipped = np.clip(x, -5, 5)
                    f = func(x_clipped)
                    calls += 1
                    return f

                res = minimize(
                    wrapper,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': max(1, (budget - calls)), 'ftol': 1e-14, 'gtol': 1e-14}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5, 5)
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
