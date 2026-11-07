"""CMA-ES based optimizer for BBOB f22 (Gallagher 21 peaks)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """CMA-ES with L-BFGS-B refinement for BBOB f22."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 285.0  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        # Evaluation counter
        eval_count = 0

        def wrapped_func(x):
            nonlocal eval_count
            if eval_count >= budget:
                return float('inf')
            eval_count += 1
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            return float(func(x))

        best_x = np.zeros(dim)
        best_f = float('inf')

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        rng = np.random.default_rng(42)

        # Phase 1: Initial diverse sampling
        n_initial = 25
        for _ in range(n_initial):
            if eval_count >= budget - 100:
                break
            if time.time() > deadline:
                break
            pt = rng.uniform(-4.5, 4.5, dim)
            f_val = wrapped_func(pt)
            update_best(pt, f_val)

        # Phase 2: CMA-ES implementation
        # Parameters
        pop_size = max(4, int(4 + 3 * np.log(dim)))
        sigma = 2.0  # Initial step size
        max_iter = 50

        # CMA-ES state
        mean = best_x.copy()
        C = np.eye(dim)  # Covariance matrix
        C_inv = np.eye(dim)  # Inverse covariance
        # Precompute square root of C
        eigvals, eigvecs = np.linalg.eigh(C)
        C_sqrt = eigvecs @ np.diag(np.sqrt(np.maximum(eigvals, 1e-10))) @ eigvecs.T
        C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(np.maximum(eigvals, 1e-10))) @ eigvecs.T

        # Adaptation parameters
        mu = pop_size // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / (weights ** 2).sum()
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        chi_N = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))

        # Path variables
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # Stagnation counter
        stagnation_count = 0
        best_f_cma = best_f
        best_x_cma = best_x.copy()

        for iteration in range(max_iter):
            if eval_count >= budget - 50:
                break
            if time.time() > deadline:
                break

            # Generate population
            population = []
            fitness = []

            for i in range(pop_size):
                if eval_count >= budget - 50:
                    break
                if time.time() > deadline:
                    break
                z = rng.standard_normal(dim)
                # Transform using Cholesky-like decomposition
                x = mean + sigma * (C_sqrt @ z)
                x = np.clip(x, -5.0, 5.0)
                f_val = wrapped_func(x)
                population.append(x)
                fitness.append(f_val)

            # Sort by fitness
            indices = np.argsort(fitness)
            population = [population[i] for i in indices]
            fitness = [fitness[i] for i in indices]

            # Update best
            if fitness[0] < best_f_cma:
                best_f_cma = fitness[0]
                best_x_cma = population[0].copy()
                stagnation_count = 0
            else:
                stagnation_count += 1

            # Update best global
            if fitness[0] < best_f:
                best_f = fitness[0]
                best_x = population[0].copy()

            # Update mean (recombination)
            old_mean = mean.copy()
            for i in range(mu):
                mean += weights[i] * population[i]

            # Update step size (sigma)
            if np.linalg.norm(C_inv_sqrt @ (mean - old_mean)) > 1e-12:
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(
                    c_sigma * (2 - c_sigma) * mu_eff) * C_inv_sqrt @ (mean - old_mean) / sigma
            sigma *= np.exp(c_sigma / chi_N * (np.linalg.norm(p_sigma) / chi_N - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Update covariance matrix
            if np.linalg.norm(mean - old_mean) > 1e-12:
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (mean - old_mean) / sigma

            delta_sig = np.sqrt(c_c * (2 - c_c) * mu_eff) * (np.linalg.norm(p_c) / chi_N - 1)
            C = ((1 - c_1 - c_mu + c_1 * delta_sig ** 2 + c_1 * (c_c * (2 - c_c)) * 
                  (np.linalg.norm(p_c) ** 2 / chi_N ** 2 - 1)) * C +
                 c_1 * np.outer(p_c, p_c))

            # Rank-mu update
            for i in range(mu):
                diff = (population[i] - old_mean) / sigma
                C += c_mu * np.outer(diff, diff)

            # Ensure positive semi-definiteness
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-10)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            C_inv = eigvecs @ np.diag(1.0 / eigvals) @ eigvecs.T
            C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

            # Restart if stagnating
            if stagnation_count > 15:
                # Reset CMA-ES around best point with larger sigma
                mean = best_x_cma.copy()
                C = np.eye(dim)
                C_inv = np.eye(dim)
                eigvals, eigvecs = np.linalg.eigh(C)
                C_sqrt = eigvecs @ np.diag(np.sqrt(np.maximum(eigvals, 1e-10))) @ eigvecs.T
                C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(np.maximum(eigvals, 1e-10))) @ eigvecs.T
                sigma = max(sigma * 2.0, 1.0)
                p_sigma = np.zeros(dim)
                p_c = np.zeros(dim)
                stagnation_count = 0

        # Phase 3: L-BFGS-B refinement from multiple starts
        start_points = [best_x.copy()]

        # Add local perturbations at different scales
        for scale in [0.5, 0.2, 0.1, 0.05]:
            for _ in range(2):
                sp = best_x + rng.normal(0, scale, dim)
                sp = np.clip(sp, -5.0, 5.0)
                start_points.append(sp)

        # Remove duplicates
        unique_starts = []
        for sp in start_points:
            is_dup = False
            for us in unique_starts:
                if np.linalg.norm(sp - us) < 0.05:
                    is_dup = True
                    break
            if not is_dup:
                unique_starts.append(sp)

        # Run L-BFGS-B from each starting point
        for sp in unique_starts:
            if eval_count >= budget - 10:
                break
            if time.time() > deadline:
                break

            try:
                result = minimize(
                    wrapped_func,
                    sp,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={'maxiter': 30, 'maxls': 10, 'ftol': 1e-12, 'gtol': 1e-7}
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass

        # Phase 4: Nelder-Mead polish
        if eval_count < budget - 5 and time.time() < deadline:
            try:
                result = minimize(
                    wrapped_func,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 30, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass

        # Phase 5: Use remaining budget for local random search
        while eval_count < budget and time.time() < deadline:
            x = best_x + rng.normal(0, 0.005, dim)
            x = np.clip(x, -5.0, 5.0)
            f_val = wrapped_func(x)
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
