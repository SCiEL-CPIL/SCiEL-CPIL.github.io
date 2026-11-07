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

        # Phase 1: Initial diverse sampling - Sobol-like space-filling
        n_initial = 30
        for i in range(n_initial):
            if eval_count >= budget - 200:
                break
            if time.time() > deadline:
                break
            # Use a mix of random and structured points
            if i == 0:
                pt = np.zeros(dim)
            elif i < 5:
                # Corner-like points
                pt = rng.choice([-4.0, -2.0, 0.0, 2.0, 4.0], dim)
            else:
                # Random points
                pt = rng.uniform(-4.5, 4.5, dim)
            f_val = wrapped_func(pt)
            update_best(pt, f_val)

        # Phase 2: CMA-ES optimization with proper implementation
        pop_size = 12  # Smaller population for better exploration
        sigma = 1.5
        max_iter = 60
        stagnation_limit = 12

        def run_cma_es(start_x, sigma0, max_evals_remaining):
            nonlocal eval_count, best_f, best_x, deadline
            n = dim
            # CMA-ES parameters
            lambda_ = pop_size
            mu = max(2, lambda_ // 2)
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights ** 2)

            cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            cs = (mueff + 2) / (n + mueff + 5)
            c1 = 2 / ((n + 1.3) ** 2 + mueff)
            cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))

            mean = start_x.copy()
            sigma = sigma0
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)

            # Precompute Cholesky
            try:
                B = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                B = np.eye(n)

            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
            local_best_f = best_f
            local_best_x = best_x.copy()
            stagnation = 0

            for iteration in range(max_iter):
                if eval_count >= budget - 100:
                    break
                if time.time() > deadline:
                    break

                # Generate population
                population = []
                fitness = []
                for i in range(lambda_):
                    if eval_count >= budget - 100:
                        break
                    if time.time() > deadline:
                        break
                    z = rng.standard_normal(n)
                    x_i = mean + sigma * (B @ z)
                    x_i = np.clip(x_i, -5.0, 5.0)
                    f_i = wrapped_func(x_i)
                    if f_i < local_best_f:
                        local_best_f = f_i
                        local_best_x = x_i.copy()
                    population.append(x_i)
                    fitness.append(f_i)

                if len(population) < mu:
                    break

                # Sort by fitness
                idx = np.argsort(fitness)
                population = [population[i] for i in idx]
                fitness = [fitness[i] for i in idx]

                # Check stagnation
                if fitness[0] >= local_best_f * (1 + 1e-10):
                    stagnation += 1
                else:
                    stagnation = 0

                # Update mean
                new_mean = np.zeros(n)
                for i in range(mu):
                    new_mean += weights[i] * population[i]
                mean = new_mean

                # Update evolution paths
                try:
                    B = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    B = np.eye(n)

                # ps update
                diff = (mean - start_x) / sigma if sigma > 1e-15 else np.zeros(n)
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (B.T @ diff)

                # sigma update
                sigma *= np.exp(cs / (1 + 2 * max(0, (mueff - 1) / (n + 1))) *
                              (np.linalg.norm(ps) / chi_n - 1))
                sigma = np.clip(sigma, 1e-8, 10.0)

                # pc update
                pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * mueff) * (B.T @ diff)

                # C update
                delta_h = (1 + cc) * np.linalg.norm(pc) / chi_n <= 1.4 + 2 / (n + 1)
                rank_one = np.outer(pc, pc)
                rank_mu = sum(weights[i] * np.outer(population[i] - start_x,
                                                    population[i] - start_x) / sigma ** 2
                              for i in range(mu))
                C = (1 + c1 * delta_h - c1 - cmu) * C + c1 * (rank_one +
                     (1 - delta_h) * (np.linalg.norm(pc) ** 2 / mueff - 1) * C) + cmu * rank_mu

                # Regularize C
                C = (C + C.T) / 2
                eigenvalues, eigenvectors = np.linalg.eigh(C)
                eigenvalues = np.clip(eigenvalues, 1e-12, None)
                C = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
                try:
                    B = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    B = np.eye(n)

                # Restart if stagnated
                if stagnation >= stagnation_limit:
                    break

            return local_best_x, local_best_f

        # Run CMA-ES from best point
        best_x, best_f = run_cma_es(best_x, 1.5, budget - 100)

        # Phase 3: Multi-start L-BFGS-B refinement
        if eval_count < budget - 20 and time.time() < deadline:
            start_points = [best_x.copy()]

            # Add points around the best
            for scale in [1.0, 0.5, 0.25]:
                for _ in range(2):
                    sp = best_x + rng.normal(0, scale, dim)
                    sp = np.clip(sp, -5.0, 5.0)
                    start_points.append(sp)

            # Add random points
            for _ in range(6):
                sp = rng.uniform(-4.5, 4.5, dim)
                start_points.append(sp)

            # Remove duplicates
            unique_starts = []
            for sp in start_points:
                is_dup = False
                for us in unique_starts:
                    if np.linalg.norm(sp - us) < 0.1:
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
                        options={'maxiter': 25, 'maxls': 15, 'ftol': 1e-12, 'gtol': 1e-7}
                    )
                    if result.fun < best_f:
                        best_f = result.fun
                        best_x = result.x.copy()
                except Exception:
                    pass

        # Phase 4: Final refinement with Nelder-Mead if we have budget left
        if eval_count < budget - 5 and time.time() < deadline:
            try:
                result = minimize(
                    wrapped_func,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 40, 'xatol': 1e-6, 'fatol': 1e-8}
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass

        # Phase 5: Use remaining budget for local random search
        while eval_count < budget and time.time() < deadline:
            x = best_x + rng.normal(0, 0.01, dim)
            x = np.clip(x, -5.0, 5.0)
            f_val = wrapped_func(x)
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
