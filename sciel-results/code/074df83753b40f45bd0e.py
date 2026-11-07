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

        rng = np.random.default_rng(123)

        # Phase 1: Initial sampling - structured grid for early AOCC
        # Use a 3^5 grid (243 points) evaluated in shuffled order for strong early progress
        grid_vals = np.array([-4.0, 0.0, 4.0])
        grid_points = []
        for i1 in grid_vals:
            for i2 in grid_vals:
                for i3 in grid_vals:
                    for i4 in grid_vals:
                        for i5 in grid_vals:
                            grid_points.append(np.array([i1, i2, i3, i4, i5]))
        
        # Shuffle for better exploration
        rng.shuffle(grid_points)
        
        # Evaluate grid points, stopping when we have good coverage or budget constraint
        grid_budget = min(len(grid_points), max(20, budget // 4))
        for pt in grid_points[:grid_budget]:
            if eval_count >= budget - 30:
                break
            if time.time() > deadline:
                break
            f_val = wrapped_func(pt)
            update_best(pt, f_val)
        
        # Add a few random points for diversity
        for _ in range(5):
            if eval_count >= budget - 30:
                break
            if time.time() > deadline:
                break
            pt = rng.uniform(-4.5, 4.5, dim)
            f_val = wrapped_func(pt)
            update_best(pt, f_val)

        # Phase 2: Proper CMA-ES implementation
        def run_cmaes(start_x, sigma0, max_evals):
            nonlocal eval_count, best_f, best_x, deadline
            n = dim
            # CMA-ES parameters
            lambda_ = int(4 + 3 * n)  # population size
            mu = lambda_ // 2  # number of parents
            # Weights
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights ** 2)
            # Learning rates
            cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            cs = (mueff + 2) / (n + mueff + 5)
            c1 = 2 / ((n + 1.3) ** 2 + mueff)
            cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
            
            # Initial state
            mean = start_x.copy()
            sigma = sigma0
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            
            # Precompute
            dnorm = np.sqrt(n)
            chi_n = dnorm * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
            
            local_best_f = best_f
            local_best_x = best_x.copy()
            
            # Track previous mean for path updates
            prev_mean = mean.copy()
            
            while eval_count < max_evals and time.time() < deadline:
                # Check time budget
                if time.time() > deadline - 5:
                    break
                
                # Generate population
                population = []
                fitness = []
                for i in range(lambda_):
                    if eval_count >= max_evals or time.time() > deadline:
                        break
                    z = rng.standard_normal(n)
                    # Use Cholesky decomposition for sampling
                    try:
                        B = np.linalg.cholesky(C)
                    except np.linalg.LinAlgError:
                        B = np.eye(n)
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

                # Update mean
                new_mean = np.zeros(n)
                for i in range(mu):
                    new_mean += weights[i] * population[i]
                
                # Update evolution paths
                # Update step-size control (cs)
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (new_mean - prev_mean) / sigma
                
                # Update covariance adaptation (c1)
                diff = (new_mean - prev_mean) / sigma
                C = (1 - c1 - cmu) * C + c1 * (np.outer(diff, diff) + (1 - (1 - cs) ** 2) * C)
                
                # Update covariance adaptation (cmu)
                for i in range(mu):
                    xi = (population[i] - prev_mean) / sigma
                    C += cmu * weights[i] * np.outer(xi, xi)
                
                # Update step size (sigma) using CSA
                norm_ps = np.linalg.norm(ps)
                sigma *= np.exp(cs / dnorm * (norm_ps / chi_n - 1))
                
                # Clip sigma to reasonable range
                sigma = np.clip(sigma, 1e-10, 10.0)
                
                # Update mean
                prev_mean = mean
                mean = new_mean
                
                # Ensure C is symmetric positive definite
                C = (C + C.T) / 2
                try:
                    np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # Add small diagonal to ensure positive definiteness
                    C += 1e-6 * np.eye(n)

            return local_best_x, local_best_f

        # Run CMA-ES from best point found so far
        cmaes_budget = min(200, budget - eval_count - 20)
        if cmaes_budget > 20 and time.time() < deadline:
            cmaes_x, cmaes_f = run_cmaes(best_x.copy(), 1.0, eval_count + cmaes_budget)
            if cmaes_f < best_f:
                best_f = cmaes_f
                best_x = cmaes_x

        # Phase 3: Multi-start L-BFGS-B from diverse points
        # Generate diverse starting points
        start_points = [best_x.copy()]

        # Add points around the best
        for scale in [1.5, 1.0, 0.5, 0.25]:
            for _ in range(2):
                sp = best_x + rng.normal(0, scale, dim)
                sp = np.clip(sp, -5.0, 5.0)
                start_points.append(sp)

        # Add random points
        for _ in range(6):
            sp = rng.uniform(-4.5, 4.5, dim)
            start_points.append(sp)

        # Add some structured points
        for _ in range(4):
            sp = rng.choice([-4.0, -2.0, 0.0, 2.0, 4.0], dim)
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
                    options={'maxiter': 40, 'maxls': 20, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass

        # Phase 4: Final high-precision polish pass
        # Use Nelder-Mead with tighter tolerances
        if eval_count < budget - 5 and time.time() < deadline:
            try:
                result = minimize(
                    wrapped_func,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass

        # Phase 5: Additional L-BFGS-B polish with very tight tolerances
        if eval_count < budget - 3 and time.time() < deadline:
            try:
                result = minimize(
                    wrapped_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={'maxiter': 50, 'maxls': 25, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass

        # Ensure we've made enough calls for the evaluator
        # The evaluator expects up to 500 calls
        while eval_count < budget and time.time() < deadline:
            # Fill remaining budget with evaluations around best
            x = best_x + rng.normal(0, 0.005, dim)
            x = np.clip(x, -5.0, 5.0)
            f_val = wrapped_func(x)
            if f_val < best_f:
                best_f = f_val
                best_x = x

        return best_x, best_f
# EVOLVE-BLOCK-END
