"""Optimizer for BBOB f6 (Attractive sector) using multi-start CMA-ES."""
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Multi-start CMA-ES optimizer for f6 with adaptive restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin before 300s limit

        best_x = np.zeros(dim)
        best_f = np.inf
        calls = 0
        max_calls = budget
        # Track top-2 points for multi-start
        second_best_x = None
        second_best_f = np.inf

        def eval_func(x):
            nonlocal best_x, best_f, calls, second_best_x, second_best_f
            x_clipped = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            f = float(func(x_clipped))
            calls += 1
            if f < best_f:
                if best_f != np.inf:
                    second_best_x = best_x.copy()
                    second_best_f = best_f
                best_f = f
                best_x = x_clipped.copy()
            elif f < second_best_f:
                second_best_f = f
                second_best_x = x_clipped.copy()
            return f

        def time_left():
            return deadline - time.time()

        # Phase 1: Initial sampling to find a good basin
        rng = np.random.default_rng(42)
        
        # Sample diverse initial points with stratified coverage
        n_init = 50
        init_points = []
        # Include origin and axis-aligned points
        init_points.append(np.zeros(dim))
        for d in range(dim):
            for s in [-3.0, 3.0]:
                p = np.zeros(dim)
                p[d] = s
                init_points.append(p)
        
        # Add random points
        while len(init_points) < n_init:
            init_points.append(rng.uniform(-4.5, 4.5, dim))
        
        init_points = np.array(init_points[:n_init])
        
        # Evaluate initial points
        for i in range(len(init_points)):
            if time_left() < 5.0 or calls >= max_calls:
                break
            f = eval_func(init_points[i])
        
        if calls >= max_calls or time_left() < 10.0:
            return best_x, best_f

        # Phase 2: CMA-ES optimization with multi-start
        def cmaes(x0, sigma0, max_evals):
            nonlocal calls, best_f, best_x
            
            if max_evals < 20:
                return
            
            n = dim
            lambda_pop = 4 + int(3 * np.log(n))  # Population size
            mu = lambda_pop // 2  # Number of parents
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mueff = 1.0 / np.sum(weights ** 2)
            
            # Strategy parameters
            cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            cs = (mueff + 2) / (n + mueff + 5)
            c1 = 2 / ((n + 1.3) ** 2 + mueff)
            cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
            
            # Initialize
            x_mean = x0.copy()
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            sigma = sigma0
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
            
            # Pre-compute square root of C
            def compute_sqrt_C(C):
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-10, None)
                return eigvecs @ np.diag(np.sqrt(eigvals))
            
            sqrt_C = compute_sqrt_C(C)
            
            n_evals = 0
            stagnant_generations = 0
            best_in_cma = best_f
            best_x_in_cma = best_x.copy()
            
            while n_evals < max_evals and time_left() > 5.0:
                # Generate population
                z = rng.standard_normal((lambda_pop, n))
                offsprings = x_mean + sigma * (z @ sqrt_C.T)
                
                # Evaluate offsprings
                fitness = np.zeros(lambda_pop)
                for i in range(lambda_pop):
                    if n_evals >= max_evals or time_left() < 5.0:
                        break
                    fitness[i] = eval_func(offsprings[i])
                    n_evals += 1
                
                # Sort by fitness
                indices = np.argsort(fitness)
                x_sorted = offsprings[indices]
                f_sorted = fitness[indices]
                
                # Update best
                if f_sorted[0] < best_f:
                    best_f = f_sorted[0]
                    best_x = x_sorted[0].copy()
                
                if f_sorted[0] < best_in_cma:
                    best_in_cma = f_sorted[0]
                    best_x_in_cma = x_sorted[0].copy()
                    stagnant_generations = 0
                else:
                    stagnant_generations += 1
                
                # Update mean
                x_mean_new = x_mean + sigma * sum(weights[i] * x_sorted[i] for i in range(mu))
                
                # Update evolution paths
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / sigma * sqrt_C.T @ (x_mean_new - x_mean)
                if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (n_evals + 1) / lambda_pop)) < (c1 + 1) * chi_n + 0.3:
                    pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * mueff) * (x_mean_new - x_mean) / sigma
                
                # Update covariance matrix
                diff = x_sorted[:mu] - x_mean
                C = (1 + c1 * cmu - c1) * C + c1 * np.outer(pc, pc) + cmu * (diff * weights[:, None]).T @ diff
                
                # Ensure positive definiteness
                C = (C + C.T) / 2
                eigvals = np.linalg.eigvalsh(C)
                if np.min(eigvals) < 1e-10:
                    C += 1e-10 * np.eye(n)
                
                # Update step size
                sigma *= np.exp(cs / chi_n * (np.linalg.norm(ps) / chi_n - 1))
                sigma = np.clip(sigma, 1e-10, 1e10)
                
                x_mean = x_mean_new
                sqrt_C = compute_sqrt_C(C)
                
                # Termination criteria
                if stagnant_generations > 20:
                    break
                if sigma < 1e-12:
                    break
            
            return best_x_in_cma, best_in_cma

        # Start CMA-ES from the best point found so far
        remaining_budget = max_calls - calls
        if remaining_budget > 50:
            # Use 70% of remaining budget for first CMA-ES
            cma_budget = int(remaining_budget * 0.7)
            if cma_budget > 30:
                x_start = best_x.copy()
                sigma_start = 1.0
                cmaes(x_start, sigma_start, cma_budget)
            
            # Second CMA-ES from second-best point if available
            remaining_budget = max_calls - calls
            if remaining_budget > 30 and second_best_x is not None:
                cma_budget2 = int(remaining_budget * 0.6)
                if cma_budget2 > 20:
                    x_start2 = second_best_x.copy()
                    sigma_start2 = 0.5
                    cmaes(x_start2, sigma_start2, cma_budget2)
        
        # Phase 3: Local polish with Nelder-Mead on the best point
        remaining_budget = max_calls - calls
        if remaining_budget > 20 and time_left() > 10.0:
            try:
                def objective(z):
                    return eval_func(z)
                
                res = minimize(
                    objective,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': remaining_budget,
                        'xatol': 1e-8,
                        'fatol': 1e-10,
                        'adaptive': True
                    }
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5.0, 5.0).copy()
            except Exception:
                pass
        
        return best_x, best_f
