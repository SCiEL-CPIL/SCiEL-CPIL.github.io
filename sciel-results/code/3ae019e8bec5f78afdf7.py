"""Optimizer for BBOB f6 (Attractive sector) using Multi-Start Nelder-Mead and CMA-ES."""
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """
    Hybrid optimizer for f6:
    1. Diverse initial sampling.
    2. Multi-start Nelder-Mead for fast local convergence on top candidates.
    3. Final CMA-ES run for robust polishing.
    """

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

        # Track a few best points for multi-start
        # We will keep a list of (f, x) for top candidates
        candidates = []

        def eval_func(x):
            nonlocal best_x, best_f, calls, candidates
            x_clipped = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            f = float(func(x_clipped))
            calls += 1
            
            if f < best_f:
                best_f = f
                best_x = x_clipped.copy()
            
            # Maintain a small list of top candidates for multi-start
            # Simple insertion sort for small list
            candidates.append((f, x_clipped.copy()))
            candidates.sort(key=lambda item: item[0])
            if len(candidates) > 10:
                candidates.pop()
                
            return f

        def time_left():
            return deadline - time.time()

        # Phase 1: Initial Diverse Sampling
        # Use a mix of random and structured points to cover the space
        rng = np.random.default_rng(42)
        
        n_init = 30
        init_points = []
        
        # Origin
        init_points.append(np.zeros(dim))
        
        # Axis-aligned points (helps if the sector is aligned)
        for i in range(dim):
            for sign in [1, -1]:
                pt = np.zeros(dim)
                pt[i] = 3.5 * sign
                init_points.append(pt)
        
        # Random points
        while len(init_points) < n_init:
            init_points.append(rng.uniform(-4.5, 4.5, dim))
        
        init_points = np.array(init_points)
        
        # Evaluate initial points
        for i in range(len(init_points)):
            if time_left() < 5.0 or calls >= max_calls:
                break
            f = eval_func(init_points[i])
        
        if calls >= max_calls or time_left() < 15.0:
            return best_x, best_f

        # Phase 2: Multi-Start Nelder-Mead
        # Pick top 3 distinct candidates from the initial sampling
        # Ensure they are not too close to each other
        top_starts = []
        for f, x in candidates:
            # Check distance to existing starts
            is_distinct = True
            for _, x_prev in top_starts:
                if np.linalg.norm(x - x_prev) < 1.0:
                    is_distinct = False
                    break
            if is_distinct:
                top_starts.append((f, x))
            if len(top_starts) >= 3:
                break
        
        if not top_starts:
            top_starts.append((best_f, best_x))

        nm_budget_per_start = 40
        nm_starts_used = 0
        
        for f_start, x_start in top_starts:
            if time_left() < 10.0 or calls >= max_calls:
                break
            
            # Check if we have enough budget for a meaningful NM run
            remaining = max_calls - calls
            if remaining < nm_budget_per_start:
                nm_budget_per_start = remaining
            if nm_budget_per_start < 10:
                break

            try:
                # Nelder-Mead is efficient for 5D
                def objective(z):
                    return eval_func(z)
                
                res = minimize(
                    objective,
                    x_start,
                    method='Nelder-Mead',
                    options={
                        'maxiter': nm_budget_per_start,
                        'xatol': 1e-8,
                        'fatol': 1e-10,
                        'adaptive': True
                    }
                )
                nm_starts_used += 1
                # eval_func already updates best_x/best_f
            except Exception:
                pass

        if calls >= max_calls or time_left() < 15.0:
            return best_x, best_f

        # Phase 3: Final CMA-ES Run
        # Use the best point found so far as the starting point for a final CMA-ES
        # This helps if NM got stuck in a local minimum or needs finer tuning
        remaining_budget = max_calls - calls
        if remaining_budget > 30:
            # Allocate remaining budget to CMA-ES
            cma_budget = remaining_budget
            x_start = best_x.copy()
            
            # Adaptive sigma: if we are close to optimum, start small? 
            # But we don't know f_opt. Start with a moderate sigma.
            sigma_start = 0.5 
            
            def run_cmaes(x0, max_evals, sigma0):
                nonlocal calls, best_f, best_x
                
                if max_evals < 20:
                    return
                
                n = dim
                lambda_pop = 4 + int(3 * np.log(n))  # Population size ~ 13 for n=5
                mu = lambda_pop // 2  # Number of parents ~ 6
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
                    
                    # If we broke early, pad fitness with inf to allow sorting
                    if n_evals < lambda_pop:
                        fitness[n_evals:] = np.inf
                        offsprings[n_evals:] = best_x_in_cma # Dummy values, won't be used in update if sorted last

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
                    # Only use valid evaluated points for mean update
                    valid_indices = indices[:n_evals]
                    if len(valid_indices) < mu:
                         break
                         
                    # We need to be careful here. The standard CMA-ES update assumes a full population.
                    # If we ran out of budget mid-generation, we should probably stop.
                    if n_evals < lambda_pop:
                         break

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
                    if stagnant_generations > 15:
                        break
                    if sigma < 1e-12:
                        break
                
                return best_x_in_cma, best_in_cma

            run_cmaes(x_start, cma_budget, sigma_start)

        return best_x, best_f
