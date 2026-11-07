"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """CMA-ES with adaptive restarts for Rosenbrock."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        
        best_x = None
        best_f = float('inf')
        eval_count = 0
        dim = self.dim
        budget = self.budget

        def wrapped_func(x):
            nonlocal eval_count, best_x, best_f
            if eval_count >= budget:
                raise Exception("Budget exceeded")
            if time.time() > deadline:
                raise Exception("Time exceeded")
            
            x = np.asarray(x, dtype=float)
            fx = func(x)
            eval_count += 1
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            return fx

        def cma_es(start_x, sigma, max_iter, max_evals):
            """Run CMA-ES from a starting point."""
            nonlocal eval_count, best_x, best_f, deadline
            
            # CMA-ES parameters
            pop_size = max(4, int(4 + 3 * np.log(dim)))
            # Initialize population
            x_mean = start_x.copy()
            C = np.eye(dim)
            C_inv = np.eye(dim)
            C_sqrt = np.eye(dim)
            C_sqrt_inv = np.eye(dim)
            
            # Update rates
            mu = pop_size // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)
            
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2.0 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
            
            # Memory
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))
            
            # Initial population
            pop = []
            fvals = []
            
            for i in range(pop_size):
                z = np.random.randn(dim)
                x_i = x_mean + sigma * C_sqrt @ z
                # Clip to bounds
                x_i = np.clip(x_i, -5.0, 5.0)
                pop.append(x_i)
                fvals.append(wrapped_func(x_i))
            
            # Sort by fitness
            order = np.argsort(fvals)
            pop = [pop[i] for i in order]
            fvals = [fvals[i] for i in order]
            
            # Initial mean
            x_mean = np.sum(np.array(pop[:mu]) * weights[:, None], axis=0)
            
            iteration = 0
            no_improve = 0
            
            while iteration < max_iter and eval_count < max_evals:
                if time.time() > deadline:
                    break
                
                iteration += 1
                
                # Update evolution paths
                d = np.sqrt(c_sigma * (5 + c_sigma))
                p_sigma = (1 - c_sigma) * p_sigma + d * np.sqrt(mu_eff) * weights @ (np.array(pop[:mu]) - x_mean) / (sigma + 1e-32)
                
                norm_p_sigma = np.linalg.norm(p_sigma)
                if norm_p_sigma > 1.4 * (1 + c_sigma) * chi_n:
                    c_sigma = min(c_sigma, 1.0)
                
                # Update covariance matrix
                hs = (norm_p_sigma / np.sqrt(1 - (1 - c_sigma) ** (2 * (iteration + 1) / (dim * c_sigma)))) / chi_n - 1 + c_sigma
                p_c = (1 - c_c) * p_c + hs * np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_mean - x_mean) / (sigma + 1e-32)
                
                # Update covariance matrix
                delta_h = (1 - c_c) ** 2 - (1 - c_c)
                C = (1 + c_1 + c_mu - c_1 - c_c * (1 + c_1 - c_c) + delta_h) * C
                C += c_1 * (np.outer(p_c, p_c) - C)
                C += c_mu * np.sum(weights[:, None] * np.outer(np.array(pop[:mu]) - x_mean, (np.array(pop[:mu]) - x_mean)), axis=0) / (sigma ** 2 + 1e-32)
                
                # Ensure positive definiteness
                C = 0.5 * (C + C.T)
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                C = 0.5 * (C + C.T)
                
                # Update square root
                try:
                    C_sqrt, _ = np.linalg.sqrtm(C)
                    C_sqrt = 0.5 * (C_sqrt + C_sqrt.T)
                    C_sqrt_inv = np.linalg.inv(C_sqrt)
                except:
                    C_sqrt = np.eye(dim)
                    C_sqrt_inv = np.eye(dim)
                
                # Update sigma
                sigma *= np.exp(c_sigma / chi_n * (norm_p_sigma / chi_n - 1))
                sigma = max(sigma, 1e-12)
                sigma = min(sigma, 10.0)
                
                # Generate new population
                new_pop = []
                new_fvals = []
                
                for i in range(pop_size):
                    z = np.random.randn(dim)
                    x_i = x_mean + sigma * C_sqrt @ z
                    x_i = np.clip(x_i, -5.0, 5.0)
                    new_pop.append(x_i)
                    new_fvals.append(wrapped_func(x_i))
                    if eval_count >= max_evals or time.time() > deadline:
                        break
                
                if len(new_pop) < pop_size:
                    break
                
                # Sort
                order = np.argsort(new_fvals)
                new_pop = [new_pop[i] for i in order]
                new_fvals = [new_fvals[i] for i in order]
                
                # Update mean
                x_mean = np.sum(np.array(new_pop[:mu]) * weights[:, None], axis=0)
                
                # Check for improvement
                if new_fvals[0] < best_f - 1e-10:
                    no_improve = 0
                else:
                    no_improve += 1
                
                if no_improve > 10:
                    break
                
                pop = new_pop
                fvals = new_fvals
            
            return x_mean, fvals[0] if fvals else float('inf')

        # Phase 1: Initial sampling with Sobol + strategic points
        try:
            from scipy.stats import qmc
            n_scout = 30
            sampler = qmc.Sobol(d=dim, scramble=True)
            samples = sampler.random(n=n_scout)
            samples = samples * 10.0 - 5.0
            scout_points = list(samples)
        except ImportError:
            n_scout = 30
            scout_points = np.random.uniform(-5, 5, size=(n_scout, dim)).tolist()

        # Add strategic points
        scout_points.append(np.zeros(dim))
        scout_points.append(np.ones(dim))
        scout_points.append(-np.ones(dim))
        scout_points.append(2.0 * np.ones(dim))
        scout_points.append(-2.0 * np.ones(dim))

        for p in scout_points:
            if time.time() > deadline or eval_count >= budget:
                break
            try:
                fp = wrapped_func(p)
            except Exception:
                break

        if best_x is None:
            best_x = np.zeros(dim)
            try:
                best_f = func(best_x)
                eval_count += 1
            except Exception:
                pass

        # Phase 2: CMA-ES with restarts
        remaining = budget - eval_count
        if remaining > 50:
            # Run CMA-ES from best point
            try:
                x_cma, f_cma = cma_es(best_x, 1.0, max_iter=50, max_evals=budget)
                if f_cma < best_f:
                    best_f = f_cma
                    best_x = x_cma.copy()
            except Exception:
                pass
        
        # Phase 3: Local polish with L-BFGS-B if budget remains
        remaining = budget - eval_count
        if remaining > 20 and best_x is not None:
            try:
                res = minimize(
                    wrapped_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={'maxiter': 50, 'maxfun': remaining, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass
        
        # Phase 4: Final Nelder-Mead polish
        remaining = budget - eval_count
        if remaining > 10 and best_x is not None:
            try:
                res = minimize(
                    wrapped_func,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 100, 'maxfev': remaining, 'xatol': 1e-10, 'fatol': 1e-12}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
