"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """CMA-ES with adaptive restarts and boundary handling for Rosenbrock."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        
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
            """Run CMA-ES from a starting point with correct updates."""
            nonlocal eval_count, best_x, best_f, deadline
            
            pop_size = max(4, int(4 + 3 * np.log(dim)))
            mu = pop_size // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)
            
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2.0 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
            
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))
            
            x_mean = start_x.copy().astype(float)
            x_mean_prev = x_mean.copy()
            C = np.eye(dim)
            L = np.eye(dim)  # Cholesky factor: C = L @ L.T
            
            # Initial population
            pop = []
            fvals = []
            for i in range(pop_size):
                z = np.random.randn(dim)
                x_i = x_mean + sigma * L @ z
                x_i = np.clip(x_i, -5.0, 5.0)
                pop.append(x_i)
                fvals.append(wrapped_func(x_i))
            
            order = np.argsort(fvals)
            pop = [pop[i] for i in order]
            fvals = [fvals[i] for i in order]
            x_mean_prev = x_mean.copy()
            x_mean = np.sum(np.array(pop[:mu]) * weights[:, None], axis=0)
            
            iteration = 0
            no_improve = 0
            local_best_f = fvals[0]
            
            while iteration < max_iter and eval_count < max_evals:
                if time.time() > deadline:
                    break
                
                iteration += 1
                
                # Compute weighted mean of selected solutions
                y = np.array(pop[:mu]) - x_mean_prev
                y_weighted = np.sum(weights[:, None] * y, axis=0)
                
                # Update evolution path for sigma
                d_sigma = np.sqrt(c_sigma * (5 + c_sigma))
                p_sigma = (1 - c_sigma) * p_sigma + d_sigma * np.sqrt(mu_eff) * y_weighted / (sigma + 1e-32)
                
                norm_p_sigma = np.linalg.norm(p_sigma)
                
                # h_sigma indicator
                hs = (norm_p_sigma / chi_n < (1.4 + c_sigma) * chi_n)
                
                # Update evolution path for covariance
                p_c = (1 - c_c) * p_c + hs * np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_mean - x_mean_prev) / (sigma + 1e-32)
                
                # Update covariance matrix
                delta_hs = (1 - hs) * c_c * (2 - c_c)
                C = (1 + c_1 + c_mu - c_1 - c_c * (1 + c_1 - c_c) + delta_hs) * C
                C += c_1 * (np.outer(p_c, p_c) - C)
                
                # Rank-mu update
                y_mu = (np.array(pop[:mu]) - x_mean_prev) / (sigma + 1e-32)
                C += c_mu * np.sum(weights[:, None] * np.outer(y_mu, y_mu), axis=0)
                
                # Ensure positive definiteness and update Cholesky
                C = 0.5 * (C + C.T)
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # Fallback: eigen decomposition
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-10)
                    C = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
                    try:
                        L = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
                    except:
                        L = np.eye(dim)
                
                # Update sigma
                sigma *= np.exp(c_sigma / chi_n * (norm_p_sigma / chi_n - 1))
                sigma = max(sigma, 1e-12)
                sigma = min(sigma, 10.0)
                
                # Generate new population
                new_pop = []
                new_fvals = []
                
                for i in range(pop_size):
                    if eval_count >= max_evals or time.time() > deadline:
                        break
                    z = np.random.randn(dim)
                    x_i = x_mean + sigma * L @ z
                    x_i = np.clip(x_i, -5.0, 5.0)
                    new_pop.append(x_i)
                    new_fvals.append(wrapped_func(x_i))
                
                if len(new_pop) < pop_size:
                    break
                
                order = np.argsort(new_fvals)
                new_pop = [new_pop[i] for i in order]
                new_fvals = [new_fvals[i] for i in order]
                
                # Update mean
                x_mean_prev = x_mean.copy()
                x_mean = np.sum(np.array(new_pop[:mu]) * weights[:, None], axis=0)
                
                # Check for improvement
                if new_fvals[0] < local_best_f - 1e-10:
                    no_improve = 0
                    local_best_f = new_fvals[0]
                else:
                    no_improve += 1
                
                if no_improve > 20:
                    break
                
                pop = new_pop
                fvals = new_fvals
            
            return x_mean, local_best_f

        # Phase 1: Initial sampling with strategic points
        # Strategic points for Rosenbrock and general exploration
        scout_points = []
        scout_points.append(np.zeros(dim))
        scout_points.append(np.ones(dim))
        scout_points.append(-np.ones(dim))
        scout_points.append(2.0 * np.ones(dim))
        scout_points.append(-2.0 * np.ones(dim))
        scout_points.append(np.array([1, 1, 1, 1, 1]))
        scout_points.append(np.array([0.5] * dim))
        scout_points.append(np.array([-0.5] * dim))
        
        # Add some random points for diversity
        rng = np.random.default_rng(42)
        for _ in range(10):
            scout_points.append(rng.uniform(-4.5, 4.5, dim))
        
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

        # Phase 2: CMA-ES with adaptive restarts
        # First run from best point
        remaining = budget - eval_count
        if remaining > 50:
            try:
                x_cma, f_cma = cma_es(best_x, 2.0, max_iter=80, max_evals=budget)
                if f_cma < best_f:
                    best_f = f_cma
                    best_x = x_cma.copy()
            except Exception:
                pass
        
        # Second CMA-ES from a different starting point
        remaining = budget - eval_count
        if remaining > 40:
            # Try starting from a random point in a promising region
            try:
                start2 = np.random.uniform(-3, 3, size=dim)
                x_cma2, f_cma2 = cma_es(start2, 1.5, max_iter=50, max_evals=budget)
                if f_cma2 < best_f:
                    best_f = f_cma2
                    best_x = x_cma2.copy()
            except Exception:
                pass

        # Third CMA-ES if budget allows
        remaining = budget - eval_count
        if remaining > 30:
            try:
                start3 = np.random.uniform(-2, 2, size=dim)
                x_cma3, f_cma3 = cma_es(start3, 1.0, max_iter=40, max_evals=budget)
                if f_cma3 < best_f:
                    best_f = f_cma3
                    best_x = x_cma3.copy()
            except Exception:
                pass

        # Phase 3: Local polish with L-BFGS-B
        remaining = budget - eval_count
        if remaining > 15 and best_x is not None:
            try:
                res = minimize(
                    wrapped_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={'maxiter': 200, 'maxfun': remaining, 'ftol': 1e-15, 'gtol': 1e-12}
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
                    options={'maxiter': 300, 'maxfev': remaining, 'xatol': 1e-14, 'fatol': 1e-16}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # Phase 5: Final L-BFGS-B pass if still have budget
        remaining = budget - eval_count
        if remaining > 5 and best_x is not None:
            try:
                res = minimize(
                    wrapped_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={'maxiter': 100, 'maxfun': remaining, 'ftol': 1e-16, 'gtol': 1e-14}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
