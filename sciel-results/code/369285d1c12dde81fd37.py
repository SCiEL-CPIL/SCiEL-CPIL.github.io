"""Black-box continuous optimizer on BBOB f11 (Discus)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer for BBOB f11 (Discus) using:
    - Phase 1: Sobol sampling for initial exploration.
    - Phase 2: CMA-ES for global search exploiting valley structure.
    - Phase 3: Multi-start L-BFGS-B local refinement from top candidates.
    - Deadline guard to ensure termination within time limits.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit

        rng = np.random.default_rng(42)
        best_x = None
        best_f = np.inf
        calls_used = 0

        def check_budget():
            nonlocal calls_used
            if calls_used >= self.budget:
                raise Exception("Budget exceeded")
            if time.time() > deadline:
                raise Exception("Time exceeded")

        def safe_call(x):
            nonlocal calls_used, best_f, best_x
            check_budget()
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Sobol sampling to build initial diverse set
        n_initial = min(32, self.budget // 8)
        from scipy.stats import qmc
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=42)
        sobol_points = sampler.random(n_initial)
        X_initial = sobol_points * 10.0 - 5.0

        evaluations = []
        for x in X_initial:
            if calls_used >= self.budget - 50:
                break
            fx = safe_call(x)
            evaluations.append((fx, x))

        if not evaluations:
            x = rng.uniform(-5, 5, self.dim)
            fx = safe_call(x)
            evaluations.append((fx, x))

        evaluations.sort(key=lambda t: t[0])

        # Phase 2: CMA-ES implementation (lightweight, no external dependency)
        def cma_es_optimize(func_eval, x0, sigma0, max_iter, budget_limit):
            """Simple CMA-ES implementation."""
            n = len(x0)
            x = x0.copy()
            sigma = sigma0
            C = np.eye(n)  # Covariance matrix
            ps = np.zeros(n)  # Evolution path for sigma adaptation
            hs = (1.4 + 2.0 / (n + 1)) * 0.5  # Threshold for sigma adaptation
            
            # Step size for evolution path
            c_sigma = (1 + 2 / (n + 1)) / (n + 4)
            c_c = (4 + n / 4) / (n + 4 + 4 / n)
            c_1 = 2 / ((n + 1.3) ** 2 + 10)
            c_mu = min(1 - c_1, 2 * ((mu_eff + 0.5) / (n + 2) ** 2 - 1))
            
            # Population size
            lambda_pop = 4 + int(3 * np.log(n))
            mu_eff = lambda_pop / 2
            weights = np.log(mu_eff + 0.5) - np.log(np.arange(1, mu_eff + 1))
            weights /= weights.sum()
            
            # Inverse square root of C
            def inv_sqrtm(M):
                eigval, eigvec = np.linalg.eigh(M)
                eigval = np.maximum(eigval, 1e-10)
                return eigvec @ np.diag(1.0 / np.sqrt(eigval)) @ eigvec.T
            
            C_sqrt_inv = inv_sqrtm(C)
            C_sqrt = C_sqrt_inv.T  # For our symmetric case
            
            # Run CMA-ES
            for iteration in range(max_iter):
                if calls_used >= budget_limit or time.time() > deadline - 10:
                    break
                
                # Generate lambda_pop candidates
                offspring = []
                for i in range(lambda_pop):
                    if calls_used >= budget_limit or time.time() > deadline - 10:
                        break
                    z = rng.standard_normal(n)
                    xi = x + sigma * (C_sqrt_inv @ z)
                    xi = np.clip(xi, -5, 5)
                    fx = func_eval(xi)
                    offspring.append((fx, xi))
                
                if len(offspring) == 0:
                    break
                
                # Sort by fitness
                offspring.sort(key=lambda t: t[0])
                
                # Update mean
                new_x = np.zeros(n)
                for i in range(min(mu_eff, len(offspring))):
                    new_x += weights[i] * offspring[i][1]
                
                # Update evolution path
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_sqrt_inv @ (new_x - x) / sigma
                chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (24 * n ** 2))
                if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (iteration + 1)) < (1.4 + 2 / (n + 1)) * chi_n:
                    # Update covariance
                    delta_hs = 1 if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (iteration + 1)) < (1.4 + 2 / (n + 1)) * chi_n else 0
                    C = (1 + c_c * delta_hs - c_1 - c_mu) * C + c_1 * np.outer(ps, ps)
                    for i in range(min(mu_eff, len(offspring))):
                        y = (offspring[i][1] - x) / sigma
                        C += c_mu * weights[i] * np.outer(y, y)
                
                # Adapt sigma
                sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1))
                sigma = np.clip(sigma, 1e-8, 10.0)
                
                x = new_x
                
                # Ensure C is positive definite
                try:
                    eigval, eigvec = np.linalg.eigh(C)
                    if np.any(eigval < 1e-10):
                        C = eigvec @ np.diag(np.maximum(eigval, 1e-10)) @ eigvec.T
                except:
                    C = np.eye(n)
                
                C_sqrt_inv = inv_sqrtm(C)
                C_sqrt = C_sqrt_inv.T
                
                # Early stopping if convergence
                if sigma < 1e-6:
                    break
            
            return x, func_eval(np.clip(x, -5, 5))

        # Use CMA-ES from the best initial point
        if calls_used < self.budget - 30 and time.time() < deadline - 30:
            fx0, x0 = evaluations[0]
            cma_max_iter = min(50, (self.budget - calls_used - 20) // 15)
            
            def cma_func(x):
                return safe_call(x)
            
            try:
                cma_x, cma_f = cma_es_optimize(cma_func, x0, 1.0, cma_max_iter, self.budget - 20)
                evaluations.append((cma_f, cma_x))
                evaluations.sort(key=lambda t: t[0])
            except Exception:
                pass

        # Phase 3: Local refinement using Nelder-Mead from top candidates
        from scipy.optimize import minimize
        remaining_budget = self.budget - calls_used
        n_restarts = min(3, len(evaluations))
        
        for i in range(n_restarts):
            if time.time() > deadline - 15:
                break
            if calls_used >= self.budget - 10:
                break
            
            fx0, x0 = evaluations[i]
            
            def nm_func(x):
                return safe_call(x)
            
            try:
                res = minimize(
                    nm_func,
                    x0,
                    method='Nelder-Mead',
                    options={
                        'maxiter': max(20, (self.budget - calls_used - 5) // n_restarts),
                        'xatol': 1e-8,
                        'fatol': 1e-10,
                        'adaptive': True
                    }
                )
            except Exception:
                pass

        # Phase 4: If budget remains, try a few more random restarts
        if calls_used < self.budget - 15 and time.time() < deadline - 20:
            possible_restarts = (self.budget - calls_used) // 25
            for _ in range(possible_restarts):
                if time.time() > deadline - 15:
                    break
                if calls_used >= self.budget - 10:
                    break
                
                # Perturb around current best
                if best_x is not None:
                    x0 = best_x + rng.normal(0, 0.5, self.dim)
                    x0 = np.clip(x0, -5, 5)
                else:
                    x0 = rng.uniform(-5, 5, self.dim)
                
                fx0 = safe_call(x0)
                
                if calls_used < self.budget - 5:
                    def nm_func2(x):
                        return safe_call(x)
                    try:
                        res = minimize(
                            nm_func2,
                            x0,
                            method='Nelder-Mead',
                            options={
                                'maxiter': 15,
                                'xatol': 1e-6,
                                'fatol': 1e-8,
                                'adaptive': True
                            }
                        )
                    except Exception:
                        pass

        if best_x is None:
            best_x = np.zeros(self.dim)
            if calls_used < self.budget:
                best_f = safe_call(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
