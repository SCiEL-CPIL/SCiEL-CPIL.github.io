import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0

        dim = self.dim
        budget = self.budget
        bounds = [(-5, 5)] * dim

        call_count = [0]
        best_f = [np.inf]
        best_x = [None]

        def wrapped_func(x):
            if time.time() > deadline:
                raise TimeoutError()
            if call_count[0] >= budget:
                raise BudgetExceeded()
            call_count[0] += 1
            fx = func(np.asarray(x, dtype=float))
            if np.isnan(fx) or np.isinf(fx):
                fx = np.inf
            if fx < best_f[0]:
                best_f[0] = fx
                best_x[0] = np.asarray(x, dtype=float).copy()
            return fx

        class BudgetExceeded(Exception):
            pass

        # Phase 1: CMA-ES (pure Python implementation)
        # CMA-ES is well-suited for continuous black-box optimization
        # and handles the hidden shift/rotation well.
        
        def cma_es(func_wrapper, dim, bounds, max_evals, seed=42):
            rng = np.random.default_rng(seed)
            lb = np.array([b[0] for b in bounds])
            ub = np.array([b[1] for b in bounds])
            
            # Initial point: center of the box
            x = np.zeros(dim)
            
            # Initial step size
            sigma = 0.5  # Start with a moderate step size
            
            # Population size
            lambda_ = 4 + int(3 * np.log(dim))
            mu = lambda_ // 2
            
            # Weights
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w /= np.sum(w)
            mueff = 1.0 / np.sum(w**2)
            
            # Adaptation rates
            c_sigma = (mueff + 2) / (dim + mueff + 5)
            c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
            c_1 = 2 / ((dim + 1.3)**2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))
            
            # Covariance matrix
            C = np.eye(dim)
            C_sqrt_inv = np.eye(dim)
            C_sqrt = np.eye(dim)
            
            # Evolution path
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            
            # Cumulative step size
            chi = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
            
            evals_used = 0
            best_x_local = x.copy()
            best_f_local = func_wrapper(x)
            evals_used += 1
            
            while evals_used < max_evals:
                if time.time() > deadline:
                    break
                
                # Sample population
                population = []
                for i in range(lambda_):
                    z = rng.standard_normal(dim)
                    y = sigma * C_sqrt @ z
                    x_trial = x + y
                    # Clip to bounds
                    x_trial = np.clip(x_trial, lb, ub)
                    population.append((x_trial, z, y))
                
                # Evaluate
                fitness = []
                for x_trial, z, y in population:
                    f = func_wrapper(x_trial)
                    evals_used += 1
                    fitness.append(f)
                    if f < best_f_local:
                        best_f_local = f
                        best_x_local = x_trial.copy()
                    if evals_used >= max_evals:
                        break
                
                if evals_used >= max_evals:
                    break
                
                # Sort by fitness
                indices = np.argsort(fitness[:evals_used - (lambda_ - len(fitness))]) if len(fitness) < lambda_ else np.argsort(fitness)
                # Actually, let's just sort the ones we have
                sorted_pop = sorted(zip(fitness, [p[0] for p in population[:len(fitness)]]), key=lambda t: t[0])
                
                if len(sorted_pop) < mu:
                    break
                
                # Mean of best mu
                mean_new = np.mean([p[1] for p in sorted_pop[:mu]], axis=0)
                
                # Weighted mean
                mean_weighted = np.zeros(dim)
                for i in range(mu):
                    mean_weighted += w[i] * sorted_pop[i][1]
                
                # Update evolution paths
                y_w = mean_weighted - x
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / sigma * y_w
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mueff) / sigma * y_w
                
                # Update sigma
                sigma *= np.exp(c_sigma / chi * (np.linalg.norm(p_sigma) / chi - 1))
                sigma = np.clip(sigma, 1e-10, 10.0)
                
                # Update covariance matrix
                delta_d = (1 + 2 * (mueff > mu + 1) * c_mu) * c_c
                outer = np.outer(p_c, p_c)
                C = (1 + c_1 + c_mu * mueff - c_1) * C + c_1 * (outer - C)
                
                # Ensure C is symmetric positive definite
                C = (C + C.T) / 2
                try:
                    C_sqrt = np.linalg.cholesky(C)
                    C_sqrt_inv = np.linalg.inv(C_sqrt)
                except:
                    # If Cholesky fails, reset
                    C = np.eye(dim)
                    C_sqrt = np.eye(dim)
                    C_sqrt_inv = np.eye(dim)
                
                # Update mean
                x = mean_weighted
                x = np.clip(x, lb, ub)
            
            return best_x_local, best_f_local

        # Phase 1: Run CMA-ES with most of the budget
        cma_budget = int(budget * 0.8)
        remaining_budget = budget - cma_budget
        
        try:
            cx, cf = cma_es(wrapped_func, dim, bounds, cma_budget, seed=42)
            if cx is not None:
                if cf < best_f[0]:
                    best_f[0] = cf
                    best_x[0] = cx.copy()
        except (TimeoutError, BudgetExceeded):
            pass
        except Exception:
            pass

        # Phase 2: Local refinement with Nelder-Mead if budget remains
        if best_x[0] is not None and call_count[0] < budget:
            try:
                from scipy.optimize import minimize
                res = minimize(
                    lambda x: wrapped_func(x),
                    best_x[0],
                    method='Nelder-Mead',
                    options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True}
                )
            except (TimeoutError, BudgetExceeded):
                pass
            except Exception:
                pass

        return best_x[0], best_f[0]
