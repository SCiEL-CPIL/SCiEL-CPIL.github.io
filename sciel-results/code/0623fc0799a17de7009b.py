import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        # Safety margin: The evaluator kills at 330s. We aim for 300s limit in task, 
        # but use 290s as hard deadline to be safe.
        deadline = t0 + 290.0 

        rng = np.random.default_rng(42)
        call_count = 0
        best_x = None
        best_f = np.inf
        self._best_x = None
        self._best_f = np.inf

        def safe_func(x):
            nonlocal call_count, best_x, best_f
            if call_count >= self.budget or time.time() > deadline - 5:
                raise ValueError("Budget or time exceeded")
            call_count += 1
            x_arr = np.asarray(x, dtype=float)
            # The problem statement says func clips to [-5,5], but we should ensure 
            # our inputs are valid just in case, though the contract says func handles it.
            # We pass x_arr directly.
            val = float(func(x_arr))
            if np.isfinite(val) and val < best_f:
                best_f = val
                best_x = np.array(x_arr, dtype=float)
                self._best_x = best_x.copy()
                self._best_f = best_f
            return val

        # Phase 1: CMA-ES for Global/Coarse Search
        # Rosenbrock in 5D has a narrow valley. Grid search is inefficient.
        # CMA-ES adapts to the valley shape.
        # Budget split: ~40% for CMA-ES, ~60% for L-BFGS-B.
        cma_budget = int(self.budget * 0.4)
        
        # Initialize CMA-ES parameters manually to avoid overhead of importing cma package 
        # (which might not be installed, though scipy is). 
        # We will implement a simple CMA-ES or use a simplified adaptive random search.
        # Actually, scipy.optimize.minimize does not have CMA-ES.
        # We can implement a simple version or use Nelder-Mead with a large initial simplex?
        # No, NM is local.
        
        # Alternative: Use a "Smart" Random Search with adaptive step size.
        # Or, since we have NumPy, we can implement a basic CMA-ES loop.
        # Let's implement a minimal CMA-ES.
        
        def cma_es_search(max_calls):
            nonlocal call_count, best_x, best_f
            
            n = self.dim
            # Population size
            pop_size = 4 + int(3 * np.log(n))
            # Initial mean: center of the box
            mean = np.zeros(n)
            # Initial sigma: cover the box [-5, 5]
            sigma = 2.5 
            
            # Strategy parameters
            mu = pop_size // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1 / np.sum(weights**2)
            c_sigma = (mueff + 2) / (n + mueff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma
            c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            c_1 = 2 / ((n + 1.3)**2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2)**2 + mueff))
            
            # Covariance matrix
            C = np.eye(n)
            # Evolution path
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)
            
            # Precompute inverse square root of C
            # Using eigendecomposition for simplicity, updated each step
            
            f_history = []
            
            for gen in range(200): # Max generations, limited by max_calls
                if call_count >= max_calls or time.time() > deadline - 10:
                    break
                
                # Update C^{-1/2}
                # This is expensive for n=5? No, 5x5 is tiny.
                eigvals, eigvecs = np.linalg.eigh(C)
                C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(np.clip(eigvals, 1e-20, None))) @ eigvecs.T
                
                # Sample population
                pop = np.zeros((pop_size, n))
                fvals = np.zeros(pop_size)
                
                for i in range(pop_size):
                    z = rng.standard_normal(n)
                    x = mean + sigma * (C_inv_sqrt @ z)
                    # Clip to bounds to be safe, though func clips
                    x = np.clip(x, -5, 5)
                    pop[i] = x
                    try:
                        fvals[i] = safe_func(x)
                    except ValueError:
                        # Budget exceeded during sampling
                        return
                    except Exception:
                        fvals[i] = np.inf

                # Sort by fitness
                order = np.argsort(fvals)
                pop = pop[order]
                fvals = fvals[order]
                
                # Update best
                if fvals[0] < best_f:
                    best_f = fvals[0]
                    best_x = pop[0].copy()
                    self._best_x = best_x.copy()
                    self._best_f = best_f
                    
                f_history.append(fvals[0])
                
                # Update mean
                m_new = sum(weights[i] * pop[i] for i in range(mu))
                
                # Update sigma
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / sigma * (C_inv_sqrt @ (m_new - mean))
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**2) - 1))
                
                # Update C
                if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**2) < (1.4 + 2 / (n + 1)) * (1 - c_sigma):
                    p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mueff) / sigma * (m_new - mean)
                    C = (1 - c_1 - c_mu) * C + c_1 * (p_c @ p_c.T - C) + c_mu / sigma**2 * sum(weights[i] * np.outer(pop[i] - mean, pop[i] - mean) for i in range(mu))
                else:
                    p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mueff) / sigma * (m_new - mean)
                    C = (1 - c_1 - c_mu) * C + c_1 * (p_c @ p_c.T - C + (1 - c_c) * (2 - c_c)) + c_mu / sigma**2 * sum(weights[i] * np.outer(pop[i] - mean, pop[i] - mean) for i in range(mu))
                
                mean = m_new
                
                # Stagnation check
                if len(f_history) > 10 and np.std(f_history[-10:]) < 1e-8:
                    break

        # Run CMA-ES
        cma_es_search(cma_budget)

        if best_x is None:
            # Fallback if CMA-ES failed completely
            best_x = np.zeros(self.dim)
            try:
                best_f = safe_func(best_x)
            except:
                pass

        # Phase 2: L-BFGS-B for Local Refinement
        remaining = self.budget - call_count
        if remaining > 20 and time.time() < deadline - 20:
            try:
                res = minimize(
                    safe_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * self.dim,
                    options={'maxiter': 1000, 'maxfun': remaining, 'maxls': 50, 'gtol': 1e-10, 'ftol': 1e-14}
                )
                if res.fun < best_f:
                    best_f = float(res.fun)
                    best_x = np.array(res.x, dtype=float)
                    self._best_x = best_x.copy()
                    self._best_f = best_f
            except Exception:
                pass

        # Phase 3: Nelder-Mead Polish
        # NM is robust and doesn't rely on gradients, good for final polish
        remaining = self.budget - call_count
        if remaining > 20 and time.time() < deadline - 10:
            try:
                res_nm = minimize(
                    safe_func,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
                )
                if res_nm.fun < best_f:
                    best_f = float(res_nm.fun)
                    best_x = np.array(res_nm.x, dtype=float)
                    self._best_x = best_x.copy()
                    self._best_f = best_f
            except Exception:
                pass

        return best_x, best_f
