import numpy as np
from scipy.optimize import minimize
import time

class Optimizer:
    """
    Multi-start Nelder-Mead followed by CMA-ES for continuous black-box optimization.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290  # Safety margin for 300s limit
        
        # Phase 1: Multi-start Nelder-Mead for global exploration
        # Use 5 starts, ~20-30 evals each
        n_starts = 5
        evals_per_start = 20
        total_exploration = n_starts * evals_per_start
        
        # Ensure we don't exceed budget
        if total_exploration > self.budget // 2:
            n_starts = max(1, (self.budget // 2) // evals_per_start)
            total_exploration = n_starts * evals_per_start

        best_x = None
        best_f = np.inf
        rng = np.random.default_rng(42)

        # Exploration
        for i in range(n_starts):
            x0 = rng.uniform(-5, 5, self.dim)
            # Nelder-Mead is fast for low dims
            try:
                res = minimize(func, x0, method='Nelder-Mead', 
                               options={'maxiter': evals_per_start, 'xatol': 1e-4, 'fatol': 1e-6})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                pass
            
            if time.time() > deadline - 10:
                break

        # If no valid result from exploration, use a random point
        if best_x is None:
            best_x = rng.uniform(-5, 5, self.dim)
            best_f = func(best_x)

        # Phase 2: CMA-ES for local exploitation
        # Use remaining budget
        # Estimate calls used in Phase 1. Nelder-Mead doesn't report exact calls easily in all versions, 
        # but we can track them or just assume worst case and leave a buffer.
        # Actually, let's just run CMA-ES for a fixed number of generations that fits in the remaining time/budget.
        # To be safe with the "500 calls" hard limit, we should track calls.
        
        # Let's implement a call counter wrapper
        calls_made = 0
        def wrapped_func(x):
            nonlocal calls_made
            calls_made += 1
            if calls_made > self.budget:
                raise RuntimeError("Budget exceeded")
            return func(x)
            
        # We need to know how many calls Phase 1 used.
        # Scipy's minimize doesn't return the number of function evaluations directly in the result object in a standard way for all methods, 
        # but `res.nfev` is available for Nelder-Mead.
        
        # Let's re-do Phase 1 with explicit counting to be precise.
        
        calls_made = 0
        best_x = None
        best_f = np.inf
        
        for i in range(n_starts):
            x0 = rng.uniform(-5, 5, self.dim)
            try:
                res = minimize(wrapped_func, x0, method='Nelder-Mead', 
                               options={'maxiter': evals_per_start, 'xatol': 1e-4, 'fatol': 1e-6})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                pass
            
            if time.time() > deadline - 10:
                break
                
        if best_x is None:
            best_x = rng.uniform(-5, 5, self.dim)
            best_f = wrapped_func(best_x)

        # Remaining budget for CMA-ES
        remaining_budget = self.budget - calls_made
        
        if remaining_budget < 10:
            return best_x, best_f

        # CMA-ES Parameters
        lam = 10  # Population size
        # Max generations
        max_gen = remaining_budget // lam
        
        # Initialize CMA-ES
        m = best_x.copy()
        sigma = 0.5  # Start with a smaller sigma as we are in a local region
        C = np.eye(self.dim)
        pc = np.zeros(self.dim)
        ps = np.zeros(self.dim)
        
        # Weights
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mu_eff = 1 / np.sum(w**2)
        
        # Adaptation parameters
        cs = (mu_eff + 2) / (self.dim + mu_eff + 5)
        ds = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (self.dim + 1)) - 1) + cs
        cc = (4 + mu_eff / self.dim) / (self.dim + 4 + 2 * mu_eff / self.dim)
        c1 = 2 / ((self.dim + 1.3)**2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((self.dim + 2)**2 + mu_eff))
        
        # CMA-ES Loop
        for g in range(max_gen):
            if time.time() > deadline - 5:
                break
                
            # Sample
            B = np.linalg.cholesky(C)
            D = np.diag(np.sqrt(np.diag(C)))
            # Actually, simpler: sample from N(0, C)
            # z ~ N(0, I)
            # x = m + sigma * B @ z
            # But B is cholesky factor.
            
            # Standard CMA-ES sampling
            z = rng.standard_normal((lam, self.dim))
            # Transform
            # x_k = m + sigma * B @ z_k
            # We need B such that B @ B.T = C
            B = np.linalg.cholesky(C)
            # Note: cholesky returns lower triangular.
            # x = m + sigma * (B @ z.T).T
            pop = m[None, :] + sigma * (B @ z.T).T
            
            # Evaluate
            fit = np.array([wrapped_func(x) for x in pop])
            
            # Sort
            idx = np.argsort(fit)
            fit_sorted = fit[idx]
            pop_sorted = pop[idx]
            
            # Update best
            if fit_sorted[0] < best_f:
                best_f = fit_sorted[0]
                best_x = pop_sorted[0].copy()
                
            # Update mean
            m_new = m + sigma * np.sum(w[:, None] * pop_sorted[:mu], axis=0)
            
            # Update covariance
            y = (m_new - m) / sigma
            # Update pc
            pc = (1 - cs) * pc + np.sqrt(cs * (2 - cs) * mu_eff) * y
            # Update ps
            ps = (1 - ds) * ps + np.sqrt(ds * (2 - ds) * cs * (1 - cs)) * (B.T @ y)
            
            # Delta term
            delta_term = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * (g + 1))) - np.sqrt((self.dim - 1) * (1 - cs**2))) * 1.0
            
            # Update C
            C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (np.dot(ps, ps) > ds * (2 - ds) * cs * (1 - cs) * (1 + 1 / ds)) - 1) * np.eye(self.dim) \
                + cmu * np.sum(w[:, None] * np.outer(pop_sorted[:mu] - m, pop_sorted[:mu] - m) / (sigma**2), axis=0)
                
            # Ensure symmetry
            C = (C + C.T) / 2
            
            # Update sigma
            sigma = sigma * np.exp(cs / ds * (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * (g + 1))) - np.sqrt(self.dim - 1) * (1 - cs**2)))
            
            # Check for positive definiteness
            try:
                np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C is not PD, reset it or add regularization
                C = C + 1e-6 * np.eye(self.dim)
                
            m = m_new
            
            # Stagnation check: if no improvement in best_f for 10 generations, break?
            # For now, just run until budget.

        return best_x, best_f
