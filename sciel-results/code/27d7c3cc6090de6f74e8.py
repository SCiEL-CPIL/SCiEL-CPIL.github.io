"""Optimizer for BBOB-style continuous functions."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start optimizer with CMA-ES and L-BFGS-B polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 310.0  # 20s safety margin before 330s limit
        budget = self.budget
        dim = self.dim

        calls = 0
        best_x = None
        best_f = np.inf

        def safe_func(x):
            nonlocal calls
            if calls >= budget:
                return float('inf')
            calls += 1
            try:
                # Clip to box to be safe, though evaluator does it too
                x_clipped = np.clip(x, -5, 5)
                return float(func(x_clipped))
            except Exception:
                return float('inf')

        def track_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 1: CMA-ES
        # We will run CMA-ES for a portion of the budget.
        # Let's allocate 60% of the budget to CMA-ES.
        cma_budget = int(budget * 0.6)
        
        # Initialize CMA-ES parameters
        x0 = np.zeros(dim)
        sigma0 = 2.0
        popsize = 4 + int(3 * np.log(dim))
        
        # Simple CMA-ES implementation
        # We use a simplified version to avoid heavy dependencies or complex state management
        # that might be error-prone in this context.
        # Actually, implementing full CMA-ES from scratch is complex. 
        # Let's use a simpler but effective strategy: Nelder-Mead is too slow for 5D multimodal.
        # Let's try a differential evolution style or just a robust random search + local refinement.
        # Given the inspiration scores, CMA-ES is key.
        # I will implement a compact CMA-ES.

        # CMA-ES State
        x_mean = x0.copy()
        sigma = sigma0
        C = np.eye(dim)
        C_sqrt = np.eye(dim)
        
        # Learning rates
        mu = popsize // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * ((mu_eff - 2 + 1. / mu_eff) / ((dim + 2) ** 2 + mu_eff)))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        chi_n = np.sqrt(dim) * (1 - 1. / (4 * dim) + 1. / (21 * dim ** 2))
        
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        
        def update_covariance(psi, yw):
            nonlocal C, C_sqrt
            C = (1 - c_1 - c_mu) * C + c_1 * (2 - c_1) * np.outer(pc, pc) + c_mu * np.dot(yw, yw.T)
            # Ensure symmetry
            C = (C + C.T) / 2
            # Cholesky decomposition
            try:
                C_sqrt = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Fallback if not positive definite
                C += 1e-8 * np.eye(dim)
                C_sqrt = np.linalg.cholesky(C)

        # Initial population
        pops = []
        for i in range(popsize):
            z = np.random.randn(dim)
            x_i = x_mean + sigma * np.dot(C_sqrt, z)
            f_i = safe_func(x_i)
            track_best(x_i, f_i)
            pops.append((x_i, f_i))
        
        while calls < cma_budget and time.time() < deadline:
            # Sort by fitness
            pops.sort(key=lambda x: x[1])
            
            # Update mean
            yw = np.zeros(dim)
            for i in range(mu):
                yw += weights[i] * (pops[i][0] - x_mean)
            
            # Update sigma
            ps = np.sqrt(1 - c_sigma * (2 - c_sigma)) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * yw
            hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls / popsize))) < (1.4 + 2 / (dim + 1)) * chi_n
            
            if hsig:
                pc = (1 - c_c) * pc + c_c * yw / sigma
            
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            
            # Update covariance
            if hsig:
                pc = (1 - c_c) * pc + c_c * yw / sigma
            C = (1 - c_1 - c_mu) * C + c_1 * (2 - c_1) * np.outer(pc, pc)
            
            yw_for_cov = np.zeros((mu, dim))
            for i in range(mu):
                yw_for_cov[i] = (pops[i][0] - x_mean) / sigma
            C += c_mu * np.dot(yw_for_cov.T, yw_for_cov)
            C = (C + C.T) / 2
            
            try:
                C_sqrt = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C += 1e-8 * np.eye(dim)
                C_sqrt = np.linalg.cholesky(C)
                
            x_mean = x_mean + sigma * yw
            
            # Generate new population
            new_pops = []
            for i in range(popsize):
                z = np.random.randn(dim)
                x_i = x_mean + sigma * np.dot(C_sqrt, z)
                f_i = safe_func(x_i)
                track_best(x_i, f_i)
                new_pops.append((x_i, f_i))
            pops = new_pops

        # Phase 2: L-BFGS-B Polish
        # Use remaining budget for polishing
        if calls < budget and best_x is not None and time.time() < deadline:
            # We can try to polish the best point found
            # L-BFGS-B uses a limited number of function evaluations
            # We need to be careful not to exceed the budget
            remaining_budget = budget - calls
            
            # If we have enough budget, try a few polish attempts
            # L-BFGS-B might use many calls, so we limit it
            # A typical L-BFGS-B run might take 50-100 calls to converge well
            if remaining_budget > 50:
                try:
                    res = minimize(safe_func, best_x, method='L-BFGS-B', bounds=[(-5, 5)] * dim, 
                                   options={'maxiter': 100, 'maxfun': remaining_budget // 2})
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x
                except Exception:
                    pass

        return best_x, best_f
# EVOLVE-BLOCK-END
