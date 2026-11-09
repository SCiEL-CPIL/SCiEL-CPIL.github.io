"""Adaptive CMA-ES style search + L-BFGS-B refinement for rotated Rosenbrock."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """CMA-ES style global search + L-BFGS-B refinement for rotated Rosenbrock."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 285  # Safety margin before 330s limit

        best_x = None
        best_f = np.inf
        call_count = 0

        def safe_func(x):
            nonlocal call_count, best_x, best_f
            if call_count >= budget:
                return 1e10
            x_clipped = np.clip(x, -5, 5)
            val = func(x_clipped)
            call_count += 1
            if val < best_f:
                best_f = val
                best_x = x_clipped.copy()
            return val

        # --- Phase 1: CMA-ES style adaptive search ---
        # Adapts step size and direction to handle rotated anisotropic landscape
        cma_budget = int(budget * 0.8)
        
        rng = np.random.default_rng(42)
        mean = np.zeros(dim)
        sigma = 3.0
        popsize = 2 * dim + 4  # ~14 for dim=5
        mu = popsize // 2
        
        # Weights for recombination
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        # Adaptive covariance matrix
        C = np.eye(dim)
        p_sigma = np.zeros(dim)
        
        chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        
        for gen in range(300):
            if call_count >= cma_budget or time.time() > deadline:
                break
            
            # Generate population using current mean, sigma, and C
            try:
                L = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
            except:
                L = np.eye(dim)
            
            solutions = []
            for i in range(popsize):
                z = rng.standard_normal(dim)
                x = mean + sigma * (L @ z)
                x = np.clip(x, -5, 5)
                solutions.append(x)
            
            # Evaluate
            fitnesses = []
            for x in solutions:
                if call_count >= cma_budget or time.time() > deadline:
                    break
                fitnesses.append(safe_func(x))
            
            if len(fitnesses) < mu:
                break
            
            # Sort and select
            indices = np.argsort(fitnesses)
            sel_idx = indices[:mu]
            
            # Update mean
            new_mean = np.zeros(dim)
            for j, idx in enumerate(sel_idx):
                new_mean += weights[j] * solutions[idx]
            
            # Update sigma
            y_sigma = (new_mean - mean) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-6, 5.0)
            
            # Simple covariance adaptation
            C_new = 0.8 * C
            for j, idx in enumerate(sel_idx):
                d = solutions[idx] - mean
                C_new += 0.2 * weights[j] * np.outer(d, d) / (sigma ** 2 + 1e-10)
            C = (C_new + C_new.T) / 2
            
            # Ensure positive definite
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-8)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            except:
                C = 0.9 * C + 0.1 * np.eye(dim)
            
            mean = new_mean
            mean = np.clip(mean, -5, 5)
        
        # --- Phase 2: L-BFGS-B refinement ---
        if best_x is not None and call_count < budget and time.time() < deadline:
            try:
                res = minimize(
                    safe_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 200, 'maxfev': budget - call_count,
                             'gtol': 1e-8}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5, 5)
            except Exception:
                pass

        # Fallback
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            call_count += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
