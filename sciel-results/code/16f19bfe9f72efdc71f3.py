"""
Hybrid Optimizer: Sobol global exploration + Nelder-Mead local exploitation.
Combines the structural idea of deterministic space-filling exploration (inspiration 2)
with the robust refinement machinery of derivative-free local search (inspiration 1).
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        best_f = np.inf
        best_x = np.zeros(dim)
        evals = 0

        def eval_func(x):
            nonlocal evals, best_f, best_x
            x = np.clip(x, -5, 5)
            f = func(x)
            evals += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Global Exploration using Sobol Sequence
        # We want to map the landscape well. For dim=5, 2^10 = 1024 points is a lot.
        # Let's use a portion of the budget for global search.
        # Strategy: Use ~20-30% of budget for global, rest for local.
        # However, we don't know how many evaluations Nelder-Mead will take.
        # Let's allocate fixed amounts based on budget size.
        
        n_global = int(budget * 0.25) 
        if n_global < 10:
            n_global = 10
            
        # Generate Sobol sequence
        # scipy.stats.qmc.Sobol requires scipy >= 1.7 or so. 
        # If not available, fallback to Halton or just random.
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=dim, scramble=True)
            # We need power of 2 for Sobol.
            # Find largest power of 2 <= n_global
            n_sobol = 1
            while n_sobol * 2 <= n_global:
                n_sobol *= 2
            # If n_global is not power of 2, we can just take the first n_global points
            # but Sobol is best with powers of 2. Let's generate enough and slice.
            # Actually, generating more than needed is fine.
            samples = sampler.random(n_sobol)
            # Map [0, 1]^dim to [-5, 5]^dim
            samples = samples * 10 - 5
            # Take first n_global points
            samples = samples[:n_global]
        except ImportError:
            # Fallback to Latin Hypercube or Random
            # Simple random for robustness if scipy version is old
            samples = rng.uniform(-5, 5, size=(n_global, dim))

        # Evaluate global samples
        for x in samples:
            if evals >= budget:
                break
            if time.time() > deadline - 10:
                break
            eval_func(x)

        # Phase 2: Local Exploitation using Nelder-Mead
        # Select the best point from global search to start NM.
        # We could also try multiple starts, but budget is limited.
        # Let's use the single best point found so far.
        
        x0 = best_x.copy()
        
        # Reserve some budget for safety
        remaining_budget = budget - evals
        
        # Nelder-Mead parameters
        # maxiter is not a direct eval count, but maxfev is.
        # We want to use most of the remaining budget.
        maxfev = max(10, int(remaining_budget * 0.9)) # Keep 10% reserve
        
        try:
            # Nelder-Mead is derivative-free.
            # initial_simplex_size is important.
            # If we start from a good global point, a moderate size is good.
            # If the function is smooth, NM converges fast.
            
            res = minimize(
                eval_func,
                x0,
                method='Nelder-Mead',
                options={
                    'maxfev': maxfev,
                    'xatol': 1e-10,
                    'fatol': 1e-12,
                    'adaptive': True
                }
            )
        except Exception:
            pass

        # Phase 3: Polish with L-BFGS-B if we have budget left and time
        # L-BFGS-B uses gradients (approximated) and is very fast for smooth parts.
        # Bi-Rastrigin has a smooth basin and a rough one. If we are in the smooth one, L-BFGS-B shines.
        
        if evals < budget - 5 and time.time() < deadline - 5:
            remaining = budget - evals
            # Use a small portion for final polish
            polish_maxfev = min(50, remaining)
            
            try:
                res2 = minimize(
                    eval_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={
                        'maxiter': 100,
                        'maxfun': polish_maxfev,
                        'gtol': 1e-10
                    }
                )
            except Exception:
                pass

        # Final fallback
        if best_x is None:
            best_x = np.zeros(dim)
            if evals < budget:
                best_f = eval_func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
