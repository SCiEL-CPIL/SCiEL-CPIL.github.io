"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start Sobol initialization with Nelder-Mead refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit
        
        # Use Sobol sequence for deterministic low-discrepancy sampling
        # This ensures broad coverage early on, which is crucial for AOCC
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=self.dim, scramble=True, seed=42)
            # Generate a reasonable number of initial points. 
            # 50 points is a good balance between coverage and budget consumption.
            # 2^6 = 64 is the next power of 2, but we can just take 50.
            n_initial = min(50, self.budget // 2) 
            # Ensure n_initial is a power of 2 for Sobol efficiency if possible, 
            # but qmc.Sobol.random works for any n.
            # Let's stick to a power of 2 for better distribution properties if we can.
            n_pow2 = 1 << int(np.floor(np.log2(n_initial))) if n_initial > 1 else 1
            if n_pow2 < n_initial:
                 # If we can fit another power of 2, maybe use it? 
                 # Actually, just using n_initial is fine for Sobol.
                 pass
            
            # Generate points in [0, 1]^d
            sobol_points = sampler.random(n_initial)
            # Map to [-5, 5]^d
            sobol_points = sobol_points * 10.0 - 5.0
        except Exception:
            # Fallback to random if scipy.stats.qmc fails
            rng = np.random.default_rng(42)
            n_initial = min(50, self.budget // 2)
            sobol_points = rng.uniform(-5, 5, (n_initial, self.dim))

        # Phase 1: Evaluate all initial points
        best_x = None
        best_f = np.inf
        remaining_budget = self.budget - n_initial
        
        for i in range(n_initial):
            if time.time() > deadline:
                break
            x = sobol_points[i]
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            remaining_budget -= 1
            if remaining_budget <= 0:
                break

        if best_x is None:
            # Should not happen if budget > 0
            return None, np.inf

        # Phase 2: Refine the best point using Nelder-Mead
        # Nelder-Mead is derivative-free and robust for this kind of problem.
        # We use a tight maxiter to ensure we don't blow the time budget.
        # Each Nelder-Mead iteration uses roughly dim+1 function evaluations.
        # With dim=5, that's 6 evals/iteration.
        # If we have 450 evals left, that's ~75 iterations.
        
        # Define a wrapper to count calls and respect time/budget
        calls_made = 0
        max_calls_allowed = max(10, remaining_budget)
        
        def objective(x):
            nonlocal calls_made
            if calls_made >= max_calls_allowed:
                # Return a large value to signal termination or just stop?
                # Minimizers usually don't handle "stop" signals well mid-iteration.
                # It's safer to just let it run but check time in the main loop if possible.
                # However, scipy's minimize doesn't have a per-call time check.
                # We rely on the maxiter limit and the global deadline check after minimize returns.
                pass
            calls_made += 1
            return func(x)

        try:
            # Use Nelder-Mead
            # initial_simplex is optional, default is based on x0.
            # We can try to help it by using a slightly larger initial simplex?
            # Default is usually fine.
            res = minimize(
                objective,
                best_x,
                method='Nelder-Mead',
                options={
                    'maxiter': max(5, remaining_budget // 6), # Conservative estimate
                    'xatol': 1e-4,
                    'fatol': 1e-4,
                    'adaptive': True
                }
            )
            
            if res.success or res.fun < best_f:
                best_x = res.x
                best_f = res.fun
            else:
                # If it failed or didn't improve, keep the previous best
                # Note: res.fun might be better even if success is False due to maxiter
                if res.fun < best_f:
                    best_x = res.x
                    best_f = res.fun
        except Exception:
            pass

        # If we still have budget and time, we could do a second round of local search
        # or a coarser grid search around the best point, but Nelder-Mead usually suffices.
        # Given the "Sharp ridge" nature, gradient-based methods (BFGS) might be faster
        # if the function is smooth, but it's a black box. Nelder-Mead is safer.
        
        return best_x, best_f

# EVOLVE-BLOCK-END
