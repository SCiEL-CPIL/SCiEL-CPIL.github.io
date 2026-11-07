"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """
    Hybrid optimizer for Rosenbrock f8:
    1. Diverse random exploration to find a good basin.
    2. L-BFGS-B for fast local convergence (quasi-Newton steps navigate the valley well).
    3. Nelder-Mead polish for final refinement.
    4. Multi-start mechanism if budget remains and initial start was poor.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        calls = 0
        best_x = None
        best_f = float('inf')

        # Deadline guard: 300s limit, keep 10s safety margin
        start_time = time.time()
        deadline = start_time + 285.0

        def wrapped_func(x):
            nonlocal calls, best_x, best_f
            if time.time() > deadline:
                return float('inf')
            if calls >= self.budget:
                return float('inf')
            
            x = np.asarray(x, dtype=float)
            if x.shape != (self.dim,):
                try:
                    x = x.reshape(self.dim)
                except:
                    return float('inf')
            
            calls += 1
            try:
                val = func(x)
                val = float(val)
            except Exception:
                val = float('inf')
            
            if val < best_f:
                best_f = val
                best_x = x.copy()
            
            return val

        # Phase 1: Diverse Random Exploration
        # Sample points to find a good starting basin.
        # Mix of uniform over full box and biased towards center.
        n_explore = min(15, self.budget // 10)
        
        for i in range(n_explore):
            if calls >= self.budget or time.time() > deadline:
                break
            if i < n_explore // 2:
                x = rng.uniform(-5, 5, self.dim)
            else:
                # Rosenbrock minimum is often near 0 or shifted, center is safer
                x = rng.uniform(-4, 4, self.dim)
            wrapped_func(x)

        # Phase 2: Local Exploitation using L-BFGS-B
        # L-BFGS-B is very effective for Rosenbrock due to its quasi-Newton steps
        # which navigate the narrow valley efficiently.
        if best_x is not None and calls < self.budget and time.time() < deadline:
            remaining = self.budget - calls
            # Reserve a small amount for final polish
            polish_budget = 50
            main_budget = max(10, remaining - polish_budget)
            
            try:
                res = minimize(
                    wrapped_func, 
                    best_x, 
                    method='L-BFGS-B', 
                    bounds=[(-5, 5)] * self.dim,
                    options={
                        'maxiter': 150,
                        'maxls': 50,
                        'maxfev': main_budget,
                        'ftol': 1e-12,
                        'gtol': 1e-10
                    }
                )
            except Exception:
                pass

        # Phase 3: High-precision Polish with Nelder-Mead
        # Nelder-Mead can sometimes refine further if L-BFGS-B stopped due to
        # tolerance or got stuck in a local flat area.
        if calls < self.budget and time.time() < deadline:
            remaining = self.budget - calls
            if remaining > 10:
                try:
                    res = minimize(
                        wrapped_func, 
                        best_x, 
                        method='Nelder-Mead', 
                        options={
                            'maxiter': 200,
                            'xatol': 1e-12,
                            'fatol': 1e-12,
                            'maxfev': remaining
                        }
                    )
                except Exception:
                    pass

        # Phase 4: If budget remains, try another local search from a random point
        # This acts as a simple multi-start mechanism if the first start was poor.
        if calls < self.budget - 20 and time.time() < deadline - 5:
            # Generate a few new random starts
            for _ in range(3):
                if calls >= self.budget - 20 or time.time() > deadline - 5:
                    break
                x_new = rng.uniform(-5, 5, self.dim)
                f_new = wrapped_func(x_new)
                
                # Only pursue if this point is competitive or significantly better
                # than the current best (to avoid wasting budget on bad starts)
                if f_new < best_f * 0.9 or best_f > 10:
                    remaining = self.budget - calls
                    if remaining > 20:
                        try:
                            res = minimize(
                                wrapped_func,
                                x_new,
                                method='L-BFGS-B',
                                bounds=[(-5, 5)] * self.dim,
                                options={
                                    'maxiter': 50,
                                    'maxfev': remaining
                                }
                            )
                        except Exception:
                            pass
                        break # Only do one additional local search to save time

        return best_x, best_f
# EVOLVE-BLOCK-END
