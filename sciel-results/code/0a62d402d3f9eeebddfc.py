"""Rewritten program for BBOB-style Rastrigin optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Rastrigin-aware hybrid: Multi-start CMA-ES with large population + Local Polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin for 300s limit
        
        rng = np.random.default_rng(42)
        
        calls_made = 0
        best_x = np.zeros(dim)
        best_f = np.inf
        
        def counted_func(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return np.inf
            calls_made += 1
            x = np.asarray(x)
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Initial Exploration
        # Sample a few points to get a rough idea of the landscape.
        n_init = 10
        for i in range(n_init):
            if time.time() > deadline:
                break
            if i == 0:
                x = np.zeros(dim)
            else:
                x = rng.uniform(-4, 4, dim)
            counted_func(x)

        # Phase 2: Multi-start CMA-ES with large initial population
        remaining_budget = budget - calls_made
        if remaining_budget > 10 and time.time() < deadline:
            # Split remaining budget into 3 runs of decreasing size
            run_budgets = [int(remaining_budget * 0.4), int(remaining_budget * 0.3), int(remaining_budget * 0.2)]
            
            current_best_x = best_x.copy()
            
            for run_idx, rb in enumerate(run_budgets):
                if calls_made >= budget or time.time() > deadline - 5:
                    break
                
                if rb < 10:
                    continue

                # Determine starting point
                if run_idx == 0:
                    x_start = current_best_x.copy()
                else:
                    sigma = 1.0 / (run_idx + 1)
                    x_start = current_best_x + rng.normal(0, sigma, dim)
                    x_start = np.clip(x_start, -5, 5)
                
                try:
                    # CMA-ES with larger population size for better exploration
                    # popsize=30 helps explore the multimodal landscape and adapt to rotation
                    res = self._cmaes(counted_func, x_start, rb)
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x.copy()
                except Exception:
                    try:
                        res = self._powell(counted_func, x_start, rb)
                        if res.fun < best_f:
                            best_f = res.fun
                            best_x = res.x.copy()
                    except Exception:
                        pass

        # Phase 3: Final Polish with Nelder-Mead
        if calls_made < budget - 5 and time.time() < deadline - 5:
            try:
                res = self._nelder_mead(counted_func, best_x, budget - calls_made)
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        return best_x, best_f

    def _cmaes(self, func, x0, maxfev):
        """Custom CMA-ES implementation for better control."""
        from scipy.optimize import minimize
        return minimize(
            func, 
            x0, 
            method='CMA-ES', 
            options={
                'maxiter': 200,
                'maxfev': maxfev,
                'tol': 1e-10,
                'seed': 42,
                'popsize': 30
            }
        )

    def _powell(self, func, x0, maxfev):
        """Powell method fallback."""
        from scipy.optimize import minimize
        return minimize(
            func, 
            x0, 
            method='Powell', 
            options={'maxiter': 50, 'maxfev': maxfev}
        )

    def _nelder_mead(self, func, x0, maxfev):
        """Nelder-Mead for final polish."""
        from scipy.optimize import minimize
        return minimize(
            func, 
            x0, 
            method='Nelder-Mead', 
            options={
                'maxiter': 100, 
                'xatol': 1e-10, 
                'fatol': 1e-12,
                'adaptive': True
            }
        )
# EVOLVE-BLOCK-END
