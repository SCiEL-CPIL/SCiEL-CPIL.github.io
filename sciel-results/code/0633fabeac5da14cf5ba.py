"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Multi-start Nelder-Mead with Population-based (1+4)-ES exploration."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin before 330s kill
        
        dim = self.dim
        budget = self.budget
        
        best_x = None
        best_f = float('inf')
        calls_made = 0
        
        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_x = x.copy()
                best_f = f

        def check_time():
            return time.time() > deadline

        # Phase 1: Population-based (1+4)-ES for global exploration
        # We maintain a population of 5 candidates.
        # This helps escape local minima better than (1+1)-ES.
        
        # Initialize population with diverse points
        # Start with origin and some random points to cover the space
        pop = []
        
        # First point: origin
        x0 = np.zeros(dim)
        f0 = func(x0)
        calls_made += 1
        update_best(x0, f0)
        pop.append((x0, f0))
        
        # Add 4 more diverse points
        # Use a coarse grid or random sampling to ensure diversity
        # Let's use random sampling within [-4, 4]
        rng = np.random.default_rng(42)
        
        for _ in range(4):
            if calls_made >= budget or check_time():
                break
            x_new = rng.uniform(-4, 4, size=dim)
            x_new = np.clip(x_new, -5, 5)
            f_new = func(x_new)
            calls_made += 1
            update_best(x_new, f_new)
            pop.append((x_new, f_new))
        
        # Sort population by fitness
        pop.sort(key=lambda x: x[1])
        
        # Adaptive step size for ES
        sigma = 1.0
        
        # Run ES for a portion of the budget
        # Let's allocate 60% of the budget to exploration
        es_budget = int(budget * 0.6)
        
        # Track success rate for sigma adaptation
        success_count = 0
        window_size = 10
        success_history = []
        
        while calls_made < es_budget and not check_time():
            # Generate offspring for each parent (or just for the best few?)
            # To save budget, let's generate one offspring for the best parent (1+1 style within pop)
            # Or better: generate offspring for all, keep best 5.
            # With budget 500, and pop size 5, we can do ~400 steps.
            # Let's do a simple (5+5)-ES: generate 5 offspring, select best 5 from total 10.
            
            # Actually, to be efficient, let's just do (1+1) on the best point, 
            # but occasionally restart from other points if stuck.
            # Given the "multi-start" inspiration, let's stick to a population.
            
            # Generate 1 offspring from the best parent
            x_parent, f_parent = pop[0]
            z = rng.standard_normal(dim)
            x_trial = x_parent + sigma * z
            x_trial = np.clip(x_trial, -5, 5)
            
            f_trial = func(x_trial)
            calls_made += 1
            
            if f_trial < f_parent:
                # Success: replace best parent
                pop[0] = (x_trial, f_trial)
                update_best(x_trial, f_trial)
                success_history.append(1)
            else:
                # Failure: maybe replace worst parent with trial? 
                # Or just keep pop as is. 
                # Let's replace the worst if trial is better than worst, to maintain diversity.
                if f_trial < pop[-1][1]:
                    pop[-1] = (x_trial, f_trial)
                success_history.append(0)
            
            # Keep history window
            if len(success_history) > window_size:
                success_history.pop(0)
                
            # Adapt sigma
            if len(success_history) == window_size:
                success_rate = np.mean(success_history)
                if success_rate > 0.2:
                    sigma *= 1.1
                elif success_rate < 0.2:
                    sigma *= 0.9
                sigma = np.clip(sigma, 0.05, 5.0)
            
            # Re-sort population
            pop.sort(key=lambda x: x[1])
            
            # Occasionally, if no progress for a while, restart from a random point
            # Check if best_f has changed in last 20 calls? 
            # Simple heuristic: if sigma is very small and success rate low, restart.
            if len(success_history) == window_size and np.mean(success_history) < 0.1 and sigma < 0.1:
                # Restart best point to a random location
                x_restart = rng.uniform(-4, 4, size=dim)
                x_restart = np.clip(x_restart, -5, 5)
                f_restart = func(x_restart)
                calls_made += 1
                update_best(x_restart, f_restart)
                pop[0] = (x_restart, f_restart)
                pop.sort(key=lambda x: x[1])
                sigma = 1.0
                success_history = []
        
        # Phase 2: Multi-start Nelder-Mead refinement
        # Refine the top 3 candidates from the population.
        
        if best_x is None:
            return best_x, best_f
        
        # Select top 3 distinct points
        candidates = []
        seen = set()
        for x, f in pop:
            key = tuple(np.round(x, 4))
            if key not in seen:
                candidates.append((x, f))
                seen.add(key)
            if len(candidates) >= 3:
                break
        
        # If we didn't get 3, use best and some perturbations
        if len(candidates) < 3:
            for i in range(3 - len(candidates)):
                x_pert = best_x + rng.normal(0, 0.5, size=dim)
                x_pert = np.clip(x_pert, -5, 5)
                candidates.append((x_pert, None)) # f unknown, will be evaluated
        
        # Allocate remaining budget to refinement
        remaining_budget = budget - calls_made
        if remaining_budget <= 0 or check_time():
            return best_x, best_f
        
        # Distribute budget among candidates (e.g., 50% to best, 25% to 2nd, 25% to 3rd)
        budgets = [int(remaining_budget * 0.5), int(remaining_budget * 0.25), int(remaining_budget * 0.25)]
        
        for i, (x_start, f_start) in enumerate(candidates):
            if calls_made >= budget or check_time():
                break
            
            # If we don't have f_start, evaluate it
            if f_start is None:
                f_start = func(x_start)
                calls_made += 1
                update_best(x_start, f_start)
            
            # Run Nelder-Mead
            # We need to implement a budget-aware NM.
            # SciPy's minimize doesn't easily stop on arbitrary call counts without wrapping.
            
            nm_calls = [0]
            max_nm_calls = budgets[i]
            
            def nm_func(x):
                nonlocal calls_made
                if calls_made >= budget or check_time() or nm_calls[0] >= max_nm_calls:
                    return float('inf')
                f_val = func(x)
                calls_made += 1
                nm_calls[0] += 1
                update_best(x, f_val)
                return f_val
            
            try:
                # Use a small initial simplex
                perturbation = 0.1
                simplex = [x_start.copy()]
                for j in range(dim):
                    point = x_start.copy()
                    point[j] += perturbation
                    point = np.clip(point, -5, 5)
                    simplex.append(point)
                
                # SciPy minimize with Nelder-Mead
                # xatol and fatol should be tight to encourage convergence
                result = minimize(
                    nm_func, 
                    x_start, 
                    method='Nelder-Mead', 
                    options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-6, 'initial_simplex': simplex}
                )
            except Exception:
                pass
        
        return best_x, best_f

# EVOLVE-BLOCK-END
