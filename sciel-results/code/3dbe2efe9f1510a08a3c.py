"""
Optimizer using Multi-Start Basin Hopping.
Strategy:
1. Coarse Grid Search: Sample a structured grid to find a good starting point for the local search.
2. Local Refinement: Use Nelder-Mead (fast convergence) followed by Powell (polishing) to exploit the basin.
3. Budget Allocation: Prioritize early calls to get a decent lower bound, then refine.
4. Robustness: Handle the "Sharp Ridge" by using a robust local optimizer that doesn't get stuck in flat areas as easily as gradient-based methods (though NM is derivative-free, it can struggle with ridges, Powell is often better for ridges).
"""
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin
        
        best_x = None
        best_f = np.inf
        calls_made = 0

        def safe_call(x):
            nonlocal calls_made
            if time.time() > deadline:
                return np.inf
            if calls_made >= budget:
                return np.inf
            calls_made += 1
            try:
                val = func(x)
                return val
            except:
                return np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # 1. Initial Coarse Exploration
        # Use a stratified sampling approach to cover the box [-5, 5]^dim
        # For dim=5, a full grid is too expensive. We use a mix of random and center.
        rng = np.random.default_rng(42)
        
        # Phase 1: Global Search (approx 20% of budget)
        n_global = max(10, budget // 5)
        
        # Start with origin and corners
        initial_points = [np.zeros(dim)]
        # Add random points
        for _ in range(n_global - 1):
            p = rng.uniform(-4.5, 4.5, dim)
            initial_points.append(p)
            
        for p in initial_points:
            f = safe_call(p)
            if f < np.inf:
                update_best(p, f)
        
        if best_x is None:
             # Fallback if something went wrong
             return np.zeros(dim), 0.0

        # 2. Local Refinement Loop
        # We will take the best point found so far and try to refine it.
        # If refinement fails or stalls, we pick the next best point from the global phase.
        
        # We need to track the history of points to pick "next best"
        history = []
        # Re-evaluate initial points to build history properly? 
        # No, we can just re-run the global phase logic or store it.
        # Let's re-do the global phase storing (f, x) pairs.
        
        # Reset for a cleaner history management
        history = []
        # We already called n_global points. Let's assume we kept the best.
        # To be safe and simple, let's just refine the current best_x.
        # If we want multi-start, we need the list.
        
        # Let's implement a proper multi-start:
        # We will maintain a list of "promising" points.
        # Since we don't store all global points in a list easily without re-calling,
        # let's just do sequential refinement of the best found.
        # However, if the first local search fails, we want a new start.
        
        # Alternative: Generate a small set of diverse starts, evaluate them, 
        # then refine the best one. If it converges, check if we can do better with another start?
        # Given the "Sharp Ridge", a good global start is crucial.
        
        # Let's try 3 distinct local optimizations on the top 3 points from the global phase.
        # We need to recover the top 3.
        
        # Let's restart the global phase to store points, it's cheap (20% budget).
        # Actually, we already spent calls. Let's just use the best_x found.
        # If the best_x is good enough, we refine it.
        
        # To be more robust, let's do a "Walk" from the best point if local opt fails?
        # No, let's stick to: 
        # 1. Global Search (20%)
        # 2. Local Search on Best (80%)
        
        # But what if the global best is a local min? 
        # For f13 (Sharp Ridge), the global min is usually the target. 
        # The "Sharp Ridge" implies a long, thin valley. 
        # Nelder-Mead might struggle with the aspect ratio. 
        # Powell's method is often better for such ridges.
        
        # Strategy:
        # - Use Nelder-Mead for fast convergence to the neighborhood.
        # - Switch to Powell for final polishing.
        
        current_x = best_x.copy()
        current_f = best_f
        
        # Local Optimization Phase
        # We have budget - calls_made remaining.
        # We will use scipy.optimize.minimize.
        # We need to wrap func to count calls.
        
        def local_func(x):
            return safe_call(x)
        
        # Try Nelder-Mead first
        # Max iterations should be bounded by remaining budget
        # NM calls func multiple times per iteration (dim+1)
        max_nit = (budget - calls_made) // (dim + 1)
        max_nit = max(10, min(max_nit, 100))
        
        try:
            res = minimize(local_func, current_x, method='Nelder-Mead', 
                           options={'maxiter': max_nit, 'xatol': 1e-4, 'fatol': 1e-4})
            if res.success and res.fun < best_f:
                update_best(res.x, res.fun)
                current_x = res.x
                current_f = res.fun
        except:
            pass
        
        # Check time
        if time.time() > deadline - 10:
            return best_x, best_f

        # Try Powell for polishing
        # Powell is more robust for ridges
        # Powell calls func many times. 
        # maxiter for Powell is number of iterations.
        # Each iteration involves dim line searches.
        # Let's be conservative with maxiter.
        max_powell_iter = (budget - calls_made) // (dim * 2)
        max_powell_iter = max(5, min(max_powell_iter, 50))
        
        try:
            res = minimize(local_func, current_x, method='Powell', 
                           options={'maxiter': max_powell_iter, 'xtol': 1e-5, 'ftol': 1e-5})
            if res.fun < best_f:
                update_best(res.x, res.fun)
        except:
            pass
            
        # If we still have budget, try a few random restarts around the best point
        # to escape any potential local minima (though f13 is unimodal usually)
        remaining = budget - calls_made
        if remaining > 10:
            # Local random search
            step_size = 0.1
            for _ in range(min(remaining, 20)):
                if time.time() > deadline:
                    break
                dx = rng.normal(0, step_size, dim)
                new_x = np.clip(best_x + dx, -5, 5)
                f = safe_call(new_x)
                if f < best_f:
                    update_best(new_x, f)
                    # If we improved, shrink step size
                    step_size *= 0.5
                else:
                    step_size *= 0.8
                    if step_size < 1e-4:
                        break

        return best_x, best_f
