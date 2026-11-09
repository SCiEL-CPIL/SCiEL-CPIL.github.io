"""
Strategy: Nelder-Mead Simplex with Simulated Annealing Restarts.
Rationale: The Discus function (f11) is a continuous, differentiable function with a single global minimum (though shifted/rotated). 
Population-based methods like CMA-ES are powerful but can be overkill for 5D smooth functions with a budget of 500.
Nelder-Mead is a direct search method that maintains a simplex of size dim+1 (6 points for dim=5). 
It is extremely sample-efficient for smooth local minima. 
The main risk is getting stuck in local minima, but f11 is known to have a relatively simple landscape (unimodal or few local minima).
To mitigate this, we use a multi-start strategy where we perform a Nelder-Mead run, and if it stagnates (step size becomes tiny or no improvement for N iterations), 
we restart from a new random point, potentially using the best point found so far as a guide or a fresh random point.
We also incorporate a simple "annealing" factor: initially we take larger steps (by scaling the simplex), and as we progress, we tighten the tolerance.
This approach focuses on rapid convergence to local minima, which is crucial for AOCC (Area Over the Curve of Convergence).
"""

import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin
        n_calls = 0
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = np.zeros(dim)
        best_f = float('inf')

        def call_f(x):
            nonlocal n_calls, best_x, best_f
            if n_calls >= budget:
                return None
            n_calls += 1
            f_val = func(x)
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        def nelder_mead(x0, max_iter, tol=1e-8, alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5):
            """
            Standard Nelder-Mead implementation.
            Returns (best_x, best_f, n_calls_used)
            """
            nonlocal n_calls, best_x, best_f
            
            # Initialize simplex
            # x0 is the center
            # Create simplex with small perturbations
            simplex = np.zeros((dim + 1, dim))
            simplex[0] = x0
            for i in range(1, dim + 1):
                simplex[i] = x0.copy()
                simplex[i][i - 1] += 0.05 # Small initial step size
            
            # Evaluate initial simplex
            fvals = np.zeros(dim + 1)
            for i in range(dim + 1):
                f = call_f(simplex[i])
                if f is None:
                    return best_x, best_f, n_calls
                fvals[i] = f
            
            # Sort simplex by fvals
            def sort_simplex():
                nonlocal simplex, fvals
                idx = np.argsort(fvals)
                simplex = simplex[idx]
                fvals = fvals[idx]
            
            sort_simplex()
            
            for _ in range(max_iter):
                if time.time() > deadline - 5:
                    break
                
                # Check convergence
                if np.std(fvals) < tol:
                    break
                
                # Calculate centroid of all points except the worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5.0, 5.0)
                f_r = call_f(x_r)
                if f_r is None:
                    break
                
                # Sort to see where x_r fits
                # If f_r is better than the worst but worse than the best, replace worst
                if fvals[0] <= f_r < fvals[-1]:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
                    sort_simplex()
                elif f_r < fvals[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5.0, 5.0)
                    f_e = call_f(x_e)
                    if f_e is None:
                        break
                    if f_e < f_r:
                        simplex[-1] = x_e
                        fvals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fvals[-1] = f_r
                    sort_simplex()
                else:
                    # Contraction
                    # Inside contraction
                    x_c = centroid + rho * (simplex[-1] - centroid)
                    x_c = np.clip(x_c, -5.0, 5.0)
                    f_c = call_f(x_c)
                    if f_c is None:
                        break
                    if f_c < fvals[-1]:
                        simplex[-1] = x_c
                        fvals[-1] = f_c
                        sort_simplex()
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            f = call_f(simplex[i])
                            if f is None:
                                return best_x, best_f, n_calls
                            fvals[i] = f
                        sort_simplex()
                        
            return best_x, best_f, n_calls

        # Multi-start strategy
        # We will run several Nelder-Mead instances.
        # The first one starts from 0.
        # Subsequent ones start from random points or near the current best if it's good.
        
        # Number of restarts is dynamic based on budget used
        # Let's say we allow up to 10 restarts, but each run has a max budget.
        
        # Initial run
        x_start = np.zeros(dim)
        # Give the first run a generous budget, e.g., 200 calls
        max_iter_run = 50 # Nelder-Mead iterations, each uses ~1-2 calls on average, but let's cap calls per run
        
        # We need to be careful with the call budget.
        # Let's define a function that runs Nelder-Mead but stops if budget is exceeded.
        # The nelder_mead function above already checks n_calls inside call_f.
        
        # Let's structure the restarts:
        # 1. Start at 0.
        # 2. If stagnation or time, restart.
        
        # To improve AOCC, we want to find a good value early.
        # Nelder-Mead from 0 is usually very fast for f11 if 0 is close to the minimum.
        # If the shift is large, it might take a while.
        
        # Let's do a quick random search for the first 20 calls to find a better starting point?
        # No, that wastes budget. Nelder-Mead from 0 is robust enough.
        
        # Let's run Nelder-Mead with a cap on calls per run.
        # If it doesn't converge, we restart.
        
        # Strategy:
        # Run 1: Start at 0, max 100 calls.
        # Run 2: Start at random point, max 80 calls.
        # Run 3: Start at random point, max 80 calls.
        # ...
        # Until budget is exhausted.
        
        # However, Nelder-Mead can be sensitive to initial simplex size.
        # Let's use a slightly larger initial step size for robustness.
        
        # Modified Nelder-Mead to accept a max_calls parameter and return early.
        # The existing nelder_mead uses a loop of max_iter. 
        # Each iteration uses 1-2 function evaluations.
        # So max_iter=50 is roughly 50-100 calls.
        
        # Let's refine the restart logic.
        # We will run Nelder-Mead repeatedly.
        # The starting point for the next run will be a random point, 
        # UNLESS the previous run found a significantly better point, 
        # in which case we might perturb that point.
        
        # Actually, for f11, the global minimum is unique. 
        # If we find a local minimum, it's likely the global one or very close.
        # So restarting from random points is good for exploration.
        
        # Let's implement the loop.
        
        # We need to track the best point globally.
        # nelder_mead updates best_x and best_f via nonlocal.
        
        # To ensure we don't waste calls on a run that is clearly stuck, 
        # we can check if the best_f improved during the run.
        
        # Let's just run a sequence of Nelder-Mead instances.
        # Instance 1: x0 = [0,0,0,0,0]
        # Instance k: x0 = rng.uniform(-4, 4, dim)
        
        # How many instances?
        # Budget 500.
        # If each instance uses ~50 calls, we can do ~10 instances.
        
        # Let's define a helper to run one instance with a specific call limit.
        # We can't easily limit calls inside the existing nelder_mead without modifying it.
        # But nelder_mead checks `if n_calls >= budget` inside call_f.
        # So if we set a local budget limit, it will stop.
        
        # Let's modify the approach:
        # We will run Nelder-Mead. 
        # We will pass a "max_calls_for_this_run" to the nelder_mead function?
        # No, nelder_mead uses the global budget.
        # We need to be smarter.
        
        # Let's create a wrapper that runs Nelder-Mead but counts calls relative to the start of the run.
        
        def run_nm(x_start, max_calls):
            nonlocal n_calls, best_x, best_f
            start_n = n_calls
            
            # We need to modify nelder_mead to stop if n_calls - start_n > max_calls
            # Or we can just run it and let the global budget guard handle it.
            # But if one run takes 400 calls and stagnates, we have no budget for restarts.
            # So we MUST limit calls per run.
            
            # Let's rewrite nelder_mead to take a max_calls parameter.
            
            simplex = np.zeros((dim + 1, dim))
            simplex[0] = x_start
            for i in range(1, dim + 1):
                simplex[i] = x_start.copy()
                simplex[i][i - 1] += 0.05
            
            fvals = np.zeros(dim + 1)
            for i in range(dim + 1):
                f = call_f(simplex[i])
                if f is None:
                    return best_x, best_f
                fvals[i] = f
            
            def sort_simplex():
                nonlocal simplex, fvals
                idx = np.argsort(fvals)
                simplex = simplex[idx]
                fvals = fvals[idx]
            
            sort_simplex()
            
            # Max iterations is roughly max_calls / 1.5
            max_iter = int(max_calls / 1.5)
            
            for _ in range(max_iter):
                if n_calls - start_n >= max_calls:
                    break
                if time.time() > deadline - 5:
                    break
                
                if np.std(fvals) < 1e-8:
                    break
                
                centroid = np.mean(simplex[:-1], axis=0)
                
                x_r = centroid + 1.0 * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5.0, 5.0)
                f_r = call_f(x_r)
                if f_r is None:
                    break
                
                if fvals[0] <= f_r < fvals[-1]:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
                    sort_simplex()
                elif f_r < fvals[0]:
                    x_e = centroid + 2.0 * (x_r - centroid)
                    x_e = np.clip(x_e, -5.0, 5.0)
                    f_e = call_f(x_e)
                    if f_e is None:
                        break
                    if f_e < f_r:
                        simplex[-1] = x_e
                        fvals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fvals[-1] = f_r
                    sort_simplex()
                else:
                    x_c = centroid + 0.5 * (simplex[-1] - centroid)
                    x_c = np.clip(x_c, -5.0, 5.0)
                    f_c = call_f(x_c)
                    if f_c is None:
                        break
                    if f_c < fvals[-1]:
                        simplex[-1] = x_c
                        fvals[-1] = f_c
                        sort_simplex()
                    else:
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            f = call_f(simplex[i])
                            if f is None:
                                return best_x, best_f
                            fvals[i] = f
                        sort_simplex()
                        
            return best_x, best_f

        # Run 1: Start at 0
        run_nm(np.zeros(dim), 100)
        
        # Subsequent runs: Start at random points
        # We have remaining budget.
        # Let's allocate 50 calls per subsequent run.
        
        while n_calls < budget and time.time() < deadline - 10:
            # Pick a new random start
            x_start = rng.uniform(-4, 4, dim)
            
            # Check if we are close to the best? 
            # If the best is very good, maybe we should perturb the best instead?
            # For f11, random restarts are safer to avoid getting stuck in the same local basin.
            
            run_nm(x_start, 50)
            
            # If we made very little progress, we might want to break early?
            # No, let's use the budget.
            
        return best_x, best_f
