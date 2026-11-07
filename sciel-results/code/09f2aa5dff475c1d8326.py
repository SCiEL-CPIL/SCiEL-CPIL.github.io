import numpy as np
import time
import math
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: 330s kill limit. Reserve 30s for safety.
        deadline = start_time + 300
        
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        calls_used = 0
        
        # We track the best x found by any method
        def call_func(x):
            nonlocal calls_used, best_x, best_f
            if calls_used >= budget:
                raise Exception("Budget exceeded")
            if time.time() > deadline:
                raise Exception("Time limit exceeded")
            
            # Ensure x is a valid vector
            x = np.asarray(x, dtype=float)
            if x.shape != (n,):
                x = x.reshape(n)
            
            try:
                f_val = func(x)
            except Exception:
                # If func fails, we treat it as a very bad value to keep going
                # or raise if we want to stop. Contract says exception keeps calls.
                # But we should probably stop if the function is broken.
                # However, clipping is done by func.
                return np.inf
                
            calls_used += 1
            
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
                
            return f_val

        # Strategy:
        # 1. Use Differential Evolution (DE) for global search.
        #    DE is robust to rotations and shifts.
        #    We limit its function evaluations strictly.
        # 2. Use Powell or BFGS for local polishing from the best DE result.
        
        # Time allocation:
        # 60% of time for Global Search (DE)
        # 30% of time for Local Polish (Powell/BFGS)
        # 10% buffer
        
        total_time_budget = 280 # seconds
        t_global = 160
        t_local = 80
        
        # --- Phase 1: Global Search with Differential Evolution ---
        # DE is a population-based method.
        # We need to wrap func to count calls and enforce time.
        
        def de_wrapper(x):
            if time.time() > start_time + t_global:
                raise TimeoutError("Global phase time exceeded")
            return call_func(x)
        
        # DE parameters:
        # popsize: 10-20 is often good for 5D. Let's use 15.
        # maxiter: We don't know how many iters fit in time. 
        # We rely on the wrapper raising TimeoutError.
        # However, DE in scipy doesn't handle mid-iteration timeouts gracefully 
        # (it might hang or return partial). 
        # To be safe, we implement a simple custom DE loop or use a very tight 
        # maxiter estimate.
        
        # Estimate calls per DE iteration:
        # DE generates `popsize` candidates per iteration (roughly).
        # If t_global = 160s, and each call takes ~0.1ms (optimistic) to 1ms (realistic for Python/NumPy overhead + func),
        # let's assume 1000 calls/sec. 160s -> 160,000 calls.
        # But we only have 500 budget.
        # So DE will be limited by BUDGET, not time, likely.
        
        # Let's allocate 300 calls to DE, 200 to Local.
        
        de_budget = 300
        local_budget = 200
        
        # We will run DE for a fixed number of iterations that consumes ~de_budget calls.
        # DE with popsize P uses ~P calls per generation.
        popsize = 10
        max_generations = de_budget // popsize # ~30 generations
        
        # Initial population
        # DE requires bounds.
        bounds = [(-5.0, 5.0)] * n
        
        # We can't use scipy.optimize.differential_evolution directly with a hard call limit 
        # that stops the *entire* program cleanly if it exceeds time, because it's a black box.
        # Instead, we implement a lightweight DE loop ourselves to have full control over 
        # call counting and time checks.
        
        def run_custom_de():
            nonlocal best_x, best_f, calls_used
            
            # Initialize population
            # Mix of random and structured
            pop = []
            # One at origin
            pop.append(np.zeros(n))
            # Others random
            for _ in range(popsize - 1):
                pop.append(rng.uniform(-5, 5, n))
            
            pop = np.array(pop)
            # Evaluate initial population
            pop_f = np.array([call_func(x) for x in pop])
            
            # DE parameters
            F = 0.8  # Scale factor
            CR = 0.9  # Crossover rate
            
            for gen in range(max_generations):
                if time.time() > start_time + t_global:
                    break
                if calls_used >= de_budget:
                    break
                
                # Find best in current population
                best_idx = np.argmin(pop_f)
                best_x_gen = pop[best_idx]
                best_f_gen = pop_f[best_idx]
                
                # Update global best
                if best_f_gen < best_f:
                    best_f = best_f_gen
                    best_x = best_x_gen.copy()
                
                # Generate offspring for each individual
                for i in range(popsize):
                    if time.time() > start_time + t_global:
                        break
                    if calls_used >= de_budget:
                        break
                    
                    # Select 3 distinct random indices different from i
                    while True:
                        r1 = rng.integers(0, popsize)
                        r2 = rng.integers(0, popsize)
                        r3 = rng.integers(0, popsize)
                        if r1 != i and r2 != i and r3 != i and r1 != r2 and r2 != r3 and r1 != r3:
                            break
                    
                    # Mutation: v = x_r1 + F * (x_r2 - x_r3)
                    v = pop[r1] + F * (pop[r2] - pop[r3])
                    
                    # Clip to bounds
                    v = np.clip(v, -5.0, 5.0)
                    
                    # Crossover
                    j_rand = rng.integers(0, n)
                    trial = np.empty(n)
                    for j in range(n):
                        if rng.random() < CR or j == j_rand:
                            trial[j] = v[j]
                        else:
                            trial[j] = pop[i][j]
                    
                    # Evaluate trial
                    trial_f = call_func(trial)
                    
                    # Selection
                    if trial_f <= pop_f[i]:
                        pop[i] = trial
                        pop_f[i] = trial_f
            
            return best_x, best_f

        # Execute Phase 1
        run_custom_de()
        
        # --- Phase 2: Local Polish ---
        # Use the best point found by DE as the starting point for local optimization.
        # We use Powell (derivative-free) as it is robust to noise and doesn't need gradients.
        # We also try BFGS if we had gradients, but we don't.
        # Powell is a good choice for 5D.
        
        if best_x is not None and calls_used < budget - 10 and time.time() < deadline - 10:
            
            def local_wrapper(x):
                if time.time() > start_time + t_global + t_local:
                    raise TimeoutError("Local phase time exceeded")
                return call_func(x)
            
            # We will run Powell from the best point.
            # Powell's maxiter is the number of iterations (each involving multiple line searches).
            # In 5D, one iteration might use ~10-20 function calls.
            # We have ~200 calls left. So ~10-20 iterations.
            
            try:
                res = minimize(local_wrapper, best_x, method='Powell',
                               bounds=[(-5.0, 5.0)] * n,
                               options={'maxiter': 50, 'ftol': 1e-12, 'xtol': 1e-12})
                
                # Powell might not update best_f if it didn't find a better point via the wrapper's internal logic?
                # No, call_func updates best_f. So best_f is already updated.
                
            except Exception:
                pass
            
            # If we still have budget and time, try a second polish from a slightly perturbed best point?
            # Or just run BFGS? BFGS requires gradients. We can approximate them, but it's risky.
            # Let's stick to one robust Powell run.
            
            # If Powell failed or didn't improve, try Nelder-Mead from a different start?
            # Let's try Nelder-Mead from the original best_x if Powell didn't move much.
            
            if calls_used < budget - 5 and time.time() < deadline - 5:
                # Try Nelder-Mead from best_x
                try:
                    res_nm = minimize(local_wrapper, best_x, method='Nelder-Mead',
                                      options={'maxiter': 100, 'fatol': 1e-12, 'xatol': 1e-12})
                except Exception:
                    pass

        # Return the best found
        if best_x is None:
            # Fallback if everything failed (should not happen)
            best_x = np.zeros(n)
            best_f = call_func(best_x)
            
        return best_x, best_f
