import numpy as np
from scipy.optimize import minimize
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.call_count = 0
        self.start_time = time.time()
        self.time_limit = 300.0

    def __call__(self, func):
        self.call_count = 0
        self.start_time = time.time()
        n = self.dim
        budget = self.budget
        
        def wrapped_func(x):
            if self.call_count >= budget:
                return float('inf')
            if time.time() - self.start_time > self.time_limit:
                return float('inf')
            self.call_count += 1
            x_clipped = np.clip(x, -5.0, 5.0)
            try:
                val = func(x_clipped)
                if np.isnan(val) or np.isinf(val):
                    return 1e10
                return val
            except Exception:
                return 1e10

        best_x = np.zeros(n)
        best_f = float('inf')
        
        # Phase 1: Global Search via Differential Evolution (DE)
        # DE is robust for global optimization and handles non-smooth/complex landscapes well.
        # We use a portion of the budget for this phase.
        NP = 12  # Population size
        F = 0.8
        CR = 0.9
        de_budget = 200  # Reserve 200 calls for DE
        
        rng = np.random.default_rng(12345)
        
        # Initialize population uniformly in the box
        pop = rng.uniform(-5.0, 5.0, size=(NP, n))
        fitness = np.zeros(NP)
        
        # Evaluate initial population
        for i in range(NP):
            fitness[i] = wrapped_func(pop[i])
            if fitness[i] < best_f:
                best_f = fitness[i]
                best_x = pop[i].copy()
        
        # DE Loop
        gen = 0
        while self.call_count < de_budget:
            if time.time() - self.start_time > self.time_limit - 10:
                break
                
            new_pop = np.zeros_like(pop)
            new_fitness = np.zeros(NP)
            
            for i in range(NP):
                if self.call_count >= de_budget:
                    break
                if time.time() - self.start_time > self.time_limit - 10:
                    break
                    
                # Select 3 distinct random indices
                r1, r2, r3 = rng.choice(NP, 3, replace=False)
                
                # DE/Rand/1
                v = pop[r1] + F * (pop[r2] - pop[r3])
                
                # Crossover
                trial = np.zeros(n)
                j_rand = rng.integers(0, n)
                for j in range(n):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                    else:
                        trial[j] = pop[i][j]
                
                # Evaluate
                trial = np.clip(trial, -5.0, 5.0)
                f_trial = wrapped_func(trial)
                
                # Selection
                if f_trial <= fitness[i]:
                    new_pop[i] = trial
                    new_fitness[i] = f_trial
                else:
                    new_pop[i] = pop[i]
                    new_fitness[i] = fitness[i]
                    
                if f_trial < best_f:
                    best_f = f_trial
                    best_x = trial.copy()
            
            pop = new_pop
            fitness = new_fitness
            gen += 1
            
            # Optional: Early stopping if no improvement for several generations?
            # For now, run full de_budget or until time/budget limit.
            
        # Phase 2: Local Polish with L-BFGS-B
        # Use the best point found by DE as the starting point for L-BFGS-B.
        # L-BFGS-B is very efficient for smooth functions and can find the precise minimum.
        
        remaining_budget = budget - self.call_count
        if remaining_budget > 20:
            # We might try a few of the best points from DE if we have budget,
            # but usually the global best is sufficient for L-BFGS-B to converge to the local min.
            # To be safe and fast, we just polish the single best point.
            
            try:
                # Use a callback to stop if we exceed budget or time
                def callback(xk):
                    if self.call_count >= budget:
                        return True
                    if time.time() - self.start_time > self.time_limit - 5:
                        return True
                    return False

                res = minimize(wrapped_func, best_x, method='L-BFGS-B', 
                               bounds=[(-5.0, 5.0) for _ in range(n)],
                               options={'maxiter': 1000, 'maxfun': remaining_budget, 'callback': callback})
                
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass
        
        # If we still have budget, we could try a second polish from a different top point,
        # or a coordinate descent, but L-BFGS-B on the best point is usually very strong for BBOB f14.
        # Let's add a simple coordinate descent polish if we have significant budget left,
        # just in case L-BFGS-B got stuck or didn't converge fully.
        
        remaining_budget = budget - self.call_count
        if remaining_budget > 50 and time.time() - self.start_time < self.time_limit - 20:
            # Simple coordinate descent
            step = 0.1
            improved = True
            while improved and self.call_count < budget and time.time() - self.start_time < self.time_limit - 5:
                improved = False
                for j in range(n):
                    # Try moving in + and - direction
                    for delta in [step, -step]:
                        x_new = best_x.copy()
                        x_new[j] += delta
                        x_new = np.clip(x_new, -5.0, 5.0)
                        f_new = wrapped_func(x_new)
                        if f_new < best_f:
                            best_f = f_new
                            best_x = x_new
                            improved = True
                # Shrink step
                step *= 0.5
                if step < 1e-5:
                    break
        
        return best_x, best_f
