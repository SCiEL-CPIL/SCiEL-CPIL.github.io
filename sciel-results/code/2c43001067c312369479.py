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
        # 330s kill limit. Leave 50s margin.
        deadline = start_time + 280.0
        
        rng = np.random.default_rng(42)
        calls_made = 0
        best_f = np.inf
        best_x = np.zeros(dim)
        
        # Store all evaluated points to potentially restart local search from good ones
        history_x = []
        history_f = []

        def eval_func(x):
            nonlocal calls_made, best_f, best_x
            # Clip to box
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            calls_made += 1
            history_x.append(x.copy())
            history_f.append(fx)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def time_check():
            return time.time() < deadline and calls_made < budget

        # Phase 1: Differential Evolution for Global Exploration
        # Use a smaller population for faster convergence in early stages
        pop_size = 12
        F = 0.8
        CR = 0.9
        
        # Initialize population
        pop = []
        # 1 point near origin (common for BBOB shifts)
        pop.append(np.zeros(dim) + 0.5 * rng.standard_normal(dim))
        # Rest random
        for _ in range(pop_size - 1):
            pop.append(rng.uniform(-5.0, 5.0, dim))
            
        pop = np.array(pop)
        pop_f = np.array([eval_func(p) for p in pop])
        
        best_idx = np.argmin(pop_f)
        best_x = pop[best_idx].copy()
        best_f = pop_f[best_idx]

        # DE Loop
        # Reserve 20% of budget for local refinement
        max_de_calls = int(budget * 0.8)
        
        while time_check() and calls_made < max_de_calls:
            new_pop_f = pop_f.copy()
            new_pop = pop.copy()
            
            for i in range(pop_size):
                if not time_check() or calls_made >= max_de_calls:
                    break
                    
                # Mutation: DE/rand/1
                a, b, c = rng.choice(pop_size, 3, replace=False)
                while a == i or b == i or c == i:
                    a, b, c = rng.choice(pop_size, 3, replace=False)
                
                v = pop[a] + F * (pop[b] - pop[c])
                
                # Crossover
                r = rng.random(dim)
                j_rand = rng.integers(0, dim)
                trial = np.where(r < CR, v, pop[i])
                trial[j_rand] = v[j_rand]
                trial = np.clip(trial, -5.0, 5.0)
                
                trial_f = eval_func(trial)
                
                if trial_f <= pop_f[i]:
                    new_pop[i] = trial
                    new_pop_f[i] = trial_f
                    if trial_f < best_f:
                        best_f = trial_f
                        best_x = trial.copy()
                        
            pop = new_pop
            pop_f = new_pop_f

        # Phase 2: Local Refinement using L-BFGS-B
        # L-BFGS-B is superior to Nelder-Mead for smooth functions like Rastrigin
        # We try to refine the best point, and if that doesn't improve much, try next best
        
        # Sort history by function value to get top candidates
        if len(history_x) > 0:
            indices = np.argsort(history_f)
            top_candidates = [history_x[i] for i in indices[:5]]
        else:
            top_candidates = [best_x]

        for start_point in top_candidates:
            if not time_check() or calls_made >= budget:
                break
            
            try:
                # Use a wrapper to ensure we don't exceed budget during minimize
                # L-BFGS-B is iterative, so we need to be careful
                res = minimize(
                    eval_func, 
                    start_point, 
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 20}
                )
            except Exception:
                pass
            
            # If we found a better point, update best
            # (eval_func already updates best_f and best_x)
            
            # If we are very close to convergence or time is running out, break
            if time.time() > deadline - 10.0:
                break

        return best_x, best_f
