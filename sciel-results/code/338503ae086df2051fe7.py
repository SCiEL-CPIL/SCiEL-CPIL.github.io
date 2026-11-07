"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Differential Evolution for global search followed by (1+1)-ES for local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin before 300s limit
        dim = self.dim
        calls = 0
        max_calls = self.budget

        def eval_func(x):
            nonlocal calls
            if calls >= max_calls or time.time() > deadline - 2.0:
                return None
            calls += 1
            return func(x)

        # Phase 1: Differential Evolution
        # Population size 20, DE parameters F=0.8, CR=0.9
        N = 20
        F = 0.8
        CR = 0.9
        
        # Initialize population
        # Include origin and random points
        pop = np.random.uniform(-5, 5, (N, dim))
        pop[0] = 0.0
        
        f_pop = np.empty(N)
        for i in range(N):
            f_val = eval_func(pop[i])
            if f_val is None:
                # Fallback if time/budget ran out immediately (unlikely)
                return pop[0], 0.0 
            f_pop[i] = f_val
            
        best_idx = np.argmin(f_pop)
        best_x = pop[best_idx].copy()
        best_f = f_pop[best_idx]

        # DE Loop
        # Target ~400 calls for DE phase
        de_target = 400
        
        while calls < de_target and time.time() < deadline - 2.0:
            improved = False
            for i in range(N):
                if calls >= de_target or time.time() > deadline - 2.0:
                    break
                
                # Select 3 distinct individuals different from i
                candidates = [j for j in range(N) if j != i]
                r1, r2, r3 = np.random.choice(candidates, 3, replace=False)
                
                # Mutation
                v = pop[r1] + F * (pop[r2] - pop[r3])
                v = np.clip(v, -5, 5)
                
                # Crossover
                # Binary crossover
                mask = np.random.random(dim) < CR
                if not np.any(mask):
                    mask[np.random.randint(dim)] = True
                trial = np.where(mask, v, pop[i])
                trial = np.clip(trial, -5, 5)
                
                f_trial = eval_func(trial)
                if f_trial is None:
                    break
                
                if f_trial < f_pop[i]:
                    pop[i] = trial
                    f_pop[i] = f_trial
                    improved = True
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()
                        
            # If no improvement in a full generation, we could break or continue
            # DE usually continues to explore. We'll continue until budget.

        # Phase 2: Local Polish with (1+1)-ES
        # Use remaining budget
        x = best_x.copy()
        fx = best_f
        sigma = 0.1
        
        # Adaptive sigma (1+1)-ES
        while calls < max_calls and time.time() < deadline - 2.0:
            y = np.clip(x + sigma * np.random.standard_normal(dim), -5, 5)
            fy = eval_func(y)
            if fy is None:
                break
            
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.15
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            else:
                sigma *= 0.85
                
            sigma = np.clip(sigma, 0.001, 3.0)

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
