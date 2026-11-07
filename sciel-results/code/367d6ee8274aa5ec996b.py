import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: Evaluator kills at 330s. We stop at 290s to be safe.
        timeout = 290.0
        
        n = self.dim
        budget = self.budget
        
        # Use a fixed seed for reproducibility if needed, but standard random is fine.
        # However, deterministic behavior might help with debugging, but here we just run.
        rng = np.random.default_rng(42)

        # --- Phase 1: Differential Evolution (DE) for global exploration ---
        # DE maintains a diverse population, helping to escape local minima in Büche-Rastrigin.
        # Population size P=10 is a good balance for dim=5 and budget=500.
        P = 10
        F = 0.5  # Mutation strength
        CR = 0.9  # Crossover rate
        
        # Initialize population uniformly in [-4, 4] (since min is in [-4,4])
        # Actually, box is [-5,5], but min is in [-4,4]. Uniform in [-4,4] is safer to avoid boundary issues initially.
        # Let's use [-4, 4] for initialization to stay close to the likely region.
        pop = rng.uniform(-4.0, 4.0, size=(P, n))
        
        # Evaluate initial population
        fitness = np.zeros(P)
        calls = 0
        
        for i in range(P):
            if calls >= budget or time.time() - start_time > timeout:
                break
            pop[i] = np.clip(pop[i], -5, 5)
            fitness[i] = func(pop[i])
            calls += 1
            
        if calls == 0:
            return np.zeros(n), float('inf')
        
        # Track global best
        best_idx = np.argmin(fitness)
        global_best_x = pop[best_idx].copy()
        global_best_f = fitness[best_idx]
        
        # DE Loop
        # We use ~70% of budget for DE
        de_budget = int(budget * 0.7)
        
        while calls < de_budget:
            if time.time() - start_time > timeout:
                break
                
            for i in range(P):
                if calls >= de_budget or time.time() - start_time > timeout:
                    break
                    
                # Select 3 distinct random individuals
                r1, r2, r3 = rng.choice(P, size=3, replace=False)
                
                # Mutation: v = x_r1 + F * (x_r2 - x_r3)
                v = pop[r1] + F * (pop[r2] - pop[r3])
                
                # Crossover: binomial
                j_rand = rng.integers(0, n)
                trial = pop[i].copy()
                for j in range(n):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                    
                trial = np.clip(trial, -5, 5)
                
                # Evaluate
                f_trial = func(trial)
                calls += 1
                
                # Selection
                if f_trial < fitness[i]:
                    pop[i] = trial
                    fitness[i] = f_trial
                    if f_trial < global_best_f:
                        global_best_f = f_trial
                        global_best_x = trial.copy()
                        
        # --- Phase 2: (1+1)-ES Polish for local refinement ---
        # Use the remaining ~30% of budget to refine the best point found by DE.
        # This mimics the high-scoring inspiration's polish phase.
        
        if calls < budget:
            x = global_best_x.copy()
            fx = global_best_f
            
            # Start with a small sigma for polishing
            sigma = 0.2
            min_sigma = 1e-6
            max_sigma = 1.0
            success_factor = 1.2
            failure_factor = 0.8
            
            while calls < budget:
                if time.time() - start_time > timeout:
                    break
                    
                y = x + sigma * rng.standard_normal(n)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls += 1
                
                if fy < fx:
                    x = y
                    fx = fy
                    sigma = min(sigma * success_factor, max_sigma)
                    if fy < global_best_f:
                        global_best_f = fy
                        global_best_x = y.copy()
                else:
                    sigma = max(sigma * failure_factor, min_sigma)
                    
                # If sigma becomes very small, we've likely converged to a local min.
                # We could restart, but with limited budget left, just continuing or breaking is fine.
                # Given the polish phase is short, we'll just let it run out.
                
        return global_best_x, global_best_f
