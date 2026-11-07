import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Parameters
        N_p = 10  # Population size
        F = 0.8   # Scale factor
        CR = 0.9  # Crossover rate
        
        # Time guard
        start_time = time.time()
        deadline = start_time + 280  # 280s safety margin
        
        rng = np.random.default_rng(0)
        
        # Initialization
        # Generate population
        X = rng.uniform(-5, 5, (N_p, self.dim))
        F_vals = np.array([func(x) for x in X])
        
        # Track best
        best_idx = np.argmin(F_vals)
        best_x = X[best_idx].copy()
        best_f = F_vals[best_idx]
        
        calls_made = N_p
        budget_remaining = self.budget - calls_made
        
        # Differential Evolution Loop
        # Each generation consumes N_p calls
        # We can run budget_remaining // N_p generations
        # But we need to handle the remainder if any? 
        # The seed program used exactly budget calls.
        # If we do full generations, we might waste calls if budget is not divisible by N_p.
        # 500 / 10 = 50 generations. 10 init + 50*10 = 510? No.
        # Init: 10 calls.
        # Gen 1: 10 calls.
        # ...
        # Gen 49: 10 calls.
        # Total: 10 + 49*10 = 500.
        # So we run 49 generations after init.
        
        num_generations = budget_remaining // N_p
        
        for gen in range(num_generations):
            # Check time
            if time.time() > deadline:
                break
                
            for i in range(N_p):
                # Mutation
                # Pick 3 distinct indices different from i
                # Simple way: shuffle indices 0..N_p-1 excluding i, pick first 3
                # Or just random choice with replacement and check? 
                # Efficient: rng.choice without replacement
                # Since N_p is small, we can just pick randomly and ensure distinctness
                r1 = rng.integers(0, N_p)
                while r1 == i:
                    r1 = rng.integers(0, N_p)
                r2 = rng.integers(0, N_p)
                while r2 == i or r2 == r1:
                    r2 = rng.integers(0, N_p)
                r3 = rng.integers(0, N_p)
                while r3 == i or r3 == r1 or r3 == r2:
                    r3 = rng.integers(0, N_p)
                    
                v = X[r1] + F * (X[r2] - X[r3])
                v = np.clip(v, -5, 5)
                
                # Crossover
                u = X[i].copy()
                # Determine crossover mask
                j_rand = rng.integers(0, self.dim)
                mask = rng.random(self.dim) < CR
                mask[j_rand] = True # Ensure at least one gene from v
                
                u[mask] = v[mask]
                
                # Evaluation
                fu = func(u)
                calls_made += 1
                
                # Selection
                if fu < F_vals[i]:
                    X[i] = u
                    F_vals[i] = fu
                    
                    if fu < best_f:
                        best_f = fu
                        best_x = u.copy()
                        
        return best_x, best_f
