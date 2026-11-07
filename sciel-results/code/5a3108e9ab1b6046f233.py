"""Optimized program for BBOB f21 (Gallagher 101 peaks) with Differential Evolution."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Differential Evolution optimizer for BBOB f21."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 325  # Safety margin before 330s kill
        
        dim = self.dim
        budget = self.budget
        n_calls = 0
        best_f = float('inf')
        best_x = None
        
        # Wrapped function to track calls and time
        def wrapped_func(x):
            nonlocal n_calls, best_f, best_x
            if n_calls >= budget:
                return 1e18
            if time.time() > deadline - 1:
                return 1e18
            n_calls += 1
            val = func(x)
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        # Differential Evolution Parameters
        # Population size typically 5-10 * dim. For dim=5, let's use 20.
        pop_size = 20
        F = 0.8  # Scale factor
        CR = 0.9  # Crossover rate
        
        # Initialize population randomly within bounds
        # We can bias slightly towards center or use full range [-5, 5]
        # The problem states minimizer is in [-4, 4]^5, but box is [-5, 5]
        # Let's initialize in [-4.5, 4.5] to be safe but broad
        rng = np.random.default_rng(42)
        
        # Initial population
        pop = rng.uniform(-4.5, 4.5, (pop_size, dim))
        
        # Evaluate initial population
        pop_f = np.array([wrapped_func(ind) for ind in pop])
        
        # Sort initial population
        idx = np.argsort(pop_f)
        pop = pop[idx]
        pop_f = pop_f[idx]
        
        # Update global best
        if pop_f[0] < best_f:
            best_f = pop_f[0]
            best_x = pop[0].copy()

        # DE Loop
        # We will run until budget or time is exhausted
        # Each generation costs pop_size evaluations (roughly, plus trials)
        # Actually, DE evaluates pop_size trial vectors per generation.
        # Total budget is 500. Pop size 20. So ~25 generations.
        
        while n_calls < budget and time.time() < deadline - 5:
            # Check if we have enough budget for a full generation
            if n_calls + pop_size > budget:
                # Do partial generation or stop? 
                # Better to just stop if we can't finish a gen, or do partial.
                # For simplicity and robustness, let's just break if we can't do a full gen 
                # to keep the algorithm clean, or handle partial.
                # Actually, we can just let the loop run and the wrapper will stop it.
                # But it's cleaner to check.
                pass
            
            for i in range(pop_size):
                if n_calls >= budget or time.time() > deadline - 5:
                    break
                    
                # Select 3 distinct random individuals
                # i is the target
                # a, b, c are distinct from i and each other
                candidates = [j for j in range(pop_size) if j != i]
                a, b, c = rng.choice(candidates, 3, replace=False)
                
                # Create trial vector
                # v = pop[a] + F * (pop[b] - pop[c])
                v = pop[a] + F * (pop[b] - pop[c])
                
                # Crossover: binomial
                # r = rng.random(dim)
                # j_rand = rng.randint(dim)
                # trial = np.where(r < CR, v, pop[i])
                # trial[j_rand] = v[j_rand]
                
                # Vectorized crossover
                r = rng.random(dim)
                j_rand = rng.integers(dim)
                trial = np.where(r < CR, v, pop[i])
                trial[j_rand] = v[j_rand]
                
                # Clip to bounds
                trial = np.clip(trial, -5, 5)
                
                # Evaluate trial
                f_trial = wrapped_func(trial)
                
                # Selection
                if f_trial <= pop_f[i]:
                    pop[i] = trial
                    pop_f[i] = f_trial

            # Sort population after each generation to keep track of best
            # This is optional for standard DE but helps if we use strategies that rely on rank
            # Here we just use random selection, so sorting isn't strictly required for the next gen's selection
            # but it helps for debugging or if we change strategy. 
            # However, sorting every gen costs O(N log N) which is negligible for N=20.
            idx = np.argsort(pop_f)
            pop = pop[idx]
            pop_f = pop_f[idx]
            
            # Update global best if the local best improved
            if pop_f[0] < best_f:
                best_f = pop_f[0]
                best_x = pop[0].copy()
                
            # Optional: Adapt F or CR? 
            # For a fixed budget, standard DE/rand/1/bin is robust.
            # We could add a restart mechanism if stagnation is detected.
            # Check stagnation: if best_f hasn't changed in last 10 generations?
            # Let's keep it simple for now.

        return best_x, best_f
# EVOLVE-BLOCK-END
