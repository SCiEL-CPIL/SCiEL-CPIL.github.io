"""Alternative strategy: Differential Evolution (DE) with adaptive parameters."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Differential Evolution optimizer for BBOB f13 (Sharp Ridge)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # Deadline guard: 330s limit, leave 10s margin
        start_time = time.time()
        deadline = start_time + 320.0

        calls_used = 0
        best_x = None
        best_f = np.inf

        def eval_point(x):
            nonlocal best_x, best_f, calls_used
            if calls_used >= budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            fx = func(x)
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Population size
        # For dim=5, a population size of 10-20 is reasonable.
        # Let's use 10 to keep evaluation cost low per generation (10 evals/gen)
        # allowing for ~40-50 generations if budget allows.
        pop_size = 10
        
        # Initialize population uniformly in [-4, 4]^5
        # The problem states minimizer is in [-4, 4]^5
        pop = np.random.uniform(-4, 4, (pop_size, dim))
        
        # Evaluate initial population
        pop_f = np.zeros(pop_size)
        for i in range(pop_size):
            if calls_used >= budget or time.time() > deadline:
                break
            pop_f[i] = eval_point(pop[i])
        
        # If we didn't even fill the population, return best found so far
        if calls_used >= budget or time.time() > deadline:
            if best_x is None:
                best_x = np.zeros(dim)
                best_f = func(best_x)
            return best_x, best_f

        # DE Parameters
        # F: Scale factor for differential variation
        # CR: Crossover rate
        # Standard DE/rand/1/bin
        # We can adapt F and CR, but let's start with standard values
        F = 0.8
        CR = 0.5
        
        # Adaptive F: Start high for exploration, decrease for exploitation?
        # Or keep constant? Standard DE often works well with constant F.
        # For "Sharp Ridge", we might need larger steps initially.
        
        max_generations = 1000 # Will be limited by budget/deadline
        
        for gen in range(max_generations):
            if calls_used >= budget or time.time() > deadline:
                break
            
            # Sort population to keep track of best
            # For DE, we usually maintain a population list.
            # We'll update the population in-place.
            
            # For each individual, generate a trial vector
            for i in range(pop_size):
                if calls_used >= budget or time.time() > deadline:
                    break
                
                # Select 3 distinct random individuals distinct from i
                # a, b, c
                indices = np.random.choice(pop_size, size=3, replace=False)
                # Ensure none of them are i
                while i in indices:
                    indices = np.random.choice(pop_size, size=3, replace=False)
                
                a, b, c = indices
                
                # Differential variation
                v = pop[a] + F * (pop[b] - pop[c])
                
                # Clip to bounds
                v = np.clip(v, -5, 5)
                
                # Crossover
                r = np.random.rand()
                j_rand = np.random.randint(dim)
                trial = np.zeros(dim)
                for j in range(dim):
                    if r <= CR or j == j_rand:
                        trial[j] = v[j]
                    else:
                        trial[j] = pop[i][j]
                
                # Evaluate trial
                f_trial = eval_point(trial)
                
                if calls_used >= budget or time.time() > deadline:
                    break
                
                # Selection
                if f_trial < pop_f[i]:
                    pop[i] = trial
                    pop_f[i] = f_trial

        # Return best found
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
