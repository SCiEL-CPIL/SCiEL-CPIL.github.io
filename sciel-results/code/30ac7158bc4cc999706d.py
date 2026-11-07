"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Differential Evolution optimizer for BBOB-style functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 300s limit
        
        dim = self.dim
        budget = self.budget
        
        # Population size: Large enough for exploration, small enough to fit budget
        # 500 budget / ~20-30 generations = ~15-25 pop size? 
        # Actually, standard DE uses NP ~ 10*D. For D=5, NP=50 is good.
        # 500 calls / 50 pop = 10 generations. This is very few.
        # Let's use a smaller population to allow more generations, or a larger one for better initial best.
        # AOCC rewards early progress. A larger initial population gives a better b_1.
        # Let's try NP = 40. 500 / 40 = 12.5 generations.
        # We can also do a "restart" strategy if it stagnates.
        
        # Let's stick to a robust single run with adaptive parameters or a simple DE.
        # Given the low budget (500), a simple DE with a decent NP is best.
        
        np_pop = 40
        
        rng = np.random.default_rng(42)
        
        # Initialize population uniformly in the box
        pop = rng.uniform(self.lb, self.ub, size=(np_pop, dim))
        
        # Evaluate initial population
        # We must be careful with the budget.
        # Calls so far: 0
        best_x = None
        best_f = float('inf')
        evals = 0
        
        def evaluate(x):
            nonlocal best_x, best_f, evals
            f_val = func(x)
            evals += 1
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val
        
        # Initial evaluation
        for i in range(np_pop):
            if evals >= budget:
                break
            if time.time() > deadline:
                break
            f_i = evaluate(pop[i])
            pop[i, :dim] = pop[i] # Ensure shape consistency if needed, though pop is already correct
        
        # Sort initial population by fitness
        # We need to store fitnesses
        fits = np.array([func(pop[i]) for i in range(np_pop) if evals < budget]) 
        # Wait, the above loop already evaluated them. Let's store them properly.
        
        # Redo initialization properly to track fitness
        pop = rng.uniform(self.lb, self.ub, size=(np_pop, dim))
        fits = np.empty(np_pop)
        for i in range(np_pop):
            if evals >= budget:
                break
            fits[i] = evaluate(pop[i])
            
        # If we ran out of budget during init, return best
        if evals >= budget:
            return best_x, best_f

        # Main DE loop
        # Parameters
        F = 0.8  # Mutation factor
        CR = 0.9  # Crossover rate
        
        # We have budget - evals left.
        # Each generation consumes np_pop evaluations.
        
        while evals < budget and time.time() < deadline:
            # Check if we can afford another generation
            if evals + np_pop > budget:
                # If we can't afford a full generation, we might do partial or just stop.
                # For simplicity and to avoid complex partial logic, we stop if a full gen isn't possible.
                # However, we can utilize remaining budget for a local search on the best point?
                # Or just stop.
                break
                
            # Sort population by fitness
            idx = np.argsort(fits)
            pop = pop[idx]
            fits = fits[idx]
            
            # Update best if needed (should be tracked in evaluate, but ensure consistency)
            if fits[0] < best_f:
                best_f = fits[0]
                best_x = pop[0].copy()
                
            # Generate offspring
            new_pop = np.empty_like(pop)
            new_fits = np.empty(np_pop)
            
            for i in range(np_pop):
                if evals + 1 > budget:
                    break
                if time.time() > deadline:
                    break
                    
                # DE/rand/1/bin
                # Pick 3 distinct random indices different from i
                r1, r2, r3 = rng.choice(np_pop, size=3, replace=False)
                while r1 == i or r2 == i or r3 == i:
                    r1, r2, r3 = rng.choice(np_pop, size=3, replace=False)
                    
                # Mutation
                v = pop[r1] + F * (pop[r2] - pop[r3])
                
                # Crossover
                # j_rand is a random dimension to ensure at least one gene is from v
                j_rand = rng.integers(0, dim)
                trial = np.empty(dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                    else:
                        trial[j] = pop[i][j]
                
                # Clip to bounds
                trial = np.clip(trial, self.lb, self.ub)
                
                # Evaluate
                f_trial = evaluate(trial)
                
                # Selection
                if f_trial <= fits[i]:
                    new_pop[i] = trial
                    new_fits[i] = f_trial
                else:
                    new_pop[i] = pop[i]
                    new_fits[i] = fits[i]
                    
            pop = new_pop
            fits = new_fits
            
            # Optional: Adaptive F and CR? 
            # For simplicity and robustness, keep fixed.
            
            # Check for stagnation? 
            # If best_f hasn't improved for X generations, we could restart or increase F.
            # Given the small budget, simple DE is often sufficient.
            
        return best_x, best_f

# EVOLVE-BLOCK-END
