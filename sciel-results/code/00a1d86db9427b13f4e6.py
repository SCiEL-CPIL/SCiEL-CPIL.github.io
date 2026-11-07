"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Differential Evolution for BBOB f20 (Schwefel)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.start_time = None

    def __call__(self, func):
        self.start_time = time.time()
        dim = self.dim
        budget = self.budget
        deadline = self.start_time + 290  # Safety margin

        rng = np.random.default_rng(42)
        
        # DE Parameters
        # Population size N=50 is a good balance for 5D.
        # It uses 50 calls for init, leaving 450 for evolution.
        N = 50 
        F = 0.8  # Scaling factor
        CR = 0.9  # Crossover rate
        
        # Check if budget is sufficient for DE
        if budget < N + 1:
            # Fallback to random search if budget is too small
            best_x = None
            best_f = np.inf
            for _ in range(budget):
                if time.time() > deadline:
                    break
                x = rng.uniform(-5, 5, dim)
                fx = func(x)
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            return best_x, best_f

        # Initialize population
        # Minimizer is in [-4, 4]^5, but box is [-5, 5]^5.
        # We sample from the full box to be safe, but biasing slightly to center might help?
        # Actually, standard DE works well with uniform sampling over the domain.
        pop = rng.uniform(-5, 5, size=(N, dim))
        fits = np.array([func(x) for x in pop])
        
        best_idx = np.argmin(fits)
        best_x = pop[best_idx].copy()
        best_f = fits[best_idx]
        
        calls_made = N
        
        # DE Main Loop
        # We generate one offspring per generation (1+1 style DE? No, standard DE replaces the whole pop).
        # To maximize early progress, we should update the population efficiently.
        # Standard DE: For each individual i, create trial vector. If better, replace.
        # This uses N evaluations per generation.
        # Total generations = (budget - N) // N
        
        while calls_made < budget:
            if time.time() > deadline:
                break
            
            # Check if we have enough budget for a full generation
            if calls_made + N > budget:
                # Do partial generation to use remaining budget
                remaining = budget - calls_made
                for i in range(remaining):
                    if time.time() > deadline:
                        break
                    
                    # Select 3 distinct random indices different from i
                    while True:
                        r1, r2, r3 = rng.choice(N, size=3, replace=False)
                        if r1 != i and r2 != i and r3 != i and len(set([r1, r2, r3, i])) == 4:
                            break
                    
                    # Mutation: DE/rand/1
                    v = pop[r1] + F * (pop[r2] - pop[r3])
                    
                    # Crossover: Binomial
                    j_rand = rng.integers(dim)
                    trial = np.empty_like(pop[i])
                    for j in range(dim):
                        if rng.random() < CR or j == j_rand:
                            trial[j] = v[j]
                        else:
                            trial[j] = pop[i][j]
                    
                    # Clip to bounds
                    trial = np.clip(trial, -5, 5)
                    
                    # Evaluate
                    f_trial = func(trial)
                    calls_made += 1
                    
                    # Selection
                    if f_trial <= fits[i]:
                        pop[i] = trial
                        fits[i] = f_trial
                        
                        # Update global best
                        if f_trial < best_f:
                            best_f = f_trial
                            best_x = trial.copy()
                
                break # Exit main loop if we exhausted budget in partial gen
            
            # Full Generation
            for i in range(N):
                if time.time() > deadline or calls_made >= budget:
                    break
                
                # Select 3 distinct random indices different from i
                # Optimization: Pre-generate randoms or use simple logic
                # For N=50, finding 3 distinct is fast
                while True:
                    r1, r2, r3 = rng.choice(N, size=3, replace=False)
                    if r1 != i and r2 != i and r3 != i and r1 != r2 and r1 != r3 and r2 != r3:
                        break
                
                # Mutation: DE/rand/1
                v = pop[r1] + F * (pop[r2] - pop[r3])
                
                # Crossover: Binomial
                j_rand = rng.integers(dim)
                trial = np.empty_like(pop[i])
                # Vectorized crossover for speed
                mask = rng.random(dim) < CR
                mask[j_rand] = True
                trial = np.where(mask, v, pop[i])
                
                # Clip to bounds
                trial = np.clip(trial, -5, 5)
                
                # Evaluate
                f_trial = func(trial)
                calls_made += 1
                
                # Selection
                if f_trial <= fits[i]:
                    pop[i] = trial
                    fits[i] = f_trial
                    
                    # Update global best
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()
                        
        return best_x, best_f
# EVOLVE-BLOCK-END
