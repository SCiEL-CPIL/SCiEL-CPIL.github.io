# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Differential Evolution optimizer with deadline guard for 5D box-constrained problems."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: evaluator kills after 330s. We return well before that.
        deadline = start_time + 320.0

        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)

        # DE Parameters
        # Population size: 10 * n is a common heuristic, but for small n, 10-20 is good.
        # Let's use 15 for n=5.
        NP = max(10, 3 * n) 
        # Ensure we don't use too many evals in one generation if budget is small
        # But budget is 500, so 15 per gen is fine.
        
        # Scale factor and Crossover rate
        F = 0.8
        CR = 0.5

        # Initialize population uniformly in [-5, 5]
        # Use a grid + random jitter to ensure coverage? Or just random.
        # Random is fine for DE.
        pop = rng.uniform(-5, 5, size=(NP, n))
        fvals = np.array([func(x) for x in pop])
        evals = NP
        
        # Track best
        best_idx = np.argmin(fvals)
        best_x = pop[best_idx].copy()
        best_f = fvals[best_idx]

        # Main loop
        while evals < budget:
            # Check deadline
            if time.time() > deadline:
                break
            
            # Generate new population
            new_pop = np.zeros_like(pop)
            new_fvals = np.zeros(NP)
            
            for i in range(NP):
                if evals >= budget:
                    break
                
                # Select three distinct random individuals r1, r2, r3 different from i
                # Simple method: sample 3 indices, ensure they are not i
                r_indices = rng.choice(NP, size=3, replace=False)
                # If any is i, resample (unlikely with NP>=10, but safe)
                while i in r_indices:
                    r_indices = rng.choice(NP, size=3, replace=False)
                
                r1, r2, r3 = r_indices
                
                # Mutation: v = x_r1 + F * (x_r2 - x_r3)
                v = pop[r1] + F * (pop[r2] - pop[r3])
                
                # Clip to box
                v = np.clip(v, -5, 5)
                
                # Crossover: binomial crossover
                # Jrand: random index
                Jrand = rng.integers(0, n)
                for j in range(n):
                    if rng.random() < CR or j == Jrand:
                        new_pop[i][j] = v[j]
                    else:
                        new_pop[i][j] = pop[i][j]
                
                # Evaluate trial individual
                fy = func(new_pop[i])
                evals += 1
                new_fvals[i] = fy
                
                # Selection
                if fy <= fvals[i]:
                    pop[i] = new_pop[i]
                    fvals[i] = fy
                    if fy < best_f:
                        best_f = fy
                        best_x = new_pop[i].copy()
                # else: keep old pop[i]
            
            # Update best in case we broke early
            current_best_idx = np.argmin(fvals)
            if fvals[current_best_idx] < best_f:
                best_f = fvals[current_best_idx]
                best_x = pop[current_best_idx].copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
