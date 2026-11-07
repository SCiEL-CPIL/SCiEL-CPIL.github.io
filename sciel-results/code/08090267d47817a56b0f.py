import numpy as np

class Optimizer:
    """Differential Evolution (DE) for BBOB f4."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.NP = 10  # Population size
        self.F = 0.8  # Mutation scale factor
        self.CR = 0.9  # Crossover rate

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        NP = self.NP
        budget = self.budget
        
        lower = -5.0
        upper = 5.0
        
        # Initialize population
        pop = rng.uniform(lower, upper, (NP, dim))
        fvals = np.array([func(x) for x in pop])
        budget -= NP
        
        gbest_idx = np.argmin(fvals)
        gbest = pop[gbest_idx].copy()
        gbest_fval = fvals[gbest_idx]
        
        if budget <= 0:
            return gbest, gbest_fval
        
        # DE loop
        while budget > 0:
            # We need NP evaluations per generation
            if budget < NP:
                # Not enough for a full generation, break or do partial?
                # Standard DE requires full population update.
                # If we can't do a full generation, we can stop.
                break
                
            for i in range(NP):
                if budget <= 0:
                    break
                
                # Select 3 distinct individuals a, b, c different from i
                # Simple method: sample 3 indices from 0..NP-1 excluding i
                indices = [j for j in range(NP) if j != i]
                a, b, c = rng.choice(indices, size=3, replace=False)
                
                # Mutation
                v = pop[a] + self.F * (pop[b] - pop[c])
                
                # Crossover
                u = pop[i].copy()
                j_rand = rng.integers(dim)
                for j in range(dim):
                    if rng.random() < self.CR or j == j_rand:
                        u[j] = v[j]
                
                # Clip to bounds
                u = np.clip(u, lower, upper)
                
                # Evaluate
                f_u = func(u)
                budget -= 1
                
                # Selection
                if f_u < fvals[i]:
                    fvals[i] = f_u
                    pop[i] = u
                    
                    if f_u < gbest_fval:
                        gbest_fval = f_u
                        gbest = u.copy()

        return gbest, gbest_fval
