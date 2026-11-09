import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget

        if budget < 10:
            rng = np.random.default_rng(0)
            best_x = rng.uniform(-5, 5, n)
            best_f = func(best_x)
            for _ in range(budget - 1):
                x = rng.uniform(-5, 5, n)
                fx = func(x)
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            return best_x, best_f

        # DE Parameters
        # N_p = 20 is a good balance for 5D with ~500 budget
        N_p = 20
        F = 0.7
        CR = 0.9

        rng = np.random.default_rng(12345)

        # Initialize population
        pop = rng.uniform(-5, 5, (N_p, n))
        fpop = np.array([func(x) for x in pop])
        calls_made = N_p

        best_idx = np.argmin(fpop)
        best_x = pop[best_idx].copy()
        best_f = fpop[best_idx]

        # Pre-generate random indices for efficiency?
        # Or just use rng inside.
        
        while calls_made < budget:
            # Determine how many individuals we can evaluate in this "generation"
            # We process one individual at a time to handle budget strictly
            # But DE is usually synchronous. Let's do synchronous generations if possible.
            
            if calls_made + N_p <= budget:
                # Synchronous generation
                new_pop = pop.copy()
                new_fpop = fpop.copy()
                
                for i in range(N_p):
                    # Select 3 distinct parents
                    a, b, c = rng.choice(N_p, 3, replace=False)
                    
                    # Mutation
                    v = pop[a] + F * (pop[b] - pop[c])
                    v = np.clip(v, -5, 5)
                    
                    # Crossover
                    mask = rng.random(n) < CR
                    j_rand = rng.integers(0, n)
                    mask[j_rand] = True
                    u = np.where(mask, v, pop[i])
                    u = np.clip(u, -5, 5)
                    
                    # Evaluate
                    fu = func(u)
                    calls_made += 1
                    
                    # Selection
                    if fu <= new_fpop[i]:
                        new_pop[i] = u
                        new_fpop[i] = fu
                        if fu < best_f:
                            best_f = fu
                            best_x = u.copy()
                
                pop = new_pop
                fpop = new_fpop
            else:
                # Asynchronous / partial generation
                for i in range(N_p):
                    if calls_made >= budget:
                        break
                    
                    a, b, c = rng.choice(N_p, 3, replace=False)
                    
                    v = pop[a] + F * (pop[b] - pop[c])
                    v = np.clip(v, -5, 5)
                    
                    mask = rng.random(n) < CR
                    j_rand = rng.integers(0, n)
                    mask[j_rand] = True
                    u = np.where(mask, v, pop[i])
                    u = np.clip(u, -5, 5)
                    
                    fu = func(u)
                    calls_made += 1
                    
                    if fu <= fpop[i]:
                        pop[i] = u
                        fpop[i] = fu
                        if fu < best_f:
                            best_f = fu
                            best_x = u.copy()
                
                break # Exit loop as budget is exhausted or near it

        return best_x, best_f
