import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        d = self.dim
        total_budget = self.budget
        calls_used = 0
        
        best_x = None
        best_f = np.inf
        
        # Multi-Start DE
        n_restarts = 3
        budget_per_restart = total_budget // n_restarts
        # Adjust last restart to use remaining budget
        budgets = [budget_per_restart] * n_restarts
        budgets[-1] += total_budget - sum(budgets)
        
        pop_size = 10
        F = 0.8
        CR = 0.9
        
        for r in range(n_restarts):
            b = budgets[r]
            if b <= 0:
                break
            
            # Initialize population for this restart
            # Use random initialization within [-5, 5]
            pop = rng.uniform(-5, 5, size=(pop_size, d))
            f_pop = np.empty(pop_size)
            
            # Evaluate initial population
            for i in range(pop_size):
                if calls_used >= total_budget:
                    break
                f_pop[i] = func(pop[i])
                calls_used += 1
                if f_pop[i] < best_f:
                    best_f = f_pop[i]
                    best_x = pop[i].copy()
            
            if calls_used >= total_budget:
                break
            
            # DE Main Loop
            while calls_used < total_budget:
                # Check if we can afford a full generation
                # We need pop_size evaluations
                if calls_used + pop_size > total_budget:
                    # Do partial generation?
                    # Just break to avoid wasting calls on incomplete generations
                    # Or run until budget is exactly hit
                    pass
                
                for i in range(pop_size):
                    if calls_used >= total_budget:
                        break
                    
                    # Select 3 distinct random individuals
                    indices = rng.choice(pop_size, size=3, replace=False)
                    r1, r2, r3 = indices
                    while r1 == i or r2 == i or r3 == i:
                        indices = rng.choice(pop_size, size=3, replace=False)
                        r1, r2, r3 = indices
                        
                    # Mutation
                    v = pop[r1] + F * (pop[r2] - pop[r3])
                    v = np.clip(v, -5, 5)
                    
                    # Crossover
                    u = pop[i].copy()
                    j_rand = rng.integers(0, d)
                    for j in range(d):
                        if rng.random() < CR or j == j_rand:
                            u[j] = v[j]
                        else:
                            u[j] = pop[i][j]
                    
                    # Selection
                    f_u = func(u)
                    calls_used += 1
                    
                    if f_u < f_pop[i]:
                        pop[i] = u
                        f_pop[i] = f_u
                        if f_u < best_f:
                            best_f = f_u
                            best_x = u.copy()
                
                # Optional: Check for convergence
                # If std of f_pop is very small, break
                if np.std(f_pop) < 1e-8:
                    break
                    
        return best_x, best_f
