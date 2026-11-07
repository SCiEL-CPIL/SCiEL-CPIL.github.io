"""Alternative strategy: Global Differential Evolution for BBOB f13."""
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
            if calls_used >= budget or time.time() > deadline:
                return np.inf
            fx = func(x)
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # DE Parameters
        # Population size: typically 5-10 * dim. Let's start with 10 * dim = 50.
        # But budget is 500. 500 / 50 = 10 generations. This is very low.
        # We need a smaller population or fewer generations?
        # If pop=10, generations=50.
        # If pop=5, generations=100.
        # Let's try pop = 2 * dim = 10.
        
        pop_size = max(2 * dim, 10) 
        # Ensure pop_size fits in budget with at least 1 generation
        if pop_size > budget:
            pop_size = budget

        # Initialize population
        # Uniform in [-5, 5]
        # Since minimizer is in [-4, 4], this is fine.
        # Clip to [-5, 5] as per problem statement.
        
        pop = np.random.uniform(-5, 5, (pop_size, dim))
        pop_f = np.array([eval_point(x) for x in pop])
        
        if calls_used >= budget or time.time() > deadline:
            return best_x, best_f

        # DE Parameters
        # F: scaling factor (0.5 - 1.0)
        # CR: crossover rate (0.1 - 0.9)
        # Standard DE/rand/1
        F = 0.8
        CR = 0.5
        
        # Adaptive CR? Maybe keep simple first.
        
        while calls_used < budget and time.time() < deadline:
            # Sort population to find best
            # Actually, DE usually keeps the best from previous gen and current gen.
            # Let's do a standard DE loop.
            
            new_pop = np.zeros_like(pop)
            new_pop_f = np.zeros_like(pop_f)
            
            # Copy current pop to new pop initially?
            # Or evaluate all offspring and select best of (parent, child)?
            # Standard DE: generate offspring for each individual, select best.
            
            for i in range(pop_size):
                if calls_used >= budget or time.time() > deadline:
                    break
                
                # Select 3 distinct random individuals r1, r2, r3 != i
                # Simple selection without replacement
                r1, r2, r3 = np.random.choice(pop_size, 3, replace=False)
                while r1 == i:
                    r1 = np.random.randint(pop_size)
                while r2 == i or r2 == r1:
                    r2 = np.random.randint(pop_size)
                while r3 == i or r3 == r1 or r3 == r2:
                    r3 = np.random.randint(pop_size)
                    
                # Mutation: v = x_r1 + F * (x_r2 - x_r3)
                v = pop[r1] + F * (pop[r2] - pop[r3])
                
                # Crossover
                j_rand = np.random.randint(dim)
                trial = np.zeros(dim)
                for j in range(dim):
                    if np.random.rand() < CR or j == j_rand:
                        trial[j] = v[j]
                    else:
                        trial[j] = pop[i][j]
                
                # Clip to bounds
                trial = np.clip(trial, -5, 5)
                
                # Evaluate
                f_trial = eval_point(trial)
                
                # Selection
                if f_trial < pop_f[i]:
                    new_pop[i] = trial
                    new_pop_f[i] = f_trial
                else:
                    new_pop[i] = pop[i]
                    new_pop_f[i] = pop_f[i]
            
            # Check if we broke out early
            if calls_used >= budget or time.time() > deadline:
                break

            # Update population
            pop = new_pop
            pop_f = new_pop_f

            # Convergence check?
            # If best_f hasn't improved in last K generations, maybe restart?
            # With small budget, restarts might be risky.
            # Let's just run until budget is exhausted.
            
            # Optional: Adaptive F and CR?
            # Keep it simple for now.
            
            # Check if all individuals are identical or similar
            if np.allclose(pop, pop[0]):
                break

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
