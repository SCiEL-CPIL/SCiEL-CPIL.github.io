import numpy as np
import time

class Optimizer:
    """
    Differential Evolution optimizer for the BBOB-style Rastrigin function.
    Uses DE/rand/1/bin strategy with adaptive scale factor.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        
        # Parameters
        N = 10  # Population size
        # Ensure N is at least 4 for DE
        if N < 4:
            N = 4
            
        # Time guard
        start_time = time.time()
        deadline = start_time + 300.0  # 300s margin within 330s limit
        
        # Initialization
        # Random initialization in [-5, 5]^dim
        pop = rng.uniform(-5, 5, size=(N, self.dim))
        
        # Evaluate initial population
        # We must be careful with the budget. 
        # Total calls = N (init) + N * generations
        # Budget is 500.
        # If N=10, we have 490 calls left for evolution.
        # Generations = 490 // 10 = 49.
        
        # To maximize early progress, we evaluate the initial population first.
        # However, the scoring tracks b_i for each call i.
        # So we must record the best value after EVERY call.
        
        # We will manage the calls manually to track the best value at each step.
        # But the interface requires calling func(x).
        # The evaluator records every call to func.
        # So we just need to call func in a loop, keeping track of the best found so far.
        
        best_x = None
        best_f = np.inf
        
        # We need to store the population and their fitnesses
        # But we can only get fitness by calling func.
        
        # Strategy:
        # 1. Initialize population. Call func for each. Update best.
        # 2. Loop through generations.
        #    For each individual in pop:
        #       Create trial. Call func. Update best.
        #       If trial better than target, replace.
        
        # We need to store the current fitness of each population member to compare.
        # pop_fit[i] is the fitness of pop[i].
        
        pop_fit = np.empty(N)
        
        # Initial Evaluation
        for i in range(N):
            if time.time() > deadline:
                break
            x = pop[i]
            fx = func(x)
            pop_fit[i] = fx
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
                
        # If we ran out of budget during init (unlikely with N=10, budget=500)
        if self.budget <= N:
            return best_x, best_f
            
        # Adaptive parameters
        F = 0.8
        CR = 0.5
        stagnation_counter = 0
        last_best_f = best_f
        
        # Calculate max generations
        # Remaining budget
        calls_made = N
        remaining = self.budget - calls_made
        
        # We will do generations until budget is exhausted
        # Each generation costs N calls
        # We can do floor(remaining / N) full generations
        # Plus maybe some partial? No, DE needs full population updates usually.
        # But we can just loop while calls_made < budget.
        
        gen = 0
        while calls_made < self.budget:
            if time.time() > deadline:
                break
                
            gen += 1
            improved_this_gen = False
            
            # Decay F slightly to help convergence later?
            # Or keep F high for Rastrigin. Let's try constant F=0.8, CR=0.7
            # Actually, let's use a simple adaptation: if no improvement, increase F.
            
            for i in range(N):
                if calls_made >= self.budget:
                    break
                if time.time() > deadline:
                    break
                    
                # Select r1, r2, r3 distinct from i
                # Simple method: shuffle indices
                indices = list(range(N))
                indices.remove(i)
                rng.shuffle(indices)
                r1, r2, r3 = indices[0], indices[1], indices[2]
                
                # Mutation
                v = pop[r1] + F * (pop[r2] - pop[r3])
                
                # Crossover
                u = pop[i].copy()
                j_rand = rng.integers(0, self.dim)
                for j in range(self.dim):
                    if rng.random() < CR or j == j_rand:
                        u[j] = v[j]
                
                # Clip to bounds
                u = np.clip(u, -5, 5)
                
                # Evaluate
                fu = func(u)
                calls_made += 1
                
                if fu < best_f:
                    best_f = fu
                    best_x = u.copy()
                    improved_this_gen = True
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1
                
                # Selection
                if fu <= pop_fit[i]:
                    pop[i] = u
                    pop_fit[i] = fu
            
            # Adapt parameters based on stagnation
            if not improved_this_gen:
                # Increase F to escape local minima
                F = min(1.0, F + 0.05)
                # Randomize CR slightly?
                if stagnation_counter > 5:
                    CR = rng.uniform(0.3, 0.9)
            else:
                # Decrease F slightly to refine
                F = max(0.5, F - 0.02)
                
        return best_x, best_f
