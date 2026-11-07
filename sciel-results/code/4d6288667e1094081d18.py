"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Differential Evolution (DE) for BBOB f4 (Büche-Rastrigin)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s kill
        
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        # DE Parameters
        # Population size typically 5-10 * dim. For dim=5, 25-50 is good.
        # Let's use 30 to balance exploration and exploitation.
        pop_size = 30
        F = 0.8  # Scale factor
        CR = 0.9 # Crossover rate
        
        # Initialize population uniformly in [-5, 5]
        pop = rng.uniform(-5, 5, size=(pop_size, dim))
        
        # Evaluate initial population
        # We need to be careful with budget. 
        # Initial evals: pop_size
        if pop_size > budget:
            pop_size = budget
            pop = rng.uniform(-5, 5, size=(pop_size, dim))
            
        fvals = np.array([func(x) for x in pop])
        calls_made = pop_size
        
        best_idx = np.argmin(fvals)
        best_x = pop[best_idx].copy()
        best_f = fvals[best_idx]
        
        # DE loop
        # Each generation costs pop_size function evaluations (for offspring)
        # We stop when calls_made + pop_size > budget or time is up.
        
        # Optimization: 
        # We can evaluate offspring one by one and update the population immediately?
        # Standard DE is usually generation-based.
        # However, with a tight budget (500) and pop_size=30, we can do ~16 generations.
        # This might be too few for high precision.
        
        # Alternative: Adaptive Pop Size or Smaller Pop Size?
        # Büche-Rastrigin is very multimodal. Larger pop helps global search.
        # Let's try to make it as efficient as possible.
        
        # If we use a smaller pop size, we get more generations.
        # Let's try pop_size = 20. 500 / 20 = 25 generations.
        # Let's try pop_size = 10. 500 / 10 = 50 generations.
        # For dim 5, pop_size 10-20 is often sufficient for DE if F and CR are tuned.
        
        # Let's stick with a moderate pop size and standard DE.
        # Actually, let's implement a version that allows for "greedy" selection 
        # within the generation to potentially converge faster if the landscape is smooth locally.
        
        # Reset for a potentially better configuration
        # Let's use pop_size = 20
        pop_size = 20
        if pop_size > budget:
             pop_size = max(5, budget // 2) # Ensure at least some pop
        
        # Re-initialize with new pop size
        pop = rng.uniform(-5, 5, size=(pop_size, dim))
        fvals = np.array([func(x) for x in pop])
        calls_made = pop_size
        
        best_idx = np.argmin(fvals)
        best_x = pop[best_idx].copy()
        best_f = fvals[best_idx]
        
        while calls_made + pop_size <= budget:
            if time.time() > deadline:
                break
                
            new_pop = np.zeros_like(pop)
            new_fvals = np.zeros_like(fvals)
            
            # For each individual in population, generate an offspring
            for i in range(pop_size):
                if calls_made >= budget:
                    break
                
                # Select 3 distinct random individuals
                # Exclude i to avoid trivial self-crossover, though not strictly required
                candidates = [j for j in range(pop_size) if j != i]
                r1, r2, r3 = rng.choice(candidates, size=3, replace=False)
                
                # Mutation: v = x[r1] + F * (x[r2] - x[r3])
                v = pop[r1] + F * (pop[r2] - pop[r3])
                
                # Clip to bounds
                v = np.clip(v, -5, 5)
                
                # Crossover
                u = np.zeros_like(v)
                j_rand = rng.integers(0, dim)
                
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        u[j] = v[j]
                    else:
                        u[j] = pop[i][j]
                        
                # Evaluate offspring
                f_u = func(u)
                calls_made += 1
                
                # Selection: Greedy
                if f_u <= fvals[i]:
                    new_pop[i] = u
                    new_fvals[i] = f_u
                else:
                    new_pop[i] = pop[i]
                    new_fvals[i] = fvals[i]
                    
            # Update global best
            gen_best_idx = np.argmin(new_fvals)
            if new_fvals[gen_best_idx] < best_f:
                best_f = new_fvals[gen_best_idx]
                best_x = new_pop[gen_best_idx].copy()
                
            # Update population
            pop = new_pop
            fvals = new_fvals
            
        return best_x, best_f

# EVOLVE-BLOCK-END
