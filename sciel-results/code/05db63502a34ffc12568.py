"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Multi-start Differential Evolution (DE/rand/1) Optimizer for Katsuura (f23).
    
    Strategy:
    1. Initialize a diverse population of 50 candidates using vertices, center, and random points.
    2. Run DE/rand/1 with adaptive F and periodic re-injection of random points to maintain diversity.
    3. Use remaining budget for a final (1+1)-ES polish on the best found solution.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin for 300s limit
        dim = self.dim
        budget = self.budget
        pop_size = 50
        
        # Ensure pop_size doesn't exceed budget
        if pop_size >= budget:
            pop_size = max(10, budget // 2)

        rng = np.random.default_rng(42)

        call_count = 0

        def counted_func(x):
            nonlocal call_count
            if call_count >= budget:
                return np.inf
            call_count += 1
            x_clipped = np.clip(x, -5.0, 5.0)
            return func(x_clipped)

        def remaining_time():
            return deadline - time.time()

        # --- Phase 1: Diverse Initialization ---
        # Generate a diverse set of initial points
        init_points = []
        
        # 1. Center
        init_points.append(np.zeros(dim))
        
        # 2. Vertices of the box mapped to [-4, 4]
        for i in range(2 ** dim):
            v = np.array([(i >> j) & 1 for j in range(dim)], dtype=float)
            v = v * 2.0 - 1.0 # Map 0,1 to -1,1
            v = v * 4.0       # Scale to [-4, 4]
            init_points.append(v)
            
        # 3. Random points to fill up to pop_size
        while len(init_points) < pop_size:
            init_points.append(rng.uniform(-4.5, 4.5, dim))
        
        # If we generated more than pop_size (unlikely given dim=5 -> 33 vertices + 1 center = 34), trim
        init_points = init_points[:pop_size]
        
        population = []
        fitness = []
        
        for x0 in init_points:
            if call_count >= budget or remaining_time() < 5.0:
                break
            f0 = counted_func(x0)
            population.append(x0.copy())
            fitness.append(f0)
            
        if not population:
            # Fallback
            x0 = np.zeros(dim)
            f0 = counted_func(x0)
            population.append(x0.copy())
            fitness.append(f0)
            pop_size = 1

        # Sort initial population to identify best
        best_idx = np.argmin(fitness)
        best_x = population[best_idx].copy()
        best_f = fitness[best_idx]

        # --- Phase 2: Differential Evolution (DE/rand/1) ---
        F = 0.8 # Mutation factor
        CR = 0.9 # Crossover rate
        reinject_interval = 3 # Re-inject every 3 generations
        reinject_count = 5 # Replace worst 5
        
        gen = 0
        last_best_f = best_f
        stagnation_counter = 0
        
        while call_count < budget and remaining_time() > 10.0:
            # Check for stagnation
            if best_f < last_best_f - 1e-12:
                last_best_f = best_f
                stagnation_counter = 0
            else:
                stagnation_counter += 1
            
            # Adaptive F: decrease if stagnating? Or keep constant? 
            # Let's keep F constant but re-inject to escape local minima.
            
            for i in range(pop_size):
                if call_count >= budget or remaining_time() < 5.0:
                    break
                
                # Select 3 distinct random individuals
                rand_indices = rng.choice(pop_size, size=3, replace=False)
                a, b, c = rand_indices
                
                # Mutation: v = x_a + F * (x_b - x_c)
                v = population[a] + F * (population[b] - population[c])
                
                # Clip to box
                v = np.clip(v, -5.0, 5.0)
                
                # Evaluate mutant
                fv = counted_func(v)
                
                # Selection: Replace if better
                if fv < fitness[i]:
                    population[i] = v
                    fitness[i] = fv
                    if fv < best_f:
                        best_f = fv
                        best_x = v.copy()
            
            gen += 1
            
            # Re-injection: Replace worst individuals with new random points
            if gen % reinject_interval == 0:
                # Get indices of worst individuals
                worst_indices = np.argsort(fitness)[-reinject_count:]
                for idx in worst_indices:
                    if call_count >= budget or remaining_time() < 5.0:
                        break
                    new_x = rng.uniform(-4.5, 4.5, dim)
                    new_f = counted_func(new_x)
                    population[idx] = new_x
                    fitness[idx] = new_f
                    if new_f < best_f:
                        best_f = new_f
                        best_x = new_x.copy()

        # --- Phase 3: Polishing with (1+1)-ES ---
        # Use remaining budget to polish the best solution
        if call_count < budget and remaining_time() > 2.0 and best_x is not None:
            x = best_x.copy()
            fx = best_f
            sigma = 0.05
            min_sigma = 1e-10
            max_sigma = 1.0
            
            while call_count < budget and remaining_time() > 1.0:
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5.0, 5.0)
                fy = counted_func(y)
                
                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma = min(sigma * 1.05, max_sigma)
                else:
                    sigma = max(sigma * 0.95, min_sigma)
                    if sigma <= min_sigma * 1.01:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
