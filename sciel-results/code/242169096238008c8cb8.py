import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_x = None
        self.best_f = np.inf
        self.calls_made = 0
        self.start_time = None

    def __call__(self, func):
        self.start_time = time.time()
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        # Time limit safety margin: Evaluator kills at 330s. 
        # We stop at 320s to be safe.
        time_limit = 320.0
        
        # State
        self.best_x = np.zeros(n)
        self.best_f = np.inf
        self.calls_made = 0
        
        def call_func(x):
            if self.calls_made >= budget:
                return np.inf
            if time.time() - self.start_time > time_limit:
                return np.inf
            # Ensure input is valid
            x = np.asarray(x, dtype=np.float64)
            if x.shape != (n,):
                x = x.reshape(n)
            x = np.clip(x, -5.0, 5.0)
            try:
                fx = func(x)
            except Exception:
                fx = np.inf
            
            self.calls_made += 1
            if fx < self.best_f:
                self.best_f = fx
                self.best_x = x.copy()
            return fx

        # --- Strategy: Island Model Evolution Strategy ---
        # We maintain multiple "islands" (sub-populations).
        # Each island evolves independently.
        # Periodically, the best individuals migrate between islands to maintain diversity and share progress.
        
        num_islands = 4
        pop_size = 10  # Size of each island population
        
        # Initialize populations
        # Use a mix of random and LHS-like spread for initial diversity
        populations = []
        for _ in range(num_islands):
            # Random points in [-5, 5]^n
            pop = rng.uniform(-5.0, 5.0, size=(pop_size, n))
            populations.append(pop)
            
        # Evaluate initial populations
        pop_fitness = []
        for i in range(num_islands):
            fits = []
            for x in populations[i]:
                fits.append(call_func(x))
            pop_fitness.append(np.array(fits))
            
            if self.calls_made >= budget or time.time() - self.start_time > time_limit:
                return self.best_x, self.best_f

        # Parameters
        sigma_base = 1.5
        migration_interval = 20  # Migrate every 20 generations
        current_gen = 0
        
        # Main Loop
        while self.calls_made < budget and time.time() - self.start_time < time_limit:
            current_gen += 1
            
            # --- Evolution Step for each Island ---
            for i in range(num_islands):
                if self.calls_made >= budget or time.time() - self.start_time > time_limit:
                    break
                
                pop = populations[i]
                fits = pop_fitness[i]
                
                # Select parents: Top 50%
                # Simple selection: keep the best half
                k = max(2, pop_size // 2)
                idx = np.argsort(fits)[:k]
                parents = pop[idx]
                parent_fits = fits[idx]
                
                # Adapt sigma based on island performance
                # If best fit is improving, we can be bolder; if stagnating, be cautious.
                # Simple heuristic: sigma proportional to spread of best 50%
                best_fit = np.min(fits)
                avg_best_fit = np.mean(parent_fits)
                # Scale sigma slightly based on how far we are from 0 (since f_opt is in [-100,100])
                # But primarily keep it stable.
                sigma = sigma_base
                
                # Generate offspring
                new_pop = []
                new_fits = []
                
                # Keep the best individual from current population (elitism)
                best_idx = np.argmin(fits)
                new_pop.append(pop[best_idx].copy())
                new_fits.append(fits[best_idx])
                
                # Fill the rest with offspring
                for _ in range(pop_size - 1):
                    if self.calls_made >= budget or time.time() - self.start_time > time_limit:
                        break
                    
                    # Pick a random parent
                    p_idx = rng.integers(len(parents))
                    parent = parents[p_idx]
                    
                    # Mutation: Gaussian noise
                    # Use a slightly larger step for exploration
                    mutation = rng.standard_normal(n) * sigma
                    child = parent + mutation
                    child = np.clip(child, -5.0, 5.0)
                    
                    fc = call_func(child)
                    new_pop.append(child)
                    new_fits.append(fc)
                
                # Update island population
                if len(new_pop) == pop_size:
                    populations[i] = np.array(new_pop)
                    pop_fitness[i] = np.array(new_fits)
                else:
                    # If we ran out of budget mid-generation, pad with best known
                    # This case should be rare due to checks above
                    pass

            # --- Migration Step ---
            if current_gen % migration_interval == 0:
                # Find the best individual in each island
                best_indices = []
                for i in range(num_islands):
                    best_indices.append(np.argmin(pop_fitness[i]))
                
                # Identify the best island
                island_best_fits = [pop_fitness[i][best_indices[i]] for i in range(num_islands)]
                best_island = np.argmin(island_best_fits)
                
                # Migrate: The best individual from the best island replaces the worst individual in each other island
                # This helps escape local minima in other islands while preserving the success of the best island
                best_x = populations[best_island][best_indices[best_island]].copy()
                best_f = pop_fitness[best_island][best_indices[best_island]]
                
                for i in range(num_islands):
                    if i == best_island:
                        continue
                    # Find worst individual in island i
                    worst_idx = np.argmax(pop_fitness[i])
                    # Replace if the migrant is better
                    if best_f < pop_fitness[i][worst_idx]:
                        populations[i][worst_idx] = best_x
                        pop_fitness[i][worst_idx] = best_f

        return self.best_x, self.best_f
