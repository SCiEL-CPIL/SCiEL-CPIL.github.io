import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        d = self.dim
        budget = self.budget
        
        def time_ok():
            return time.time() < deadline

        # Initialize population
        # For Discus, we want to cover the space well initially.
        # Uniform in [-4.5, 4.5] is a good start since minimizer is in [-4, 4].
        pop_size = 20
        population = np.random.uniform(-4.5, 4.5, (pop_size, d))
        
        # Evaluate initial population
        fitness = np.empty(pop_size)
        calls = 0
        for i in range(pop_size):
            if calls >= budget or not time_ok():
                break
            fitness[i] = func(population[i])
            calls += 1
        
        if calls == 0:
             return np.zeros(d), float('inf')

        best_idx = np.argmin(fitness)
        best_x = population[best_idx].copy()
        best_f = fitness[best_idx]

        # Parameters for the random walk
        # Initial step size: large enough to explore, but not so large we miss the basin
        # The box is [-5, 5], width 10. Minimizer in [-4, 4].
        # A step of ~1.0 covers a significant chunk of the space.
        step_size = 1.0
        # Decay rate per generation (not per call)
        decay = 0.95
        # Minimum step size
        min_step = 1e-4
        
        # We will run generations.
        # Each generation:
        # 1. Sort population by fitness
        # 2. Keep top half (exploit)
        # 3. Replace bottom half with perturbations of the best few (explore/refine)
        
        while calls < budget and time_ok():
            # Sort population
            order = np.argsort(fitness)
            population = population[order]
            fitness = fitness[order]
            
            # Update global best
            if fitness[0] < best_f:
                best_f = fitness[0]
                best_x = population[0].copy()

            # Check if we are stuck (step size too small or no improvement)
            # If step_size is very small, we might be converged or stuck in local min.
            # For Discus, there are no local minima other than the global one? 
            # Actually, Discus is unimodal but flat in some directions. 
            # If step_size decays too much, we might stop moving in flat directions.
            # We can restart if no improvement for a while? 
            # Let's rely on the decay. If step_size < min_step, we can try a small random jump?
            # Or just stop if we are very precise.
            
            half = pop_size // 2
            
            # The top 'half' remain.
            # The bottom 'half' are replaced.
            # We generate new points by perturbing the best few points.
            
            # Which points to perturb? 
            # Let's perturb the top 'quarter' or so, to maintain some diversity.
            # Or just perturb the best point? 
            # Perturbing multiple good points is better for avoiding local optima (if any) and covering the basin.
            
            num_replacements = pop_size - half
            new_points = np.empty((num_replacements, d))
            new_fitness = np.empty(num_replacements)
            
            for i in range(num_replacements):
                if calls >= budget or not time_ok():
                    break
                    
                # Choose a parent from the top half (or top 1/4)
                # Let's choose from the top 1/4 to keep exploitation strong but some diversity
                top_k = max(1, half // 2)
                parent_idx = np.random.randint(0, top_k)
                parent = population[parent_idx]
                
                # Perturb
                # Normal distribution
                noise = np.random.randn(d) * step_size
                new_point = parent + noise
                
                # Clip to box
                new_point = np.clip(new_point, -5, 5)
                
                new_points[i] = new_point
                new_fitness[i] = func(new_point)
                calls += 1
                
            if i < num_replacements - 1:
                # Budget or time ran out during replacement
                # Update population with whatever we have
                # The rest of the bottom half remains as is? Or we just stop?
                # If we stop, we return best_x.
                # Let's update the population with the new points we did evaluate.
                # The un-evaluated slots in the bottom half keep their old (worse) values.
                # This is fine.
                
                # Replace the first 'i+1' slots of the bottom half
                # The bottom half starts at index 'half'
                population[half:half+i+1] = new_points[:i+1]
                fitness[half:half+i+1] = new_fitness[:i+1]
                
                # Break the loop as we are out of budget/time
                break
            
            # Update population
            population[half:] = new_points
            fitness[half:] = new_fitness
            
            # Decay step size
            step_size *= decay
            
            # If step size is too small, we might be stuck.
            # For Discus, if we are close, small steps are good.
            # If we are far, we need large steps.
            # The decay handles this.
            
            # Optional: If we haven't improved in many generations, increase step size?
            # For now, simple decay is robust.

        return best_x, best_f
