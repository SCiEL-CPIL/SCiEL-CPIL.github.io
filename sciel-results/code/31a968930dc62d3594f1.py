import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        # Deadline guard
        def time_left():
            return 320.0 - (time.time() - start_time)

        # State
        best_x = None
        best_f = np.inf
        calls_made = 0
        
        # Population size
        pop_size = 10
        population = []
        
        # Initialize population with random points
        # Use a mix of random and slightly structured points
        for _ in range(pop_size):
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            calls_made += 1
            population.append((fx, x))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
                
        # Sort population by fitness
        population.sort(key=lambda item: item[0])
        
        # Adaptive sigma for mutation
        sigma = 1.0
        stagnation_count = 0
        last_best_f = best_f
        
        # Local search function (Nelder-Mead simplified or just hill climbing)
        # Since we have limited calls, a simple coordinate descent or few Nelder-Mead steps is good.
        # Let's use a simple "polish" step: try moving in each dimension + and -
        # This uses 2*dim calls. For dim=5, that's 10 calls.
        # Maybe too many. Let's try 2 calls: random direction step.
        # Or better: use scipy.optimize.minimize with Nelder-Mead, maxiter=5.
        # But importing scipy inside __call__ might be slow if not cached.
        # Let's implement a simple local search manually.
        
        def local_search(x, f_current):
            """Perform a few steps of local optimization. Returns best x and f found."""
            # Simple hill climbing with adaptive step
            # Try 3 random directions
            best_local_x = x.copy()
            best_local_f = f_current
            step_size = 0.5
            for _ in range(3): # 3 attempts
                direction = rng.normal(size=dim)
                direction /= np.linalg.norm(direction)
                # Try step forward
                x_trial = np.clip(best_local_x + step_size * direction, -5, 5)
                f_trial = func(x_trial)
                calls_made += 1 # Note: this is a closure variable, need to handle nonlocal or pass state
                # Wait, calls_made is in outer scope. We need to update it.
                # Python closure doesn't allow assignment to nonlocal int easily without nonlocal keyword.
                # Let's restructure to avoid closure complexity or use a class attribute.
                pass
            return best_local_x, best_local_f, calls_made

        # To handle calls_made correctly, let's inline the logic or use a list for calls.
        calls_counter = [calls_made]
        
        def evaluate(x):
            fx = func(x)
            calls_counter[0] += 1
            return fx

        # Re-evaluate initial population with counter? No, we already did.
        # Let's just use the counter for new evaluations.
        calls_counter[0] = calls_made

        while calls_counter[0] < budget - 10: # Leave some buffer for safety
            if time_left() < 1.0:
                break
                
            # Select parent: Best in population
            parent_f, parent_x = population[0]
            
            # Mutate
            # Adapt sigma based on stagnation
            if stagnation_count > 10:
                sigma = max(0.5, sigma * 0.8) # Reduce step size
            elif stagnation_count == 0:
                sigma = min(5.0, sigma * 1.05) # Increase step size slightly
            
            x_child = parent_x + sigma * rng.normal(size=dim)
            x_child = np.clip(x_child, -5, 5)
            
            # Evaluate child
            f_child = func(x_child)
            calls_counter[0] += 1
            
            # Local Search: Try to improve x_child
            # Use a simple "polish" with 2-3 steps
            best_local_x = x_child.copy()
            best_local_f = f_child
            step_size = 0.2
            
            for _ in range(2): # 2 local search steps
                if calls_counter[0] >= budget - 5:
                    break
                # Try a random direction
                dir = rng.normal(size=dim)
                dir /= np.linalg.norm(dir)
                
                # Try forward
                x_try = np.clip(best_local_x + step_size * dir, -5, 5)
                f_try = func(x_try)
                calls_counter[0] += 1
                
                if f_try < best_local_f:
                    best_local_x = x_try
                    best_local_f = f_try
                    step_size *= 1.1 # Move further in same direction
                else:
                    # Try backward
                    x_try = np.clip(best_local_x - step_size * dir, -5, 5)
                    f_try = func(x_try)
                    calls_counter[0] += 1
                    if f_try < best_local_f:
                        best_local_x = x_try
                        best_local_f = f_try
                        step_size *= 1.1
                    else:
                        step_size *= 0.7 # Reduce step size
                    
            # Update global best
            if best_local_f < best_f:
                best_f = best_local_f
                best_x = best_local_x.copy()
                stagnation_count = 0
            else:
                stagnation_count += 1
                
            # Update population
            # Replace worst if new is better
            worst_f, _ = population[-1]
            if best_local_f < worst_f:
                population[-1] = (best_local_f, best_local_x.copy())
                population.sort(key=lambda item: item[0])
            
            # Check for complete stagnation to trigger restart
            if stagnation_count > 50:
                # Restart from a random point
                x_restart = rng.uniform(-5, 5, dim)
                f_restart = func(x_restart)
                calls_counter[0] += 1
                if f_restart < best_f:
                    best_f = f_restart
                    best_x = x_restart.copy()
                # Insert into population
                population[-1] = (f_restart, x_restart.copy())
                population.sort(key=lambda item: item[0])
                stagnation_count = 0
                sigma = 1.0

        return best_x, best_f
