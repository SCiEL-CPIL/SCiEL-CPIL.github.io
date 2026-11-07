import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Fixed seed for reproducibility within a single run, though evaluator handles instances
        rng = np.random.default_rng(12345)
        start_time = time.time()
        # Safety margin: 330s limit, allow 280s for computation
        deadline = start_time + 280.0 

        dim = self.dim
        budget = self.budget
        
        # Hyperparameters tuned for BBOB f24 (Lunacek bi-Rastrigin)
        # Lambda: number of offspring per generation
        lambda_pop = 7 
        # Initial step size
        sigma = 1.0
        # Success rule parameters
        mu = lambda_pop // 2  # Number of successful offspring considered for success rate
        
        # State
        x_best = rng.uniform(-5.0, 5.0, dim)
        f_best = func(x_best)
        calls = 1
        f_ever = f_best
        x_ever = x_best.copy()
        
        # Restart logic state
        no_improve_count = 0
        restart_threshold = 30 # Generations without improvement before restart

        # We run for enough generations to consume the budget
        # Each generation costs lambda_pop + 1 (if we count the parent, but here we only evaluate offspring)
        # Actually, standard (1+lambda) ES evaluates lambda offspring. 
        # Total calls = 1 (initial) + generations * lambda_pop
        max_generations = (budget - 1) // lambda_pop
        
        for gen in range(max_generations):
            # Check time
            if time.time() > deadline:
                break
                
            # Generate offspring
            # x_best + sigma * normal
            offspring = x_best + sigma * rng.standard_normal((lambda_pop, dim))
            offspring = np.clip(offspring, -5.0, 5.0)
            
            # Evaluate offspring
            # Optimization: Evaluate in a list comprehension to minimize overhead
            f_offspring = np.array([func(x) for x in offspring])
            calls += lambda_pop
            
            # Find the best offspring
            best_off_idx = np.argmin(f_offspring)
            f_best_off = f_offspring[best_off_idx]
            x_best_off = offspring[best_off_idx]
            
            # Update step size using success rule
            # Count how many offspring improved upon the current parent x_best
            num_success = np.sum(f_offspring < f_best)
            
            # 1/5 success rule logic
            # If success rate > 1/5, increase sigma (exploit less, explore more)
            # If success rate < 1/5, decrease sigma (exploit more)
            # Smooth adaptation:
            if num_success > mu:
                sigma *= 1.1
            elif num_success < mu:
                sigma *= 0.9
            else:
                sigma *= 1.0
            
            # Clamp sigma to reasonable bounds
            sigma = np.clip(sigma, 1e-4, 10.0)

            # Accept/Reject
            if f_best_off < f_best:
                x_best = x_best_off
                f_best = f_best_off
                no_improve_count = 0
                
                if f_best < f_ever:
                    f_ever = f_best
                    x_ever = x_best.copy()
            else:
                no_improve_count += 1
                # Do not move x_best, keep the current best
                
            # Restart logic
            # If we haven't improved in 'restart_threshold' generations, we might be stuck in a local minimum
            # Lunacek has two basins. A restart helps find the other one.
            if no_improve_count >= restart_threshold:
                # Random restart to escape local basin
                # Pick a new random point
                x_restart = rng.uniform(-5.0, 5.0, dim)
                f_restart = func(x_restart)
                calls += 1
                
                if f_restart < f_best:
                    x_best = x_restart
                    f_best = f_restart
                    if f_best < f_ever:
                        f_ever = f_best
                        x_ever = x_best.copy()
                else:
                    # Keep old best if new one is worse, but reset counter and sigma
                    # This forces a new search direction from the known best
                    pass 
                
                no_improve_count = 0
                sigma = 1.5 # Reset sigma to a moderate value for new exploration
                
                # If we ran out of budget for the restart evaluation, break
                if calls >= budget:
                    break

        return x_ever, f_ever
