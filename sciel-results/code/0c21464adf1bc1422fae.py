import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.deadline = time.time() + 280

    def __call__(self, func):
        budget = self.budget
        dim = self.dim
        rng = np.random.default_rng(42)
        
        # Initial population
        pop_size = min(50, budget)
        if budget < 10:
            pop_size = budget
            
        # Simple CMA-ES like approach or just a robust random restart hill climber
        # Given the strict time and call limits, a well-tuned random restart with local search is safe.
        
        best_x = np.zeros(dim)
        best_f = np.inf
        calls = 0
        
        def evaluate(x):
            nonlocal best_x, best_f, calls
            x = np.clip(x, -5, 5)
            f = func(x)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f
        
        # Phase 1: Random Sampling to find a good starting point
        # We have 500 calls. Let's use 50 for initial sampling.
        for _ in range(pop_size):
            if calls >= budget:
                break
            x = rng.uniform(-5, 5, dim)
            evaluate(x)
            
        # Phase 2: Local Search from the best point
        # Use a simple coordinate descent or random steps with decreasing step size
        
        current_x = best_x.copy()
        current_f = best_f
        
        step_size = 1.0
        no_improve_count = 0
        max_no_improve = 50
        
        while calls < budget and time.time() < self.deadline:
            improved = False
            
            # Try random steps
            for _ in range(10):
                if calls >= budget or time.time() > self.deadline:
                    break
                
                # Generate a random perturbation
                direction = rng.normal(0, 1, dim)
                direction /= (np.linalg.norm(direction) + 1e-8)
                
                new_x = current_x + step_size * direction
                new_f = evaluate(new_x)
                
                if new_f < current_f:
                    current_x = new_x
                    current_f = new_f
                    improved = True
                    no_improve_count = 0
                    step_size *= 1.1 # Increase step size slightly if we improve
                else:
                    no_improve_count += 1
                    step_size *= 0.9 # Decrease step size if no improvement
                
            if not improved:
                # If we didn't improve in the last batch, restart from a random point or current best
                if no_improve_count > max_no_improve:
                    # Restart
                    if calls < budget:
                        new_start = rng.uniform(-5, 5, dim)
                        new_f = evaluate(new_start)
                        if new_f < best_f:
                            current_x = new_start
                            current_f = new_f
                        else:
                            current_x = best_x.copy()
                            current_f = best_f
                        
                        step_size = 1.0
                        no_improve_count = 0
                    else:
                        break
                else:
                    # Reduce step size further if stuck
                    step_size *= 0.8
                    if step_size < 1e-3:
                        # Try a few more random restarts
                        for _ in range(5):
                            if calls >= budget:
                                break
                            new_start = rng.uniform(-5, 5, dim)
                            new_f = evaluate(new_start)
                            if new_f < best_f:
                                current_x = new_start
                                current_f = new_f
                                step_size = 1.0
                                no_improve_count = 0
                                break
                        
                        if step_size < 1e-3 and no_improve_count > max_no_improve * 2:
                            break
            else:
                # Reset no_improve if we improved
                pass

        return best_x, best_f
