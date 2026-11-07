"""Combined optimizer for BBOB f12 (Bent cigar): (1+1)-ES with restarts."""
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 285.0  # safety margin before 330s limit

        rng = np.random.default_rng(42)
        call_count = 0
        best_x = None
        best_f = np.inf

        def evaluate(x):
            nonlocal call_count, best_x, best_f
            if call_count >= budget:
                return np.inf
            x = np.clip(x, -5.0, 5.0)
            f_val = func(x)
            call_count += 1
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # Initialize
        start_x = np.zeros(dim)
        f_start = evaluate(start_x)
        best_x = start_x.copy()
        best_f = f_start

        # Parameters for (1+1)-ES
        sigma = 1.0
        sigma_min = 1e-8
        sigma_max = 10.0
        increase_factor = 1.1
        decrease_factor = 1.0 / increase_factor
        no_improve_limit = 20  # Restart if no improvement for 20 steps
        restarts = 0
        max_restarts = 5

        while call_count < budget and time.time() < deadline:
            x = best_x.copy()
            f_x = best_f
            no_improve = 0
            
            # Inner loop: (1+1)-ES
            while call_count < budget and time.time() < deadline and no_improve < no_improve_limit:
                # Generate candidate
                step = rng.standard_normal(dim)
                y = x + sigma * step
                y = np.clip(y, -5.0, 5.0)
                
                f_y = evaluate(y)
                
                if f_y < f_x:
                    x = y.copy()
                    f_x = f_y
                    no_improve = 0
                    # Increase step size
                    sigma = min(sigma * increase_factor, sigma_max)
                else:
                    no_improve += 1
                    # Decrease step size
                    sigma = max(sigma * decrease_factor, sigma_min)
            
            # If we exited inner loop due to no_improve or budget, check for restart
            if call_count >= budget or time.time() >= deadline:
                break
            
            # If sigma is too small, we are likely stuck. Restart from best with larger sigma.
            if sigma <= sigma_min * 1.1:
                restarts += 1
                if restarts > max_restarts:
                    break
                # Reset sigma to a moderate value
                sigma = 2.0
            else:
                # If we broke out due to no_improve but sigma is still large, 
                # it might be a local minimum. We can try a random jump.
                if no_improve >= no_improve_limit:
                    # Random restart from best
                    jump = rng.standard_normal(dim) * 0.5
                    new_start = best_x + jump
                    new_start = np.clip(new_start, -5.0, 5.0)
                    f_new = evaluate(new_start)
                    if f_new < best_f:
                        best_x = new_start.copy()
                        best_f = f_new
                    sigma = 1.0
                    restarts += 1
                    if restarts > max_restarts:
                        break

        return best_x, best_f
