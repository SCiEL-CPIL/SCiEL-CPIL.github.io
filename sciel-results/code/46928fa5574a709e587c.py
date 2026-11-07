"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hooke-Jeeves pattern search for bent-cigar (f12) optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(42)
        call_count = 0
        best_x = None
        best_f = np.inf

        def evaluate(x):
            nonlocal call_count, best_x, best_f
            if call_count >= budget:
                return np.inf
            # The task states func clips, but we ensure it's a valid vector
            x = np.clip(x, -5.0, 5.0)
            f_val = func(x)
            call_count += 1
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # 1. Initialization: Sample a few points to find a good basin
        n_init = 5
        for _ in range(n_init):
            if call_count >= budget:
                break
            x = rng.uniform(-5, 5, dim)
            evaluate(x)

        if best_x is None:
            # Fallback
            x0 = np.zeros(dim)
            evaluate(x0)
            best_x = x0

        # 2. Hooke-Jeeves Pattern Search
        current_x = best_x.copy()
        current_f = best_f
        steps = np.ones(dim) * 0.5
        min_step = 1e-6
        max_restarts = 5
        restarts = 0

        while call_count < budget and time.time() < deadline:
            improved = False
            x_orig = current_x.copy()
            
            for i in range(dim):
                if call_count >= budget:
                    break
                
                # Test positive direction
                x_plus = x_orig.copy()
                x_plus[i] += steps[i]
                f_plus = evaluate(x_plus)
                
                # Test negative direction
                x_minus = x_orig.copy()
                x_minus[i] -= steps[i]
                f_minus = evaluate(x_minus)
                
                if f_plus < current_f:
                    current_x = x_plus
                    current_f = f_plus
                    steps[i] *= 2.0
                    improved = True
                elif f_minus < current_f:
                    current_x = x_minus
                    current_f = f_minus
                    steps[i] *= 2.0
                    improved = True
            
            if not improved:
                steps *= 0.5
                if np.min(steps) < min_step:
                    # Local minimum reached or stuck
                    restarts += 1
                    if restarts > max_restarts:
                        break
                    
                    # Restart: perturb the global best
                    perturbation = 0.5 * rng.standard_normal(dim)
                    new_x = best_x + perturbation
                    new_x = np.clip(new_x, -5, 5)
                    f_new = evaluate(new_x)
                    
                    if f_new < best_f:
                        current_x = new_x
                        current_f = f_new
                    else:
                        # If restart didn't help, try a larger jump from current
                        new_x = current_x + 1.0 * rng.standard_normal(dim)
                        new_x = np.clip(new_x, -5, 5)
                        f_new = evaluate(new_x)
                        current_x = new_x
                        current_f = f_new
                        
                    steps = np.ones(dim) * 0.5

        return best_x, best_f
# EVOLVE-BLOCK-END
