"""Combined optimizer for BBOB f12 (Bent cigar): Adaptive Hooke-Jeeves with restarts."""
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 285.0  # Safety margin

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

        # --- Phase 1: Multi-start Initialization ---
        # Evaluate origin first
        start_x = np.zeros(dim)
        evaluate(start_x)
        
        # Evaluate a few random points to ensure we are in a good basin
        # Bent Cigar is unimodal, but the hidden shift/rotation might make 
        # the origin far from the minimum.
        for _ in range(10):
            if call_count >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            evaluate(x)

        if best_x is None:
            # Fallback if something went wrong
            best_x = np.zeros(dim)
            best_f = evaluate(best_x)

        # --- Phase 2: Adaptive Hooke-Jeeves ---
        # Initialize step sizes. Start moderately large to find the valley direction.
        # Bent Cigar has a very narrow dimension. Large steps help traverse the wide dimensions.
        steps = np.full(dim, 1.0)
        min_step = 1e-6
        max_step = 10.0
        shrink_factor = 0.5
        grow_factor = 1.5
        max_no_improve = 20
        no_improve_count = 0
        restarts = 0
        max_restarts = 5

        while call_count < budget and time.time() < deadline:
            x = best_x.copy()
            f_x = best_f
            improved_in_cycle = False

            # Hooke-Jeeves Pattern Search
            for i in range(dim):
                if call_count >= budget or time.time() > deadline:
                    break
                
                # Try positive direction
                x_test = x.copy()
                x_test[i] += steps[i]
                x_test = np.clip(x_test, -5.0, 5.0)
                f_test = evaluate(x_test)
                
                if f_test < f_x:
                    x = x_test
                    f_x = f_test
                    improved_in_cycle = True
                    # Double the step size for this dimension if we improved
                    # This helps move quickly along the valley
                    steps[i] = min(steps[i] * grow_factor, max_step)
                else:
                    # Try negative direction
                    x_test = x.copy()
                    x_test[i] -= steps[i]
                    x_test = np.clip(x_test, -5.0, 5.0)
                    f_test = evaluate(x_test)
                    
                    if f_test < f_x:
                        x = x_test
                        f_x = f_test
                        improved_in_cycle = True
                        steps[i] = min(steps[i] * grow_factor, max_step)
                    else:
                        # No improvement in this dimension, shrink step
                        steps[i] = max(steps[i] * shrink_factor, min_step)

            # Update global best if we improved
            if improved_in_cycle:
                if f_x < best_f:
                    best_f = f_x
                    best_x = x.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1

            # Check for stagnation
            if no_improve_count >= max_no_improve:
                # Stagnation detected.
                # Check if steps are too small
                if np.all(steps <= min_step * 1.1):
                    # We are stuck in a local minimum or precision limit.
                    # Restart from best point with larger steps.
                    restarts += 1
                    if restarts > max_restarts:
                        break
                    steps = np.full(dim, 2.0)
                    no_improve_count = 0
                else:
                    # Steps are not tiny, but no improvement. 
                    # Maybe we are oscillating or the function is flat.
                    # Try a random jump to escape.
                    jump = rng.standard_normal(dim) * 0.5
                    new_start = best_x + jump
                    new_start = np.clip(new_start, -5.0, 5.0)
                    f_new = evaluate(new_start)
                    if f_new < best_f:
                        best_x = new_start.copy()
                        best_f = f_new
                    steps = np.full(dim, 1.0)
                    no_improve_count = 0
                    restarts += 1
                    if restarts > max_restarts:
                        break

        return best_x, best_f
