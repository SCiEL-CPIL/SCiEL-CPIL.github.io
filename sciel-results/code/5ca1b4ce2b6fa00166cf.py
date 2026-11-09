"""Combined optimizer: LHS initialization + (1+1)-ES with stagnation restarts and deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with LHS initialization, stagnation restarts, and deadline guard for Katsuura f23."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Phase 1: Latin Hypercube-style sampling for fast initial basin discovery
        # Use a decent number of initial samples to find a good basin
        n_lhs = min(40, budget // 10)
        best_x = None
        best_f = np.inf
        lhs_points = []
        lhs_fits = []

        for i in range(n_lhs):
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            lhs_points.append(x)
            lhs_fits.append(fx)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: (1+1)-ES with adaptive sigma and stagnation restarts
        x = best_x.copy()
        fx = best_f
        sigma = 1.5
        calls_made = n_lhs
        stagnation_count = 0
        restart_threshold = 20 # Restart if no improvement for 20 steps
        lhs_index = 0 # Pointer to cycle through LHS points for restarts if needed, or just use random

        while calls_made < budget:
            # Deadline check with safety margin
            # The evaluator kills at 330s. We aim to finish well before.
            if time.time() - start_time > 320.0:
                break

            # Generate candidate
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls_made += 1

            if fy < fx:
                # Accept
                x = y
                fx = fy
                best_x = x.copy()
                best_f = fx
                stagnation_count = 0
                # Increase sigma on success
                sigma *= 1.2
            else:
                # Reject
                stagnation_count += 1
                # Decrease sigma on failure
                sigma *= 0.85

            # Stagnation Restart Logic
            # If we are stuck, try a new starting point.
            # We can restart from a random point or a previously sampled LHS point.
            # Using a random point is simpler and explores new areas.
            # Using LHS points might re-explore known good basins.
            # Let's try a mix: if we restart, pick a random point, but if we have many restarts, 
            # maybe we should just keep going? No, Katsuura has many local minima.
            
            if stagnation_count >= restart_threshold:
                # Restart from a new random location
                # To ensure we don't get stuck in the same local minimum, 
                # we pick a completely new random point.
                new_x = rng.uniform(-5, 5, dim)
                new_fx = func(new_x)
                calls_made += 1
                
                # If the new point is better, we move there. 
                # Even if it's worse, we might want to start optimizing from there 
                # if the current local minimum is very deep but not global? 
                # Actually, (1+1)-ES is greedy. If we restart, we should probably 
                # only restart if we think we are stuck in a bad basin.
                # A safer strategy: Only restart if the new point is significantly better 
                # OR if we have been stuck for a very long time.
                # Given the budget, let's just restart the optimizer state to the new point 
                # if it's not too bad, or just force a jump.
                
                # Heuristic: If we are stuck, the current basin is likely local. 
                # Let's jump to a new random point.
                x = new_x
                fx = new_fx
                sigma = 1.5 # Reset sigma
                stagnation_count = 0
                
                # Update global best if this new point is better
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
