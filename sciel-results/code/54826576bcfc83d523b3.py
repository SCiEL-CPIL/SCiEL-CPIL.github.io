"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Pure Random Coordinate Descent (RCD) optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 330s kill

        n = self.dim
        budget = self.budget
        calls_made = 0
        best_x = np.zeros(n)
        best_f = float('inf')
        
        # Pre-generate random indices to save time
        rng = np.random.default_rng(12345)
        
        # 1. Initial sampling to find a decent starting point
        # We use 10% of the budget for initial exploration
        init_budget = max(10, int(budget * 0.1))
        
        # Start at origin
        x = np.zeros(n)
        f_val = func(x)
        calls_made += 1
        best_f = f_val
        best_x = x.copy()
        
        # Random sampling to find a better starting point
        # This helps if the origin is far from the minimum
        for _ in range(init_budget - 1):
            if time.time() > deadline:
                break
            x_rand = rng.uniform(-5, 5, n)
            f_rand = func(x_rand)
            calls_made += 1
            if f_rand < best_f:
                best_f = f_rand
                best_x = x_rand.copy()

        # 2. Random Coordinate Descent
        # We will use the remaining budget for RCD.
        # Strategy: 
        # - Keep a current best point.
        # - In each step, pick a random coordinate.
        # - Try to move in + and - directions.
        # - Step size starts large and shrinks geometrically.
        # - If no improvement is found for a while, we might need to restart or increase diversity, 
        #   but pure RCD as requested focuses on local refinement.
        #   To handle the "rotated" aspect and potential local minima of Rosenbrock (though Rosenbrock has no local minima other than the global one, 
        #   the valley is narrow), RCD can struggle if the step size is too large or too small.
        #   Geometric shrinking helps converge.
        
        # Let's implement a simple RCD with adaptive step size.
        # We will alternate between exploring with larger steps and exploiting with smaller steps.
        
        # A more robust RCD approach:
        # For each call:
        # 1. Pick a random dimension d.
        # 2. Pick a step size from a distribution or shrinking sequence.
        # 3. Try x[d] + step and x[d] - step.
        # 4. If improvement, update best.
        
        # To ensure we don't get stuck in a local flat area (Rosenbrock valley is flat in one direction),
        # we need to ensure steps are small enough eventually.
        
        # Let's use a fixed number of steps per "phase" and shrink the max step size.
        # Or simply use a geometric sequence for step size.
        
        # Total remaining calls: budget - calls_made
        # Let's say we do 2 tries per coordinate change (plus and minus).
        # So effective coordinate updates = (budget - calls_made) // 2
        
        # We will use a simple loop.
        # Step size starts at 1.0 and shrinks by factor 0.95 each time we don't improve? 
        # Or just shrink based on total iterations?
        # Let's shrink based on total calls made so far in the RCD phase.
        
        rcd_calls = 0
        total_rcd_calls = budget - calls_made
        
        # Initial step size
        step_size = 1.0
        # Shrink factor
        shrink_factor = 0.98
        
        # We need to be careful: if we shrink too fast, we stop moving.
        # If we shrink too slow, we oscillate.
        
        # Alternative: Use a random step size from a shrinking range.
        # step = uniform(0, max_step)
        
        # Let's try:
        # max_step starts at 2.0
        # In each iteration, pick a random dim.
        # Pick a random offset in [-max_step, max_step].
        # Evaluate.
        # If improved, update best and maybe keep max_step or shrink slightly.
        # If not improved, shrink max_step.
        
        max_step = 2.0
        
        while calls_made < budget and time.time() < deadline:
            # Pick random dimension
            d = rng.integers(0, n)
            
            # Pick random offset
            # We want to explore both directions.
            # offset = rng.uniform(-max_step, max_step)
            # This might be inefficient if it's too small.
            # Let's try a deterministic sweep within the random pick? No, that's not RCD.
            # Pure RCD usually implies trying a specific step.
            
            # Let's try two steps: +step and -step.
            # This doubles the cost per dimension visit but ensures directional search.
            # If budget is tight, maybe just one random direction?
            # With 500 calls, 10 used for init, 490 left.
            # 490 / 2 = 245 dimension visits.
            # 245 / 5 dims = ~49 visits per dim.
            # This seems reasonable.
            
            # Generate step
            # Let's use a geometrically shrinking step size for the "exploit" phase.
            # But we also need "explore".
            # Let's mix: 50% chance of small step (exploit), 50% chance of larger step (explore).
            
            # Actually, let's stick to the directive: "geometrically shrinking step size".
            # This implies a single trajectory of steps that gets smaller.
            # This is good for convergence but bad for global search if we start in a bad spot.
            # The initial sampling phase handles global search.
            # So RCD phase can be pure exploitation with shrinking steps.
            
            # Step size sequence: s_k = s_0 * gamma^k
            # We need to map call index to step size.
            # Let's just update step_size every iteration.
            
            offset = rng.uniform(-max_step, max_step)
            
            # Try + offset
            x_trial = best_x.copy()
            x_trial[d] += offset
            x_trial[d] = np.clip(x_trial[d], -5, 5)
            
            f_trial = func(x_trial)
            calls_made += 1
            
            if f_trial < best_f:
                best_f = f_trial
                best_x = x_trial.copy()
                # If we improved, maybe don't shrink as fast?
                # Or keep shrinking to converge.
                # Let's shrink slowly on improvement.
                max_step *= 0.95
            else:
                # No improvement, shrink faster to escape local noise?
                # Or maybe the step was in the wrong direction.
                # Try - offset?
                if calls_made < budget and time.time() < deadline:
                    x_trial2 = best_x.copy()
                    x_trial2[d] -= offset
                    x_trial2[d] = np.clip(x_trial2[d], -5, 5)
                    
                    f_trial2 = func(x_trial2)
                    calls_made += 1
                    
                    if f_trial2 < best_f:
                        best_f = f_trial2
                        best_x = x_trial2.copy()
                        max_step *= 0.95
                    else:
                        # Neither direction improved. Shrink step size.
                        max_step *= 0.90
                        if max_step < 1e-4:
                            # If step size is too small, we might be stuck.
                            # Let's reset to a medium size to explore again?
                            # But pure RCD with shrinking steps might just stop.
                            # Let's allow a "kick" if we stagnate.
                            # If max_step is very small, let's try a larger random jump.
                            # This deviates from "pure" RCD but helps performance.
                            # Let's stick to the directive strictly first.
                            # If it scores badly, we can add restarts.
                            pass
            
            # Ensure max_step doesn't go to 0
            if max_step < 1e-6:
                max_step = 1e-6

        return best_x

# EVOLVE-BLOCK-END
