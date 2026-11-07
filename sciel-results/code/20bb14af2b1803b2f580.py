"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-Start Adaptive Coordinate Descent with Stagnation-Triggered Random Jumps."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Time guard: return best if we exceed 320s (kill is at 330s)
        start_time = time.time()
        deadline = start_time + 320.0

        best_x = None
        best_fx = np.inf
        calls = 0

        def try_eval(x):
            nonlocal best_x, best_fx, calls
            if calls >= budget:
                return
            fx = func(x)
            calls += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

        # Initial exploration: Random starts to find promising basins
        # Use 10% of budget for initial random sampling
        init_budget = max(10, int(0.1 * budget))
        # Start with center and some random points
        start_points = [np.zeros(dim)]
        for _ in range(init_budget - 1):
            p = rng.uniform(-5, 5, dim)
            start_points.append(p)
        
        # Evaluate initial points
        for p in start_points:
            try_eval(p)
        
        # Coordinate Descent with Adaptive Step Size and Jumps
        # We will iterate through a set of "active" points, refining them with CD.
        # If a point stagnates (no improvement in last N steps), we jump it.
        
        # Current active point to refine
        # We'll cycle through the top few initial points found
        # To keep it simple and effective, we maintain a list of candidates
        # and perform CD on the current best, then if it stagnates, jump.
        # Alternatively, we can have a small population of CD agents.
        
        # Let's use a single-agent approach with aggressive restarting/jumping
        # or a small population of CD agents.
        # Given the "axis-aligned" hint, CD is very efficient on Sphere/Rastrigin.
        # For Rosenbrock/Gallagher, we need to escape local minima.
        
        # Strategy:
        # 1. Start with best point found in init.
        # 2. Perform CD.
        # 3. If no improvement in `stagnation_limit` iterations, perform a random jump.
        # 4. Repeat until budget or time exhausted.
        
        # To leverage parallelism of evaluation (though sequential here), we can
        # just run CD on the global best. But if we get stuck, we jump.
        # A better strategy: Keep a small pool of "survivors".
        # Let's stick to a robust single-tracker with jumps for simplicity and speed,
        # as CD is very fast per step (dim evaluations per full cycle).
        
        current_x = best_x.copy() if best_x is not None else np.zeros(dim)
        current_fx = best_fx
        
        # Adaptive step size for CD
        step_size = 1.0
        stagnation_count = 0
        stagnation_limit = 10  # Number of CD iterations without improvement before jumping
        min_step_size = 1e-4
        max_step_size = 5.0
        
        # Directions to try in CD cycle
        # Standard basis vectors
        directions = [np.zeros(dim) for _ in range(dim)]
        for i in range(dim):
            directions[i][i] = 1.0

        # We will iterate. Each "iteration" of the outer loop consists of
        # trying to improve current_x along each coordinate.
        
        while calls < budget:
            if time.time() > deadline:
                break
                
            improved_in_cycle = False
            old_fx = current_fx
            
            for d in directions:
                if calls >= budget:
                    break
                if time.time() > deadline:
                    break
                    
                # Try positive and negative steps
                # Adaptive step: start with current step_size
                # If improvement, keep step or increase slightly.
                # If no improvement, decrease step.
                
                # Binary search for best step in [-step_size, step_size]?
                # Or just try a few discrete steps to save calls?
                # CD usually tries 1 step per coordinate per cycle.
                # Let's try a few steps to be more robust: +step, -step, +step/2, -step/2?
                # That's 4 calls per coordinate -> 20 calls per cycle for dim=5.
                # Budget 300 -> 15 cycles. Might be too few.
                
                # Better: 1 call per coordinate per cycle, update step adaptively.
                # Try +step
                y = current_x + step_size * d
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls += 1
                
                if fy < current_fx:
                    current_x = y
                    current_fx = fy
                    improved_in_cycle = True
                    # Increase step size slightly if we made a big move?
                    # Or just keep it.
                    # Let's keep it for stability.
                    if step_size < max_step_size:
                        step_size *= 1.1
                else:
                    # Try -step
                    y = current_x - step_size * d
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    calls += 1
                    
                    if fy < current_fx:
                        current_x = y
                        current_fx = fy
                        improved_in_cycle = True
                        if step_size < max_step_size:
                            step_size *= 1.1
                    else:
                        # No improvement in either direction
                        # Decrease step size
                        step_size *= 0.5
                        if step_size < min_step_size:
                            step_size = min_step_size

            # Update global best
            if current_fx < best_fx:
                best_fx = current_fx
                best_x = current_x.copy()
            
            # Check for stagnation
            if not improved_in_cycle:
                stagnation_count += 1
            else:
                stagnation_count = 0
                
            # If stagnated, perform a random jump
            if stagnation_count >= stagnation_limit:
                # Jump to a random location, but biased towards best_x?
                # Or fully random?
                # Fully random helps escape deep local minima in Gallagher.
                # But might waste calls.
                # Let's jump to a random point in the box, or around best_x with larger radius.
                
                # Option 1: Random restart
                # Option 2: Jump from current position with larger step
                
                # Let's try a mix: 50% random restart, 50% large local jump
                if rng.random() < 0.5:
                    current_x = rng.uniform(-5, 5, dim)
                else:
                    # Large jump from current
                    jump_vec = rng.normal(0, 2.0, dim)
                    current_x = np.clip(current_x + jump_vec, -5, 5)
                
                current_fx = func(current_x)
                calls += 1
                
                if current_fx < best_fx:
                    best_fx = current_fx
                    best_x = current_x.copy()
                    
                # Reset stagnation and step size
                stagnation_count = 0
                step_size = 1.0
                
        return best_x, best_fx
# EVOLVE-BLOCK-END
