"""Deterministic multi-scale grid search with local hill climb for f21."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Deterministic multi-scale grid search with local hill climb for f21."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin before 330s kill

        global_best_x = None
        global_best_f = np.inf
        calls_made = 0
        total_budget = self.budget
        dim = self.dim

        def time_left():
            return deadline - time.time()

        def evaluate(x):
            nonlocal calls_made, global_best_f, global_best_x
            fx = func(x)
            calls_made += 1
            if fx < global_best_f:
                global_best_f = fx
                global_best_x = x.copy()
            return fx

        # Phase 1: Deterministic Multi-Scale Grid Search
        # We use a coarse grid to ensure we hit a good region early.
        # In 5D, a full grid of 3^5=243 points is affordable and provides good coverage.
        # Points: -4, 0, 4 (since minimizer is in [-4, 4]^5)
        
        # Generate grid points
        # To be safe and deterministic, we evaluate them in a specific order.
        # Order: Start from center (0,0,0,0,0) then expand? Or just lexicographic?
        # Lexicographic is fine, but center-out might be better for "early progress" if center is good.
        # However, the problem states hidden shift. So any point is a gamble.
        # But a grid ensures we cover the space.
        
        # Let's use a 3x3x3x3x3 grid: values [-4, 0, 4]
        grid_values = [-4.0, 0.0, 4.0]
        
        # Generate all combinations
        grid_points = []
        for d1 in grid_values:
            for d2 in grid_values:
                for d3 in grid_values:
                    for d4 in grid_values:
                        for d5 in grid_values:
                            grid_points.append(np.array([d1, d2, d3, d4, d5]))
                            
        # Evaluate grid points until budget or time runs out
        # We have 243 points. Budget is 500.
        # This uses ~243 calls.
        
        for pt in grid_points:
            if calls_made >= total_budget:
                break
            if time_left() < 1.0:
                break
            evaluate(pt)

        # Phase 2: Local Hill Climb from Best Grid Point
        # If we found a good point, refine it.
        # If we didn't find any (shouldn't happen), fallback to random.
        if global_best_x is None:
            # Fallback: evaluate origin
            x = np.zeros(dim)
            evaluate(x)
        
        x = global_best_x.copy()
        fx = global_best_f
        
        # Fixed-step hill climb
        # Start with a moderate step size, decrease if stuck, increase if improving?
        # Or just decrease step size gradually to converge?
        # Since we are on a grid, we might be far from the optimum.
        # A "hill climb" with fixed steps is risky if the function is rugged (f21 is rugged).
        # However, the directive says "fixed-step local hill climb".
        # Let's use a sequence of decreasing step sizes to simulate multi-scale refinement.
        
        # Steps: 1.0, 0.5, 0.25, 0.125, ...
        # For each step size, try 26 directions (axis +/- and center?) or just random?
        # Deterministic directions: 2*dim axes.
        
        # To be more effective on rugged landscapes, we might need to jump.
        # But "fixed-step local hill climb" implies local moves.
        # Let's try a "coordinate descent" style or "axis-aligned" moves.
        
        # We have ~257 calls left.
        # Let's do a simple adaptive step hill climb.
        # Step sizes: 2.0, 1.0, 0.5, 0.25, 0.125, 0.0625
        # For each step size, try all 2*dim axis-aligned neighbors.
        
        step_sizes = [2.0, 1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125]
        
        for step in step_sizes:
            if calls_made >= total_budget:
                break
            if time_left() < 0.5:
                break
                
            # Try each axis, positive and negative
            improved_in_this_step = False
            for _ in range(2): # Passes? Or just one pass?
                if calls_made >= total_budget:
                    break
                if time_left() < 0.5:
                    break
                for i in range(dim):
                    if calls_made >= total_budget:
                        break
                    if time_left() < 0.5:
                        break
                    for sign in [1, -1]:
                        if calls_made >= total_budget:
                            break
                        if time_left() < 0.5:
                            break
                        y = x.copy()
                        y[i] += sign * step
                        y = np.clip(y, -5, 5)
                        fy = evaluate(y)
                        if fy < fx:
                            x = y.copy()
                            fx = fy
                            improved_in_this_step = True
                            # Restart the axis loop? Or continue?
                            # Continuing is simpler.
        
        # If we have budget left, do a final fine-tuning with very small steps
        # Or just stop.
        
        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
