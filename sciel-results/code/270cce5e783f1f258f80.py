"""Optimized search for BBOB f15 (rotated Rastrigin).
Combines multi-start (1+1)-ES exploitation with a best-first memory structure
to escape local minima, followed by a final local polish.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-Start (1+1)-ES with best-first memory and local polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(12345)
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280  # Safety margin

        # Global best tracking
        best_x = None
        best_f = np.inf
        call_count = 0

        def eval_func(x):
            nonlocal call_count, best_x, best_f
            if call_count >= budget:
                return np.inf
            call_count += 1
            x_clipped = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            fx = float(func(x_clipped))
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx

        def time_ok():
            return time.time() < deadline

        def budget_ok():
            return call_count < budget

        # --- Phase 1: Initialization ---
        # Start with a small diverse population
        init_count = 10
        history = []  # List of (fx, x)
        
        for _ in range(init_count):
            if not budget_ok() or not time_ok():
                break
            x = np.clip(rng.uniform(-5, 5, dim), -5, 5)
            fx = eval_func(x)
            history.append((fx, x))
        
        if not history:
            return best_x, best_f

        # Sort history by fitness
        history.sort(key=lambda item: item[0])
        
        # Current state for (1+1)-ES
        current_x = history[0][1].copy()
        current_f = history[0][0]
        
        # (1+1)-ES parameters
        sigma = 1.0
        stagnation_limit = 30
        stagnation_counter = 0
        
        # Memory of best points for restarts
        # We keep the top 5 distinct points
        # We need to track which points are "active" or "exploited"
        # Simple strategy: If stuck, pick the next best from history that hasn't been used as a start recently.
        used_indices = {0}
        
        while budget_ok() and time_ok():
            # (1+1)-ES Step
            # Generate child
            child_x = current_x + sigma * rng.standard_normal(dim)
            child_x = np.clip(child_x, -5.0, 5.0)
            
            child_f = eval_func(child_x)
            
            if child_f < current_f:
                # Accepted
                current_x = child_x
                current_f = child_f
                stagnation_counter = 0
                # Add to history
                history.append((child_f, child_x.copy()))
                # Keep history small and sorted
                if len(history) > 20:
                    history = sorted(history, key=lambda item: item[0])[:20]
                
                # Adaptive sigma: increase slightly on success
                sigma = min(sigma * 1.05, 5.0)
            else:
                # Rejected
                stagnation_counter += 1
                # Adaptive sigma: decrease on failure
                sigma = max(sigma * 0.9, 0.05)
            
            # Check for Stagnation
            if stagnation_counter >= stagnation_limit:
                # Restart from a different point in history
                # Find the best point in history not in used_indices
                restart_found = False
                # Sort history again to ensure we pick the best available
                history.sort(key=lambda item: item[0])
                
                for i, (fx, x) in enumerate(history):
                    if i not in used_indices:
                        # Restart here
                        current_x = x.copy()
                        current_f = fx
                        used_indices.add(i)
                        sigma = 1.5  # Reset sigma to medium
                        stagnation_counter = 0
                        restart_found = True
                        break
                
                if not restart_found:
                    # If all top points used, generate a new random point
                    # Or just continue with a larger sigma
                    new_x = np.clip(rng.uniform(-5, 5, dim), -5, 5)
                    new_f = eval_func(new_x)
                    history.append((new_f, new_x))
                    history.sort(key=lambda item: item[0])
                    current_x = history[0][1].copy()
                    current_f = history[0][0]
                    used_indices = {0}
                    sigma = 2.0
                    stagnation_counter = 0

        # --- Phase 2: Final Local Polish ---
        # Use remaining budget to polish the global best point
        if best_x is not None and budget_ok() and time_ok():
            x_polish = best_x.copy()
            f_polish = best_f
            
            # Coordinate-wise fine search
            # Use a small step size
            step = 0.1
            # Try shrinking step if no improvement
            for _ in range(5): # Max 5 rounds of polishing
                if not budget_ok() or not time_ok():
                    break
                improved = False
                for axis in range(dim):
                    if not budget_ok() or not time_ok():
                        break
                    # Try +/- step
                    for delta in [-step, step]:
                        y = x_polish.copy()
                        y[axis] += delta
                        y = np.clip(y, -5.0, 5.0)
                        fy = eval_func(y)
                        if fy < f_polish:
                            f_polish = fy
                            x_polish = y
                            improved = True
                            # If improved, try smaller step in same direction? 
                            # For simplicity, just accept and move to next axis
                            break
                if not improved:
                    step *= 0.5
                    if step < 0.01:
                        break
                else:
                    # If improved, keep step size or slightly reduce
                    step = max(step * 0.8, 0.01)
            
            # Update global best if polish found something better
            if f_polish < best_f:
                best_f = f_polish
                best_x = x_polish

        return best_x, best_f
# EVOLVE-BLOCK-END
