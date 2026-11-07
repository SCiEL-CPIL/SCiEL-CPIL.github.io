import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        def time_ok():
            return time.time() < deadline

        best_x = np.zeros(dim)
        best_f = float('inf')
        calls = 0

        def call_func(x):
            nonlocal calls, best_x, best_f
            if calls >= budget or not time_ok():
                return None
            x = np.clip(x, -5, 5)
            f_val = func(x)
            calls += 1
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # Phase 1: Sobol-based global exploration for good initial coverage
        n_global = min(60, budget // 5)
        
        # Add origin
        if calls < budget and time_ok():
            call_func(np.zeros(dim))

        # Use a quasi-random sequence for better coverage in low dimensions
        # Simple Halton-like or just uniform with stratification
        # For 5D, 60 points is sparse, but f4 has a large basin.
        # Let's use a mix: some random, some structured.
        for i in range(n_global - 1):
            if calls >= budget or not time_ok():
                break
            # Stratified sampling: divide each dimension into intervals
            # But simpler: just uniform random is often enough if we have a good local search
            # Let's use random but with a slight bias towards center? No, uniform is safer.
            x = rng.uniform(-4.5, 4.5, dim)
            call_func(x)

        if calls >= budget or not time_ok():
            return best_x, best_f

        # Phase 2: Multi-restart (1+1)-ES with adaptive sigma
        # This is often more robust and faster to converge than CMA-ES for small budgets
        # and simple landscapes, especially with restarts.
        
        max_time = 270.0 # Leave 10s margin
        
        # We will run a (1+1)-ES with restarts.
        # Each restart starts from a different point.
        # The first restart starts from the best point found in Phase 1.
        
        # Parameters for (1+1)-ES
        # Step size adaptation: 1/5 rule
        # p_success target = 0.2
        # If success rate > 0.2, increase sigma
        # If success rate < 0.2, decrease sigma
        
        # We'll implement a simple version:
        # sigma starts at 1.0
        # After every 10 steps, check success rate.
        # If > 0.25, sigma *= 1.5
        # If < 0.15, sigma /= 1.5
        
        # Restart logic:
        # If no improvement in last 20 steps, restart from a random point
        # or from the global best with a larger sigma.
        
        # Let's use a population-based approach instead? No, (1+1) is simpler and often effective.
        # But the inspiration mentioned "continuous multi-restart (1+1)-ES with adaptive sigma".
        
        # Let's implement a robust (1+1)-ES.
        
        # We will allocate the remaining budget to this phase.
        # We will run multiple independent (1+1)-ES runs, each with a portion of the budget,
        # or a single run with restarts.
        
        # A single run with restarts is better to keep the global best.
        
        x = best_x.copy()
        sigma = 1.0
        last_improvement = 0
        steps_since_improvement = 0
        success_count = 0
        total_steps = 0
        
        # We'll use a fixed number of steps per "epoch" to adapt sigma
        epoch_len = 10
        
        while calls < budget and time.time() < deadline - 5.0:
            # Generate a new point
            z = rng.standard_normal(dim)
            x_new = x + sigma * z
            
            f_new = call_func(x_new)
            if f_new is None:
                break
            
            total_steps += 1
            
            if f_new < best_f:
                # Improvement
                x = x_new
                best_f = f_new
                best_x = x_new.copy()
                success_count += 1
                steps_since_improvement = 0
                last_improvement = total_steps
            else:
                steps_since_improvement += 1
            
            # Adapt sigma every epoch_len steps
            if total_steps % epoch_len == 0:
                success_rate = success_count / epoch_len
                # Reset counters for next epoch
                success_count = 0
                
                if success_rate > 0.25:
                    sigma *= 1.5
                elif success_rate < 0.15:
                    sigma /= 1.5
                
                # Clamp sigma
                sigma = np.clip(sigma, 0.01, 5.0)
            
            # Restart logic
            # If no improvement for a long time, restart from a random point
            # or from the global best with a larger sigma.
            if steps_since_improvement > 30:
                # Restart from a random point in the box
                # Or from the global best with a larger sigma to escape local min
                # Let's try restarting from a random point to explore other basins
                x = rng.uniform(-4.5, 4.5, dim)
                sigma = 2.0 # Larger step size for exploration
                steps_since_improvement = 0
                success_count = 0
                total_steps = 0 # Reset epoch counter? No, keep total steps for time limit
                # Actually, resetting total_steps might mess up the epoch logic.
                # Let's just reset the local counters for adaptation.
                # We'll keep total_steps for the main loop condition.
                # But the epoch logic depends on total_steps % epoch_len.
                # If we reset total_steps, the modulo will change.
                # It's better to not reset total_steps, but just reset the success_count.
                # The restart is a heuristic.
                # Let's just reset the adaptation counters.
                success_count = 0
                # And keep total_steps as is.
                # But the "steps_since_improvement" is reset.
                # This should trigger exploration.

        return best_x, best_f
