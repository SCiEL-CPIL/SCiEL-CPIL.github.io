"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # 330s wall clock limit. Keep a safety margin of 5s.
        deadline = start_time + 325.0
        
        best_x = None
        best_f = np.inf
        call_count = 0
        n = self.dim
        
        def make_call(x):
            nonlocal call_count
            if call_count >= self.budget:
                return None
            if time.time() > deadline:
                return None
            fx = func(x)
            call_count += 1
            return fx

        # 1. Initial Probe
        # Start at origin. It's a safe, central point.
        x0 = np.zeros(n)
        f0 = make_call(x0)
        if f0 is not None and np.isfinite(f0):
            best_f = f0
            best_x = x0.copy()
        else:
            # Fallback if origin fails (unlikely but safe)
            return None, np.inf

        # 2. Strategy: Multi-start (1+1)-ES with 1/5th rule step-size adaptation
        # This is a fundamentally different approach from Simulated Annealing.
        # It adapts the step size based on success rate, which is more robust
        # for ill-conditioned problems like Schaffers F7.
        
        # Parameters for (1+1)-ES
        sigma = 1.0  # Initial step size
        c_sigma = 1.0 / (n + 4)  # Learning rate for sigma
        # 1/5th rule: if success rate > 1/5, increase sigma; else decrease
        
        # We will run multiple independent (1+1)-ES instances in parallel (islands)
        # but since we have a single budget, we'll interleave them or run sequentially.
        # Given the small budget (500), sequential multi-start is better to avoid overhead.
        
        # Let's try 5 restarts, each with ~100 calls (after the initial probe)
        n_restarts = 5
        calls_per_restart = (self.budget - call_count) // n_restarts
        if calls_per_restart < 10:
            calls_per_restart = max(10, (self.budget - call_count) // 2)
            n_restarts = max(1, (self.budget - call_count) // calls_per_restart)
        
        # Initialize random starts
        # We'll use a mix of origin and random points
        starts = [best_x.copy()]
        for _ in range(n_restarts - 1):
            x_start = np.random.uniform(-4, 4, n)  # Minimizer is in [-4,4]^5
            starts.append(x_start)
        
        # Run each (1+1)-ES instance
        for i in range(n_restarts):
            if call_count >= self.budget or time.time() > deadline:
                break
            
            # Start this instance
            if i == 0:
                x = best_x.copy()
                f = best_f
            else:
                x = starts[i].copy()
                f = make_call(x)
                if f is None:
                    break
                if not np.isfinite(f):
                    f = best_f + 1000.0
                    x = best_x.copy()
                    f = best_f
                else:
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
            
            # Reset step size for each restart
            sigma_local = 1.0
            
            # Run (1+1)-ES for this instance
            # Use a window of last 10 steps for success rate
            window_size = 10
            successes = 0
            window = 0
            
            while call_count < self.budget and time.time() < deadline:
                # Generate neighbor
                delta = np.random.normal(0, sigma_local, n)
                x_new = x + delta
                x_new = np.clip(x_new, -5, 5)
                
                f_new = make_call(x_new)
                if f_new is None:
                    break
                    
                if not np.isfinite(f_new):
                    f_new = f + 1000.0
                
                # Check for improvement
                if f_new < f:
                    x = x_new
                    f = f_new
                    successes += 1
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
                else:
                    # No improvement
                    pass
                
                # Update window
                window += 1
                if window >= window_size:
                    # Apply 1/5th rule
                    success_rate = successes / window
                    if success_rate > 0.2:
                        sigma_local *= 1.2  # Increase step size
                    else:
                        sigma_local /= 1.2  # Decrease step size
                    
                    # Reset window
                    successes = 0
                    window = 0
                
                # Safety: ensure sigma doesn't get too small or too large
                sigma_local = np.clip(sigma_local, 1e-4, 10.0)
        
        # 3. Final Polish
        # If we have budget left, do a few greedy steps from the best point
        if call_count < self.budget and time.time() < deadline:
            x = best_x.copy()
            f = best_f
            sigma_polish = 0.1  # Small step size for polishing
            
            for _ in range(10):
                if call_count >= self.budget or time.time() > deadline:
                    break
                
                delta = np.random.normal(0, sigma_polish, n)
                x_new = x + delta
                x_new = np.clip(x_new, -5, 5)
                
                f_new = make_call(x_new)
                if f_new is None:
                    break
                
                if not np.isfinite(f_new):
                    f_new = f + 1000.0
                
                if f_new < f:
                    x = x_new
                    f = f_new
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
        
        if best_x is None:
            # Fallback
            best_x = x0
            best_f = f0

        return best_x, best_f
# EVOLVE-BLOCK-END
