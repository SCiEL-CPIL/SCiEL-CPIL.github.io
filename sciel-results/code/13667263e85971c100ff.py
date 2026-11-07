"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with restarts and polish for Büche-Rastrigin optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s kill
        n = self.dim
        budget = self.budget
        
        # Global best tracking
        best_x = None
        best_f = np.inf
        total_calls = 0
        
        # Helper to check time and calls
        def check_deadline():
            nonlocal total_calls
            return time.time() > deadline or total_calls >= budget

        # (1+1)-ES implementation
        def run_1plus1_es(x0, sigma0, max_calls, deadline, remaining_calls):
            nonlocal best_x, best_f, total_calls
            
            x = x0.copy()
            f_x = func(x)
            total_calls += 1
            remaining_calls -= 1
            
            if f_x < best_f:
                best_f = f_x
                best_x = x.copy()
            
            sigma = sigma0
            calls_in_run = 1
            
            while calls_in_run < max_calls:
                if time.time() > deadline or remaining_calls <= 0:
                    break
                
                # Generate offspring
                z = np.random.standard_normal(n)
                y = x + sigma * z
                y = np.clip(y, -5, 5)
                
                f_y = func(y)
                total_calls += 1
                remaining_calls -= 1
                calls_in_run += 1
                
                if f_y < f_x:
                    # Success
                    x = y
                    f_x = f_y
                    # Increase step size
                    sigma *= 1.05
                    # Update global best
                    if f_x < best_f:
                        best_f = f_x
                        best_x = x.copy()
                else:
                    # Failure
                    # Decrease step size
                    sigma *= 0.95
                
                # Keep sigma within reasonable bounds
                sigma = max(sigma, 1e-8)
                sigma = min(sigma, 10.0)
            
            return remaining_calls

        # Strategy:
        # 1. Initial exploration with a few (1+1)-ES runs from different starting points.
        # 2. If a run stagnates (sigma becomes very small or no improvement), restart.
        # 3. Final polish with very small sigma.

        # Phase 1: Exploration
        # We will do 3 main (1+1)-ES runs.
        # Run 1: Start at center, moderate sigma.
        # Run 2: Start at best from Run 1, but with a random kick if it got stuck, or just continue.
        # Actually, better to restart from the global best with a fresh sigma if the previous run is exhausted.
        
        # Let's allocate budget:
        # 150 calls for Run 1
        # 150 calls for Run 2
        # 150 calls for Run 3
        # 50 calls for Polish
        
        runs = [
            {"x0": np.zeros(n), "sigma0": 2.0, "calls": 150},
            {"x0": np.random.uniform(-3, 3, n), "sigma0": 1.5, "calls": 150},
            {"x0": np.random.uniform(-4, 4, n), "sigma0": 3.0, "calls": 150},
        ]
        
        # However, starting from random points every time is inefficient if we found a good point.
        # Better: Run 1 from center. Run 2 from best of Run 1 (or random if Run 1 failed?).
        # Let's stick to a simpler, robust strategy:
        # Run 1: Center, sigma=2.0, 100 calls.
        # Run 2: Best from Run 1, sigma=1.0, 100 calls. (Refinement)
        # Run 3: Best from Run 2, sigma=0.5, 100 calls. (Further refinement)
        # Run 4: Best from Run 3, sigma=0.1, 50 calls. (Polish)
        # This sequential refinement is very effective for smooth-ish landscapes like f4 after the initial jump.
        
        # But f4 has local minima. If Run 1 gets stuck in a local min, Runs 2-4 will just polish that local min.
        # To mitigate this, we should check for stagnation.
        # If no improvement in last K steps, restart with a larger sigma or from a random point.
        
        # Let's implement a single loop with adaptive restarts.
        
        # Initial point
        x = np.zeros(n)
        f_x = func(x)
        total_calls += 1
        best_f = f_x
        best_x = x.copy()
        
        sigma = 2.0
        
        # We will run for the remaining budget.
        # We monitor improvement. If we don't improve for a long time, we might be stuck.
        # For simplicity and robustness, we'll use the sequential refinement strategy but with a "kick" if stuck.
        
        # Actually, the inspiration says (1+1)-ES first is best. Let's just run one long (1+1)-ES with adaptive sigma.
        # The adaptation (increase on success, decrease on failure) naturally handles the scale.
        # If it gets stuck in a local min, sigma will shrink to near zero.
        # We can detect this: if sigma < 1e-4 and no improvement for last 50 calls, restart from best_x with a random perturbation.
        
        remaining_calls = budget - 1
        no_improve_count = 0
        last_improve_call = 0
        
        while remaining_calls > 0:
            if time.time() > deadline:
                break
            
            # Check for stagnation
            if sigma < 1e-4 and (total_calls - last_improve_call) > 50:
                # Stuck. Restart from best_x with a random kick and larger sigma.
                x = best_x + np.random.standard_normal(n) * 0.5
                x = np.clip(x, -5, 5)
                f_x = func(x)
                total_calls += 1
                remaining_calls -= 1
                if f_x < best_f:
                    best_f = f_x
                    best_x = x.copy()
                    last_improve_call = total_calls
                else:
                    # Keep old best, but try to move from there?
                    # Actually, just restart the search from the best point with a new sigma
                    x = best_x.copy()
                    f_x = best_f
                    sigma = 1.0
                    no_improve_count = 0
                    continue
            
            # Generate offspring
            z = np.random.standard_normal(n)
            y = x + sigma * z
            y = np.clip(y, -5, 5)
            
            f_y = func(y)
            total_calls += 1
            remaining_calls -= 1
            
            if f_y < f_x:
                x = y
                f_x = f_y
                sigma *= 1.05
                no_improve_count = 0
                last_improve_call = total_calls
                if f_x < best_f:
                    best_f = f_x
                    best_x = x.copy()
            else:
                sigma *= 0.95
                no_improve_count += 1
            
            sigma = max(sigma, 1e-8)
            sigma = min(sigma, 10.0)
            
            # If we have very few calls left, reduce sigma to ensure we don't jump out of the basin
            if remaining_calls < 20:
                sigma = min(sigma, 0.01)
        
        return best_x, best_f

# EVOLVE-BLOCK-END
