"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Hybrid optimizer for BBOB f4 (Büche-Rastrigin).
    Strategy:
    1. Sobol Sequence for dense global exploration (first 60 calls).
    2. Multi-Start (1+1)-ES with stagnation-based restarts for exploitation.
       This leverages the structural idea from the top-scoring inspiration (Multi-Start ES)
       which outperformed the previous CMA-ES approach (score 3.28 vs 0.63).
       The (1+1)-ES is lightweight and robust, and restarts help escape local minima
       in the multimodal landscape.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: 330s limit, allow 290s for safety
        deadline = start_time + 290.0
        rng = np.random.default_rng(42)
        
        dim = self.dim
        budget = self.budget
        calls = 0
        
        best_x = None
        best_f = np.inf

        # --- Stage 1: Global Exploration with Sobol Sequence ---
        # Büche-Rastrigin is multimodal with many local minima.
        # A structured quasi-random sequence covers the space better than pure random init.
        n_explore = min(60, budget)
        
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=dim, scramble=True, seed=123)
            # Generate power of 2 for efficiency, then slice
            next_pow2 = 1
            while next_pow2 < n_explore:
                next_pow2 *= 2
            points = sampler.random(next_pow2)[:n_explore]
        except Exception:
            # Fallback to random if scipy fails
            points = rng.random((n_explore, dim))
            
        # Map from [0, 1]^dim to [-5, 5]^dim
        points = points * 10.0 - 5.0
        
        for i in range(n_explore):
            if time.time() > deadline or calls >= budget:
                break
            x = points[i]
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            # Fallback if no calls were made
            x0 = np.zeros(dim)
            best_f = func(x0)
            best_x = x0.copy()
            calls = 1

        # --- Stage 2: Multi-Start (1+1)-ES with Restarts ---
        # The key insight from the high-scoring inspiration is that Multi-Start (1+1)-ES
        # with restarts is superior to CMA-ES for this specific task in the given budget.
        
        # Parameters for (1+1)-ES
        sigma = 1.0
        sigma_min = 1e-4
        sigma_max = 5.0
        
        # Restart logic:
        # We run a local search from the current best.
        # If we stagnate (no improvement for N steps), we restart from a new point
        # sampled near the global best or randomly.
        
        stagnation_window = 20  # Number of steps without improvement to trigger restart
        max_stagnation = 100    # Hard cap on stagnation steps per restart
        restarts = 0
        max_restarts = 10       # Limit total restarts to avoid wasting budget
        
        # We keep a list of "promising" points found during exploration to restart from
        # Actually, simple restarts from the global best with perturbation or random restarts
        # are standard. Let's use random restarts but biased towards the best region.
        
        while calls < budget and time.time() < deadline:
            # Start a local search trajectory
            # If it's the first run, start from global best.
            # Otherwise, we might have restarted.
            # Let's structure this as a loop of "trajectories".
            
            # Initialize trajectory state
            x_cur = best_x.copy()
            f_cur = best_f
            sigma_cur = 1.0
            
            # 1/5 success rule state
            success_count = 0
            window_size = 10
            steps_since_improvement = 0
            
            # Run this trajectory until stagnation or budget/time out
            while calls < budget and time.time() < deadline:
                # Generate candidate
                # (1+1)-ES step
                y = x_cur + sigma_cur * rng.standard_normal(dim)
                # Clip to box
                y = np.clip(y, -5, 5)
                
                fy = func(y)
                calls += 1
                
                if fy < f_cur:
                    # Improvement
                    x_cur = y
                    f_cur = fy
                    steps_since_improvement = 0
                    success_count += 1
                    
                    # Update global best
                    if f_cur < best_f:
                        best_f = f_cur
                        best_x = x_cur.copy()
                else:
                    steps_since_improvement += 1
                    # No improvement, but we don't update x_cur in (1+1) if it fails
                    # (Strict (1+1) ES: x_{t+1} = x_t if f(y) >= f(x_t))
                
                # Adapt step size based on success rate in the window
                # We use a simple version: if success rate > 1/5, increase sigma, else decrease.
                # For efficiency, we check every window_size steps.
                if steps_since_improvement == 0:
                    # Reset window counter logic? 
                    # A simpler adaptive scheme:
                    # If last step was success, increase sigma slightly.
                    # If failure, decrease sigma slightly.
                    # This is effectively (1+1)-ES with 1/5 rule approximation.
                    if fy < f_cur: # Note: f_cur is already updated if success
                         sigma_cur *= 1.05
                    else:
                         sigma_cur *= 0.95
                else:
                    # If we haven't improved for a while, shrink sigma faster to escape?
                    # Or just let the stagnation counter handle restart.
                    pass
                
                sigma_cur = np.clip(sigma_cur, sigma_min, sigma_max)
                
                # Check for stagnation
                if steps_since_improvement >= stagnation_window:
                    # Stagnation detected. 
                    # Check if we've hit the hard cap for this trajectory
                    if steps_since_improvement >= max_stagnation:
                        break # Break inner loop to restart
                    
                    # If we just crossed the threshold, we could shrink sigma drastically 
                    # or decide to restart immediately. 
                    # Given the "Multi-Start" strategy, restarting is the key.
                    # Let's restart if we've been stuck for the full window.
                    break

            # If we broke out due to stagnation, check if we should restart
            if calls < budget and time.time() < deadline:
                restarts += 1
                if restarts > max_restarts:
                    break
                
                # New starting point for the next trajectory
                # Option A: Random point in the box
                # Option B: Point near the global best (local restart)
                # Option C: Point from the initial Sobol set that was good but not best
                # Let's use a mix: 50% random, 50% near global best.
                if rng.random() < 0.5:
                    x_new_start = rng.uniform(-5, 5, dim)
                else:
                    # Perturb global best
                    x_new_start = best_x + rng.standard_normal(dim) * 2.0
                    x_new_start = np.clip(x_new_start, -5, 5)
                
                # Evaluate the new start point
                f_new_start = func(x_new_start)
                calls += 1
                
                if f_new_start < best_f:
                    best_f = f_new_start
                    best_x = x_new_start.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
