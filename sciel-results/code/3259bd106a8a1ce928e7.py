"""Rewritten optimizer using Simulated Annealing with adaptive step size."""
import numpy as np
import time


class Optimizer:
    """
    Simulated Annealing optimizer for black-box continuous optimization.
    Designed to handle flat plateaus (like BBOB f7) by allowing uphill moves
    and adapting the step size based on the acceptance rate.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: aim to finish within 250s to be safe against 300s limit
        max_time = 250.0
        
        n = self.dim
        budget = self.budget
        
        # Initialize
        # Use a deterministic seed for reproducibility in local tests, 
        # but in the evaluator, randomness is usually fine. 
        # However, fixed seed is safer for consistent behavior if no rng is passed.
        rng = np.random.default_rng(12345)
        
        # Start at origin, which is often a good heuristic for symmetric boxes
        # or use a random start. Origin is safer for step functions centered near 0.
        x = np.zeros(n)
        fx = func(x)
        
        # Annealing parameters
        # Initial temperature should be large enough to explore the space [-5, 5]^5
        # A step size of 2.0 covers a significant portion of the domain.
        T = 2.0 
        # Cooling rate: geometric decay
        # We want T to drop significantly over the budget.
        # alpha = 0.995 means T decreases by ~0.5% per step.
        # After 500 steps: 0.995^500 ~ 0.08. So T goes from 2.0 to 0.16.
        # This seems reasonable.
        alpha = 0.995
        
        # Adaptive step size parameters
        # We track the acceptance rate over a window to adjust T (step size)
        # If acceptance is too low, increase T. If too high, decrease T.
        # This is a simplified version of the "success rule" in ES.
        
        # State for adaptive step size
        recent_accepts = 0
        window_size = 20
        # Target acceptance rate: typically around 0.2-0.5 is good for SA/ES
        target_accept_rate = 0.3 
        
        # We will iterate until budget is exhausted or time limit is reached
        # We track elapsed time every few steps to avoid overhead of time.time() every single step
        # but 500 steps is small, so time.time() is cheap.
        
        for i in range(budget - 1):
            # Check time limit
            if time.time() - start_time > max_time:
                break
            
            # Generate a neighbor
            # Step size is proportional to T
            # We use a Gaussian distribution for the step
            step = T * rng.standard_normal(n)
            x_new = np.clip(x + step, -5, 5)
            
            # Evaluate
            fx_new = func(x_new)
            
            # Acceptance criterion
            # 1. If better, always accept
            # 2. If worse, accept with probability exp(-delta / T)
            if fx_new <= fx:
                accept = True
            else:
                delta = fx_new - fx
                # Avoid division by zero or underflow issues
                if T > 1e-10:
                    prob = np.exp(-delta / T)
                else:
                    prob = 0.0
                accept = (rng.random() < prob)
            
            if accept:
                x = x_new
                fx = fx_new
                recent_accepts += 1
            
            # Cool down
            T *= alpha
            
            # Adaptive Step Size Adjustment
            # Check every 'window_size' steps
            if (i + 1) % window_size == 0:
                accept_rate = recent_accepts / window_size
                
                # If acceptance rate is too low, we are stuck or T is too small.
                # Increase T to explore more.
                if accept_rate < 0.1:
                    T *= 1.5
                # If acceptance rate is too high, we are moving too much randomly.
                # Decrease T to refine.
                elif accept_rate > 0.6:
                    T *= 0.7
                # Otherwise, keep T as is (cooling continues)
                
                # Reset counter
                recent_accepts = 0
                
            # Ensure T doesn't get too small to prevent stalling
            # Also ensure T doesn't get too large to prevent wild jumps
            T = np.clip(T, 1e-4, 10.0)

        return x, fx
