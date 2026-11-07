"""Optimized program for BBOB f21 (Gallagher 101 peaks) with portfolio search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Portfolio CMA-ES-like search with restarts and adaptive step size."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 330s kill
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        total_calls = 0
        
        # Portfolio: multiple restarts with different initializations
        # f21 has 101 peaks, so we need good exploration
        n_restarts = 10
        calls_per_restart = self.budget // n_restarts
        
        for r in range(n_restarts):
            if total_calls >= self.budget:
                break
            if time.time() > deadline:
                break
            
            # Initialize: mix of center and random points
            if r == 0:
                x = np.zeros(self.dim)
            elif r == 1:
                x = rng.uniform(-4, 4, self.dim)
            else:
                # Cluster around previous best if available
                if best_x is not None and rng.random() < 0.5:
                    x = best_x + rng.normal(0, 1.0, self.dim)
                else:
                    x = rng.uniform(-4, 4, self.dim)
            
            x = np.clip(x, -5, 5)
            fx = func(x)
            total_calls += 1
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            # CMA-ES-like local search with restart
            sigma = 1.5
            C = np.eye(self.dim)  # covariance matrix
            step_size = sigma
            no_improve = 0
            max_no_improve = 30
            
            while total_calls < self.budget and no_improve < max_no_improve:
                if time.time() > deadline:
                    break
                
                # Generate candidate
                z = rng.standard_normal(self.dim)
                # Apply covariance matrix (Cholesky for efficiency)
                try:
                    L = np.linalg.cholesky(C)
                    y = x + step_size * (L @ z)
                except:
                    y = x + step_size * z
                
                y = np.clip(y, -5, 5)
                fy = func(y)
                total_calls += 1
                
                if fy < fx:
                    # Success: update
                    direction = (y - x) / step_size
                    # Update covariance (simple rank-1 update)
                    C = 0.9 * C + 0.1 * np.outer(direction, direction)
                    # Normalize C to maintain scale
                    C = C / np.trace(C) * self.dim
                    x, fx = y, fy
                    step_size *= 1.3
                    no_improve = 0
                    
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    no_improve += 1
                    step_size *= 0.75
                    # Occasionally randomize direction
                    if no_improve % 10 == 0:
                        C = 0.5 * C + 0.5 * np.eye(self.dim)
                        C = C / np.trace(C) * self.dim
            
            # Between restarts, if we have time, do a few more evaluations
            if total_calls < self.budget and time.time() < deadline:
                # Quick random search around best
                for _ in range(min(5, self.budget - total_calls)):
                    if time.time() > deadline:
                        break
                    y = best_x + rng.normal(0, 0.5, self.dim)
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    total_calls += 1
                    if fy < best_f:
                        best_f = fy
                        best_x = y.copy()
        
        return best_x, best_f
# EVOLVE-BLOCK-END
