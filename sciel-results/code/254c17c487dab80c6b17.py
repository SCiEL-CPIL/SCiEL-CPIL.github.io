"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-start (1+1)-ES for BBOB f20 (Schwefel)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.time_limit = 290.0  # Safety margin for 300s limit
        self.start_time = None

    def __call__(self, func):
        self.start_time = time.time()
        rng = np.random.default_rng(42)
        
        # Phase 1: Global Exploration
        # Use a subset of the budget for global search to find the basin of attraction.
        # For dim=5, a few dozen points is reasonable.
        n_global = min(50, self.budget // 10)
        
        # Generate diverse points using Latin Hypercube Sampling concept
        # Or simply random uniform with a few structured points
        points = []
        # Add center and corners
        points.append(np.zeros(self.dim))
        for i in range(self.dim):
            for j in range(2):
                p = np.zeros(self.dim)
                p[i] = 5.0 if j else -5.0
                points.append(p)
        
        # Add random points
        while len(points) < n_global:
            points.append(rng.uniform(-5, 5, self.dim))
        
        best_x = None
        best_f = float('inf')
        
        # Evaluate global points
        for p in points:
            if time.time() - self.start_time > self.time_limit:
                break
            fp = func(p)
            if fp < best_f:
                best_f = fp
                best_x = p.copy()
            
            if len(points) >= n_global:
                break

        if best_x is None:
            # Fallback if no points evaluated
            best_x = np.zeros(self.dim)
            best_f = func(best_x)

        # Phase 2: Local Exploitation
        # Use (1+1)-ES starting from best_x
        x = best_x
        fx = best_f
        sigma = 0.5  # Start with a moderate step size
        calls_made = n_global
        
        # Adaptive step size based on 1/5 success rule
        # We don't have a full archive, so we use a simple heuristic:
        # If improvement, increase sigma; if not, decrease.
        # To be more robust, we can track a window of successes.
        success_count = 0
        window_size = 10
        successes_in_window = []

        while calls_made < self.budget:
            if time.time() - self.start_time > self.time_limit:
                break
            
            # Generate candidate
            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)
            calls_made += 1
            
            if fy < fx:
                x, fx = y, fy
                successes_in_window.append(1)
                # Increase step size if we are making progress
                sigma *= 1.2
            else:
                successes_in_window.append(0)
                # Decrease step size if we are stuck
                sigma *= 0.9

            # Maintain window
            if len(successes_in_window) > window_size:
                successes_in_window.pop(0)
            
            # Adjust sigma based on success rate
            if len(successes_in_window) == window_size:
                success_rate = np.mean(successes_in_window)
                if success_rate > 0.2:
                    sigma *= 1.1
                elif success_rate < 0.1:
                    sigma *= 0.9

            # Ensure sigma doesn't get too small or too large
            sigma = np.clip(sigma, 1e-6, 10.0)

        return x, fx
# EVOLVE-BLOCK-END
