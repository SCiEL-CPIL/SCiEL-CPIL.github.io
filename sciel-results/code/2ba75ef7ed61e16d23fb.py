"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with zero-centered init, stable step-size adaptation, and time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget

        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_f = fx
        sigma = 2.0
        calls_made = 1

        # Main (1+1)-ES phase with adaptive step size
        # Use slightly more aggressive growth for early exploration
        for _ in range(budget - 1):
            if time.time() > deadline:
                break
            z = rng.standard_normal(dim)
            y = np.clip(x + sigma * z, -5, 5)
            fy = func(y)
            calls_made += 1

            if fy < fx:
                x, fx = y, fy
                if fx < best_f:
                    best_f, best_x = fx, x.copy()
                # Slightly more aggressive growth for faster convergence
                sigma = min(sigma * 1.25, 15.0)
            else:
                sigma = max(sigma * 0.92, 1e-6)

        # Final high-precision polish pass with geometrically decreasing sigma
        x = best_x.copy()
        fx = best_f
        remaining = budget - calls_made
        if remaining > 0:
            # Start with a smaller sigma for finer search
            sigma = 0.05
            # Use a fraction of remaining budget for polishing
            polish_budget = min(remaining, max(50, budget // 5))
            
            for i in range(polish_budget):
                if time.time() > deadline:
                    break
                # Gradually decrease sigma for high precision
                decay_factor = 0.98
                sigma *= decay_factor
                if sigma < 1e-8:
                    sigma = 1e-8
                    
                z = rng.standard_normal(dim)
                y = np.clip(x + sigma * z, -5, 5)
                fy = func(y)
                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f, best_x = fx, x.copy()
                    # Small boost on success to explore nearby
                    sigma = min(sigma * 1.05, 0.5)
                else:
                    # More aggressive reduction on failure
                    sigma = max(sigma * 0.85, 1e-10)

        return best_x, best_f
# EVOLVE-BLOCK-END
