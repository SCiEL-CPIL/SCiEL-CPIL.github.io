"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Adaptive Coordinate Descent tuned for Bent Cigar anisotropy."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s kill

        rng = np.random.default_rng(12345)

        # Initial point: center is often good for shifted problems
        x = np.zeros(self.dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx

        # Per-dimension step sizes
        steps = np.ones(self.dim) * 1.0
        # Track success count per dimension
        successes = np.zeros(self.dim, dtype=int)
        failures = np.zeros(self.dim, dtype=int)

        # Adaptive coordinate descent
        calls_used = 1
        stagnation_count = 0

        while calls_used < self.budget:
            if time.time() > deadline:
                break

            # Pick a dimension to try
            # Weighted random selection: dimensions with more recent success get slightly less weight
            # to encourage exploration of others, but we want to exploit the narrow valley
            # For Bent Cigar, one dimension is much narrower. We want to shrink steps there fast.
            
            dim_idx = rng.integers(0, self.dim)
            
            # Try positive and negative direction
            for sign in (1, -1):
                if calls_used >= self.budget or time.time() > deadline:
                    break
                    
                y = x.copy()
                y[dim_idx] = np.clip(y[dim_idx] + sign * steps[dim_idx], -5, 5)
                fy = func(y)
                calls_used += 1
                
                if fy < fx:
                    # Improvement
                    x = y
                    fx = fy
                    successes[dim_idx] += 1
                    # Grow step slightly on success
                    steps[dim_idx] *= 1.2
                    stagnation_count = 0
                else:
                    # No improvement
                    failures[dim_idx] += 1
                    # Shrink step on failure
                    steps[dim_idx] *= 0.7
                    stagnation_count += 1

            # Update best
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

            # If step size becomes too small in a dimension, reset it occasionally
            # to allow escape from local minima (though Bent Cigar is unimodal)
            for i in range(self.dim):
                if steps[i] < 1e-6:
                    steps[i] = 0.1
                    successes[i] = 0
                    failures[i] = 0

            # Periodically try a random restart if stuck
            if stagnation_count > 50 and calls_used < self.budget * 0.8:
                # Small probability of a larger random jump
                if rng.random() < 0.05:
                    x_new = x + rng.standard_normal(self.dim) * 0.5
                    x_new = np.clip(x_new, -5, 5)
                    fx_new = func(x_new)
                    calls_used += 1
                    if fx_new < fx:
                        x = x_new
                        fx = fx_new
                        steps = np.ones(self.dim) * 0.5
                        stagnation_count = 0

        return best_x, best_fx
# EVOLVE-BLOCK-END
