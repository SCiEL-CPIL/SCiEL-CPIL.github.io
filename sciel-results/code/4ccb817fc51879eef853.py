"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Coordinate descent with adaptive per-dimension step sizes for Bent Cigar."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 290s margin within 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Initial point
        x = rng.uniform(-5, 5, dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx

        # Per-dimension step sizes
        sigma = np.full(dim, 1.0)

        # Count function evaluations
        calls = 1

        def time_ok():
            return time.time() < deadline and calls < budget

        # Phase 1: Coordinate descent with adaptive steps
        # We alternate between dimensions, trying to improve each one.
        # For each dimension, we do a small line search.
        
        # First, let's do a few full passes of coordinate descent to get a rough idea
        # of which dimensions matter (have larger curvature).
        
        # Adaptive coordinate descent
        for it in range(1, budget):
            if time.time() > deadline:
                break
            if calls >= budget:
                break
            
            # Cycle through dimensions
            d = it % dim
            
            # Try to optimize dimension d
            # Use a few trial steps in both directions
            improved = False
            # Try increasing step
            for _ in range(3):
                if not time_ok():
                    break
                x_trial = x.copy()
                x_trial[d] = np.clip(x[d] + sigma[d], -5, 5)
                calls += 1
                fx_trial = func(x_trial)
                if fx_trial < fx:
                    x = x_trial
                    fx = fx_trial
                    improved = True
                    sigma[d] *= 1.3
                    if fx < best_fx:
                        best_x = x.copy()
                        best_fx = fx
                else:
                    sigma[d] *= 0.7
                    break
                
            # Try decreasing step if we haven't improved
            if not improved:
                for _ in range(2):
                    if not time_ok():
                        break
                    x_trial = x.copy()
                    x_trial[d] = np.clip(x[d] - sigma[d], -5, 5)
                    calls += 1
                    fx_trial = func(x_trial)
                    if fx_trial < fx:
                        x = x_trial
                        fx = fx_trial
                        improved = True
                        sigma[d] *= 1.3
                        if fx < best_fx:
                            best_x = x.copy()
                            best_fx = fx
                    else:
                        sigma[d] *= 0.7
                        break

            # If no improvement in either direction, shrink step
            if not improved:
                sigma[d] *= 0.8

        return best_x, best_fx
# EVOLVE-BLOCK-END
