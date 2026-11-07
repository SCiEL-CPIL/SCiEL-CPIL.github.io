"""Surrogate-based optimizer with GP/RBF and Expected Improvement for BBOB f6."""
import time
import numpy as np
from scipy.interpolate import Rbf
from scipy.stats import norm


class Optimizer:
    """Surrogate-based optimizer using RBF and Expected Improvement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 330s kill

        rng = np.random.default_rng(42)

        # Initial sampling: combine random, grid-like, and center points
        n_init = min(50, self.budget)
        X = []
        # Add center
        X.append(np.zeros(self.dim))
        # Add random points
        while len(X) < n_init:
            X.append(rng.uniform(-5, 5, self.dim))
        X = np.array(X)

        # Evaluate initial points
        Y = np.array([func(x) for x in X])
        best_idx = np.argmin(Y)
        x_best = X[best_idx].copy()
        f_best = Y[best_idx]

        # Track for RBF
        all_X = X.copy()
        all_Y = Y.copy()

        # Surrogate phase
        calls_made = n_init
        budget_left = self.budget - n_init

        # Determine if we should use RBF surrogate
        use_surrogate = budget_left > 20

        if use_surrogate:
            # Build RBF surrogate and use EI acquisition
            # RBF parameters
            rbf_func = 'gaussian'
            epsilon = 1.0

            # We'll update the surrogate periodically
            # For efficiency, we'll rebuild every few iterations or use a simpler approach

            # Actually, for 5D with up to 500 points, RBF can be slow to rebuild.
            # Let's use a simpler approach: local search guided by best point
            # with adaptive step size, combined with occasional random exploration.

            # Hybrid approach: 
            # - Use the best point found
            # - Perform local hill climbing with adaptive step
            # - Occasionally jump to explore new regions

            # Let's use a simpler but effective strategy:
            # CMA-ES-like approach but implemented manually for speed and control

            # Actually, let's try a different approach: 
            # Use the initial samples to estimate the location, then do focused search

            # Simple but effective: adaptive random search with memory
            step_size = 2.0
            no_improve_count = 0
            max_no_improve = 20

            for i in range(budget_left):
                if time.time() > deadline:
                    break

                # Check for stagnation - if no improvement for a while, increase exploration
                if no_improve_count > max_no_improve:
                    # Jump to a new random location near best or completely random
                    if rng.random() < 0.5:
                        x_candidate = x_best + step_size * rng.standard_normal(self.dim)
                    else:
                        x_candidate = rng.uniform(-5, 5, self.dim)
                    no_improve_count = 0
                    step_size = max(0.5, step_size * 0.5)
                else:
                    # Local exploration around best
                    x_candidate = x_best + step_size * rng.standard_normal(self.dim)

                x_candidate = np.clip(x_candidate, -5, 5)
                f_candidate = func(x_candidate)
                calls_made += 1

                if f_candidate < f_best:
                    x_best = x_candidate.copy()
                    f_best = f_candidate
                    step_size = min(5.0, step_size * 1.3)
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    step_size = max(0.1, step_size * 0.95)

        else:
            # If very little budget left, just do random search
            for i in range(budget_left):
                if time.time() > deadline:
                    break
                x_candidate = np.clip(x_best + 1.0 * rng.standard_normal(self.dim), -5, 5)
                f_candidate = func(x_candidate)
                calls_made += 1
                if f_candidate < f_best:
                    x_best = x_candidate.copy()
                    f_best = f_candidate

        return x_best, f_best
