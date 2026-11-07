"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Derivative-free trust-region reflective method with local quadratic models."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        n = self.dim
        budget = self.budget

        # State
        x = np.array([0.0] * n)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx

        # Trust region radius
        delta = 2.0
        delta_min = 1e-6
        delta_max = 4.0

        # Step counter for budget management
        calls_made = 1

        # Powell's UOBYQA-like algorithm
        # We maintain a set of interpolation points to build a local quadratic model
        # For simplicity and robustness, we use a coordinate-descent based approach
        # with adaptive step sizes, combined with occasional global perturbations.

        # Adaptive step sizes per dimension
        steps = np.ones(n) * 1.0

        # Main loop
        while calls_made < budget:
            # Check deadline
            if time.time() > deadline:
                break

            improved = False
            for i in range(n):
                if time.time() > deadline or calls_made >= budget:
                    break

                # Try positive and negative steps along dimension i
                for sign in [1.0, -1.0]:
                    if calls_made >= budget:
                        break

                    trial = x.copy()
                    trial[i] += sign * steps[i]
                    # Reflect to stay in bounds
                    trial = np.clip(trial, -5.0, 5.0)

                    f_trial = func(trial)
                    calls_made += 1

                    if f_trial < fx:
                        x = trial
                        fx = f_trial
                        steps[i] *= 1.5  # Expand step
                        improved = True

                        if fx < best_fx:
                            best_fx = fx
                            best_x = x.copy()
                    else:
                        steps[i] *= 0.7  # Shrink step

            # If no improvement, try a larger random perturbation
            if not improved:
                # Reduce all steps
                steps *= 0.8

                # If steps are too small, do a global restart-like perturbation
                if np.all(steps < 1e-4):
                    # Random restart within a smaller region
                    rng = np.random.default_rng()
                    perturbation = rng.uniform(-1.0, 1.0, n)
                    trial = np.clip(x + perturbation, -5.0, 5.0)
                    f_trial = func(trial)
                    calls_made += 1
                    if f_trial < fx:
                        x = trial
                        fx = f_trial
                        if fx < best_fx:
                            best_fx = fx
                            best_x = x.copy()
                    steps = np.ones(n) * 0.5

        return best_x, best_fx
# EVOLVE-BLOCK-END
