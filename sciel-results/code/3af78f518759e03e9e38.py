"""Rewritten optimizer: Sobol quasi-random global search with adaptive local exploitation."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc


class Optimizer:
    """Sobol-based global search with adaptive step-size exploitation for Rastrigin-like functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin

        rng = np.random.default_rng(42)

        # Initialize Sobol sampler
        sobol = qmc.Sobol(d=self.dim, scramble=True)
        # Generate a large batch of Sobol points upfront (log2(2^20) = 20, so 2^20 points)
        n_sobol = 2 ** 18  # 262144 points, plenty for 500 calls
        sobol_points = sobol.random(n_sobol)  # shape (n_sobol, dim) in [0,1]^d
        # Map to [-5, 5]^d
        sobol_points = sobol_points * 10.0 - 5.0

        # State
        x_best = np.zeros(self.dim)
        fx_best = np.inf
        call_count = 0
        sigma = 2.0
        last_improvement = 0
        stall_counter = 0

        def evaluate(x):
            nonlocal x_best, fx_best, call_count, last_improvement, stall_counter
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            call_count += 1
            if fx < fx_best - 1e-12:
                x_best = x.copy()
                fx_best = fx
                last_improvement = call_count
                stall_counter = 0
            else:
                stall_counter += 1
            return fx

        # Phase 1: Systematic Sobol exploration
        # Use Sobol points to cover the domain uniformly
        i_sobol = 0
        phase1_calls = min(200, self.budget)  # Use first 200 calls for exploration

        for i in range(phase1_calls):
            if time.time() > deadline:
                break
            if call_count >= self.budget:
                break

            x = sobol_points[i_sobol]
            i_sobol += 1
            evaluate(x)

        # Phase 2: Adaptive local search around best point
        # Use smaller steps, exploiting Rastrigin periodicity
        # Rastrigin has local minima at integer points (after rotation/shift, they're shifted)
        # We'll do a multi-scale local search

        remaining = self.budget - call_count
        if remaining <= 0:
            return x_best, fx_best

        # Adaptive step sizes: start moderate, shrink over time
        # Use a schedule that tries multiple scales
        step_scales = [1.5, 1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01]
        scale_idx = 0
        calls_in_phase = 0
        max_phase2_calls = remaining

        for _ in range(max_phase2_calls):
            if time.time() > deadline:
                break
            if call_count >= self.budget:
                break

            # If we've been stalled, try a larger jump (restart-like)
            if stall_counter > 30:
                # Try a random restart near a different region
                x_try = x_best + 3.0 * rng.standard_normal(self.dim)
                fx_try = evaluate(x_try)
                if fx_try < fx_best:
                    # Reset stall
                    stall_counter = 0
                    # Stay at current scale
                else:
                    # Shrink step
                    if scale_idx < len(step_scales) - 1:
                        scale_idx += 1
                calls_in_phase += 1
                continue

            # Normal local search
            step_size = step_scales[scale_idx]
            x_try = x_best + step_size * rng.standard_normal(self.dim)
            fx_try = evaluate(x_try)
            calls_in_phase += 1

            if fx_try < fx_best:
                # Improved, keep step size or slightly increase
                if scale_idx > 0 and stall_counter < 10:
                    scale_idx -= 1  # Try larger steps again
            else:
                # No improvement
                if stall_counter > 15:
                    if scale_idx < len(step_scales) - 1:
                        scale_idx += 1  # Shrink step

        return x_best, fx_best
# EVOLVE-BLOCK-END
