"""Combined optimizer: lattice-based Rastrigin-aware search with adaptive local refinement."""
import numpy as np
import time


class Optimizer:
    """Lattice-based Rastrigin optimizer with adaptive local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s kill

        best_x = None
        best_f = np.inf
        calls_used = 0
        rng = np.random.default_rng(42)

        def try_point(x):
            nonlocal best_x, best_f, calls_used
            x = np.clip(x, -5.0, 5.0)
            f = float(func(x))
            calls_used += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()

        def time_ok():
            return calls_used < self.budget and time.time() < deadline

        # Phase 1: Lattice search exploiting Rastrigin periodicity
        # Rastrigin minima occur at integer points (after hidden shift/rotation).
        # We evaluate a structured grid: integers and half-integers in [-4, 4].
        # This gives 9 points per dimension (from -4 to 4 step 0.5), total 9^5 = 59049.
        # Too many for 500 budget, so we use a smarter approach:
        # - First, evaluate integers in [-4, 4]: 9^5 = 59049, still too many.
        # - Use a two-stage approach: coarse random + structured local refinement.

        # Stage 1a: Random sampling to identify promising regions
        n_random = 80
        for i in range(n_random):
            if not time_ok():
                break
            x = rng.uniform(-4.5, 4.5, self.dim)
            try_point(x)

        # Stage 1b: Structured lattice refinement around best point
        # Try offsets of 0.5 and 1.0 in each dimension from the best point
        if best_x is not None and time_ok():
            base = best_x.copy()
            # Try all combinations of offsets in a limited way
            # For 5D, we can't try all 3^5 = 243 combinations with full budget
            # Instead, do coordinate-wise sweeps with multiple step sizes
            
            # First, try larger steps to find the basin
            for step in [1.0, 0.5]:
                if not time_ok():
                    break
                improved = True
                while improved and time_ok():
                    improved = False
                    for d in range(self.dim):
                        if not time_ok():
                            break
                        for sign in [1, -1]:
                            if not time_ok():
                                break
                            x_trial = base.copy()
                            x_trial[d] += sign * step
                            x_trial = np.clip(x_trial, -5.0, 5.0)
                            f_trial = float(func(x_trial))
                            calls_used += 1
                            if f_trial < best_f:
                                best_f = f_trial
                                best_x = x_trial.copy()
                                base = x_trial.copy()
                                improved = True
                            elif f_trial < base_f if 'base_f' in locals() else True:
                                pass

        # Stage 2: Adaptive local refinement with shrinking step sizes
        if best_x is not None and time_ok():
            # Multiple refinement passes with decreasing step sizes
            step_sizes = [0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005]
            for ss in step_sizes:
                if not time_ok():
                    break
                improved = True
                max_iters = 10  # Limit iterations per step size to conserve budget
                iter_count = 0
                while improved and time_ok() and iter_count < max_iters:
                    improved = False
                    iter_count += 1
                    for d in range(self.dim):
                        if not time_ok():
                            break
                        for sign in [1, -1]:
                            if not time_ok():
                                break
                            x_trial = best_x.copy()
                            x_trial[d] += sign * ss
                            x_trial = np.clip(x_trial, -5.0, 5.0)
                            f_trial = float(func(x_trial))
                            calls_used += 1
                            if f_trial < best_f:
                                best_f = f_trial
                                best_x = x_trial.copy()
                                improved = True

        # Stage 3: Final fine-tuning with very small steps if budget remains
        if best_x is not None and time_ok():
            for ss in [0.002, 0.001]:
                if not time_ok():
                    break
                for d in range(self.dim):
                    if not time_ok():
                        break
                    for sign in [1, -1]:
                        if not time_ok():
                            break
                        x_trial = best_x.copy()
                        x_trial[d] += sign * ss
                        x_trial = np.clip(x_trial, -5.0, 5.0)
                        f_trial = float(func(x_trial))
                        calls_used += 1
                        if f_trial < best_f:
                            best_f = f_trial
                            best_x = x_trial.copy()

        return best_x, best_f
