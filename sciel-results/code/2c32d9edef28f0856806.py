"""Combined optimizer for BBOB f9 (rotated Rosenbrock) with structured search and refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Hybrid optimizer for rotated Rosenbrock (f9):
      1. Quick evaluation of structured candidates (including Rosenbrock minimum).
      2. Broad random sampling to find promising basins.
      3. Population-based local search with adaptive step sizes.
      4. Coordinate descent polish at the end.
    Explicit wall-clock deadline guard to stay within limits.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t_start = time.time()
        deadline = t_start + 265.0  # safety margin before 300s kill

        def timed():
            return (time.time() - t_start) > (deadline - t_start)

        def clip_box(x):
            x = np.clip(x, -5.0, 5.0)
            x = np.nan_to_num(x, nan=0.0, posinf=5.0, neginf=-5.0)
            return x

        best_x = None
        best_f = np.inf
        calls_used = 0
        rng = np.random.default_rng(42)

        def evaluate(x):
            nonlocal best_x, best_f, calls_used
            if calls_used >= budget or timed():
                return np.inf
            x_safe = clip_box(x)
            try:
                f_val = func(x_safe)
            except Exception:
                return np.inf
            calls_used += 1
            if np.isfinite(f_val) and f_val < best_f:
                best_f = f_val
                best_x = x_safe.copy()
            return f_val

        # --- Phase 1: Structured candidates ---
        candidates = [
            np.ones(dim),
            np.zeros(dim),
            np.full(dim, 1.0),
            -np.ones(dim),
            np.linspace(-4, 4, dim),
            np.full(dim, 2.0),
            np.full(dim, -1.0),
            np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
        ]

        for cand in candidates:
            if calls_used >= budget or timed():
                break
            evaluate(cand)

        if best_x is None:
            return np.zeros(dim), np.inf

        # --- Phase 2: Broad random sampling ---
        n_initial = 80
        for _ in range(n_initial):
            if calls_used >= budget or timed():
                break
            x_rand = rng.uniform(-5.0, 5.0, dim)
            evaluate(x_rand)

        if best_x is None:
            return np.zeros(dim), np.inf

        # --- Phase 3: Multi-start local search with adaptive step ---
        # Create diverse starting points
        num_starts = 4
        starts = []
        starts.append(best_x.copy())
        for _ in range(num_starts - 1):
            if rng.random() < 0.5:
                start = rng.uniform(-5, 5, dim)
            else:
                start = best_x + rng.standard_normal(dim) * 1.5
            starts.append(clip_box(start))

        for start in starts:
            if calls_used >= budget or timed():
                break

            x = start.copy()
            f_x = evaluate(x)
            if not np.isfinite(f_x):
                continue

            sigma = 1.0
            no_improve = 0
            local_budget = max(20, (budget - calls_used) // (num_starts + 1))

            for _ in range(local_budget):
                if calls_used >= budget or timed():
                    break

                # Generate trial in random direction
                direction = rng.standard_normal(dim)
                direction /= (np.linalg.norm(direction) + 1e-12)
                y = clip_box(x + sigma * direction)
                f_y = evaluate(y)

                if np.isfinite(f_y) and f_y < f_x:
                    x, f_x = y, f_y
                    sigma = min(sigma * 1.1, 3.0)
                    no_improve = 0
                else:
                    sigma *= 0.75
                    no_improve += 1
                    if no_improve > 12:
                        # Restart from best with fresh sigma
                        x = best_x + rng.standard_normal(dim) * 0.5
                        x = clip_box(x)
                        f_x = evaluate(x)
                        no_improve = 0
                        sigma = 0.8

        # --- Phase 4: Coordinate descent refinement ---
        if best_x is not None and calls_used < budget:
            x = best_x.copy()
            f_x = best_f

            for iteration in range(15):
                if calls_used >= budget or timed():
                    break

                improved = False
                for d in range(dim):
                    if calls_used >= budget or timed():
                        break

                    step = 0.3
                    for trial in range(8):
                        if calls_used >= budget or timed():
                            break
                        for sign in [1, -1]:
                            if calls_used >= budget or timed():
                                break
                            trial_x = x.copy()
                            trial_x[d] += sign * step
                            f_trial = evaluate(trial_x)
                            if np.isfinite(f_trial) and f_trial < f_x:
                                x = trial_x
                                f_x = f_trial
                                improved = True
                        step *= 0.65

                if not improved:
                    break

        # Final clamp
        if best_x is not None:
            best_x = clip_box(best_x)
        else:
            best_x = np.zeros(dim)

        return best_x, best_f
# EVOLVE-BLOCK-END
