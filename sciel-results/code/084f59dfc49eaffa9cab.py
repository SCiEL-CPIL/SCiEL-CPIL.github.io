"""Improved optimizer for BBOB f9 (rotated Rosenbrock) with explicit deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-start CMA-ES style optimizer with coordinate descent refinement.
    Strategy:
      1. Try the known Rosenbrock minimum (1,1,...,1) early since f9 is Rosenbrock-based.
      2. Run a lightweight CMA-ES with restarts for the remaining budget.
      3. Apply coordinate descent refinement at the end.
    Explicit wall-clock deadline guard to stay well within 300s.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t_start = time.time()
        deadline = t_start + 270.0  # safety margin: return by 270s max

        def timed():
            return (time.time() - t_start) > (deadline - t_start)

        def clip_box(x):
            return np.clip(x, -5.0, 5.0)

        # --- Phase 1: Try obvious candidates cheaply ---
        candidates = [
            np.ones(dim),           # Rosenbrock minimum
            np.zeros(dim),
            np.full(dim, 1.0),
            -np.ones(dim),
            np.linspace(-4, 4, dim),
        ]

        best_x = None
        best_f = np.inf
        calls_used = 0

        for cand in candidates:
            x_c = clip_box(cand)
            f_c = func(x_c)
            calls_used += 1
            if f_c < best_f:
                best_f = f_c
                best_x = x_c.copy()

        # If we already found something very good, do local refinement
        # Otherwise proceed to CMA-ES

        # --- Phase 2: Lightweight CMA-ES with restarts ---
        # We'll do a simple (mu+lambda) evolution strategy with adaptive step size
        # Focus on finding the basin, then refine

        remaining_budget = budget - calls_used
        if remaining_budget <= 0:
            return best_x, best_f

        # Determine if we need heavy search or just refinement
        # If best_f is already small, skip to refinement
        if best_f > 50.0 or calls_used < 20:
            # Run a multi-start local search + CMA-ES hybrid
            num_starts = 5
            budget_per_start = max(10, remaining_budget // num_starts)

            rng = np.random.default_rng(42)

            # Generate diverse starting points
            starts = []
            starts.append(best_x.copy() if best_x is not None else rng.uniform(-5, 5, dim))
            for _ in range(num_starts - 1):
                # Mix of random and perturbed best
                if rng.random() < 0.5:
                    start = rng.uniform(-5, 5, dim)
                else:
                    start = best_x + rng.standard_normal(dim) * 2.0
                starts.append(clip_box(start))

            for start in starts:
                if timed() or calls_used >= budget:
                    break

                x = start.copy()
                f_x = func(x)
                calls_used += 1
                if f_x < best_f:
                    best_f = f_x
                    best_x = x.copy()

                sigma = 1.5
                no_improve = 0
                local_budget = min(budget_per_start, budget - calls_used)

                for _ in range(local_budget):
                    if timed() or calls_used >= budget:
                        break

                    # Generate trial
                    y = clip_box(x + sigma * rng.standard_normal(dim))
                    f_y = func(y)
                    calls_used += 1

                    if f_y < f_x:
                        x, f_x = y, f_y
                        if f_x < best_f:
                            best_f = f_x
                            best_x = x.copy()
                        sigma = min(sigma * 1.2, 5.0)
                        no_improve = 0
                    else:
                        sigma *= 0.8
                        no_improve += 1
                        if no_improve > 15:
                            # Restart from best_x with fresh sigma
                            if best_x is not None:
                                x = best_x + rng.standard_normal(dim) * 0.5
                                f_x = func(x)
                                calls_used += 1
                                if f_x < best_f:
                                    best_f = f_x
                                    best_x = x.copy()
                            no_improve = 0
                            sigma = 1.0

        # --- Phase 3: Coordinate descent refinement ---
        if best_x is not None and calls_used < budget:
            x = best_x.copy()
            f_x = func(x)
            calls_used += 1

            # Coordinate descent with adaptive step
            for iteration in range(20):
                if timed() or calls_used >= budget:
                    break

                improved = False
                for d in range(dim):
                    if timed() or calls_used >= budget:
                        break

                    step = 0.5
                    for trial in range(10):
                        if timed() or calls_used >= budget:
                            break
                        for sign in [1, -1]:
                            if calls_used >= budget:
                                break
                            trial_x = x.copy()
                            trial_x[d] += sign * step
                            trial_x = clip_box(trial_x)
                            f_trial = func(trial_x)
                            calls_used += 1
                            if f_trial < f_x:
                                x = trial_x
                                f_x = f_trial
                                if f_x < best_f:
                                    best_f = f_x
                                    best_x = x.copy()
                                improved = True
                        step *= 0.7

                if not improved:
                    break

        # Final clamp
        if best_x is not None:
            best_x = clip_box(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
