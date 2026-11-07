"""Deterministic Hooke-Jeeves pattern search for BBOB f6 (Attractive Sector)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hooke-Jeeves pattern search with structured initialization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 270  # safety margin
        dim = self.dim
        budget = self.budget
        calls = 0

        def check_deadline():
            return time.time() >= deadline

        # Phase 1: Structured initialization to find a good starting point
        # Evaluate at origin and along axes
        candidates = []
        
        # Origin
        x0 = np.zeros(dim)
        f0 = func(x0)
        candidates.append((x0, f0))
        calls += 1

        # Along each axis, both directions
        probe_len = 3.0
        for i in range(dim):
            for sign in [1, -1]:
                if check_deadline() or calls >= budget:
                    break
                x = np.zeros(dim)
                x[i] = sign * probe_len
                fx = func(x)
                candidates.append((x, fx))
                calls += 1

        # A few random points for diversity
        rng = np.random.default_rng(123)
        for _ in range(10):
            if check_deadline() or calls >= budget:
                break
            x = rng.uniform(-4.5, 4.5, dim)
            fx = func(x)
            candidates.append((x, fx))
            calls += 1

        # Find the best starting point
        candidates.sort(key=lambda t: t[1])
        x = candidates[0][0].copy()
        f = candidates[0][1]

        # Phase 2: Hooke-Jeeves pattern search
        # Initial step size
        step = 1.0
        min_step = 1e-4
        max_step = 5.0
        pattern_dir = None  # Last successful direction for pattern move

        while not check_deadline() and calls < budget:
            improved = False
            last_successful_dir = None

            # Exploratory moves along each coordinate
            for i in range(dim):
                if check_deadline() or calls >= budget:
                    break

                # Try positive direction
                x_trial = x.copy()
                x_trial[i] += step
                x_trial = np.clip(x_trial, -5.0, 5.0)
                f_trial = func(x_trial)
                calls += 1

                if f_trial < f:
                    x = x_trial
                    f = f_trial
                    last_successful_dir = np.zeros(dim)
                    last_successful_dir[i] = step
                    improved = True
                    continue

                # Try negative direction
                x_trial = x.copy()
                x_trial[i] -= step
                x_trial = np.clip(x_trial, -5.0, 5.0)
                f_trial = func(x_trial)
                calls += 1

                if f_trial < f:
                    x = x_trial
                    f = f_trial
                    last_successful_dir = np.zeros(dim)
                    last_successful_dir[i] = -step
                    improved = True

            # Pattern move: if we made progress, try to accelerate along the last direction
            if last_successful_dir is not None and not check_deadline() and calls < budget:
                # Try a pattern step along the accumulated direction
                # Use the direction from the last successful move, scaled
                pattern_step = last_successful_dir.copy()
                # Try multiple pattern steps to accelerate
                for _ in range(3):
                    if check_deadline() or calls >= budget:
                        break
                    x_trial = x + pattern_step
                    x_trial = np.clip(x_trial, -5.0, 5.0)
                    f_trial = func(x_trial)
                    calls += 1

                    if f_trial < f:
                        x = x_trial
                        f = f_trial
                        # Keep going in this direction
                        continue
                    else:
                        break

            if not improved:
                # No improvement: reduce step size
                step *= 0.5
                if step < min_step:
                    break
            else:
                # If improved, we can keep the step or slightly increase it
                # But be conservative
                step = min(step * 1.1, max_step)

        return x, f
# EVOLVE-BLOCK-END
