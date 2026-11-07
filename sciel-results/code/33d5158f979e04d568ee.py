"""Optimized black-box continuous optimizer for BBOB-style functions."""
import numpy as np
import time


class Optimizer:
    """
    Hybrid optimizer combining:
    - Initial random sampling to find a good starting point
    - Coordinate descent with adaptive step sizes for exploitation
    - Random restarts with local search for exploration
    - Deadline-aware execution to stay within time limits
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 20s safety margin before 300s limit

        # State tracking
        best_x = None
        best_f = np.inf
        calls_made = 0

        def evaluate(x):
            nonlocal best_x, best_f, calls_made
            if calls_made >= self.budget:
                return np.inf
            x = np.clip(x, -5.0, 5.0)
            fx = float(func(x))
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Initial random sampling (use ~10% of budget)
        n_initial = max(5, self.budget // 10)
        rng = np.random.default_rng(42)

        for _ in range(n_initial):
            x = rng.uniform(-5, 5, self.dim)
            evaluate(x)
            if time.time() > deadline:
                break

        if best_x is None:
            best_x = np.zeros(self.dim)
            best_f = evaluate(best_x)

        # Phase 2: Coordinate descent with adaptive step sizes
        # Use remaining budget for CD iterations
        remaining_budget = self.budget - calls_made
        if remaining_budget <= 0 or time.time() > deadline:
            return best_x, best_f

        # CD parameters
        step_sizes = np.ones(self.dim) * 1.0
        min_step = 1e-4
        max_step = 10.0
        shrink_factor = 0.7
        grow_factor = 1.3
        stagnation_limit = 5  # consecutive non-improving steps per coordinate

        # Main CD loop
        x = best_x.copy()
        fx = best_f
        total_stagnation = 0

        while calls_made < self.budget and time.time() < deadline:
            improved = False
            for i in range(self.dim):
                if time.time() > deadline or calls_made >= self.budget:
                    break

                # Try positive and negative steps
                for sign in [1, -1]:
                    x_new = x.copy()
                    x_new[i] += sign * step_sizes[i]
                    x_new = np.clip(x_new, -5.0, 5.0)

                    if np.allclose(x_new, x):
                        continue

                    fx_new = evaluate(x_new)
                    if fx_new < fx:
                        x = x_new
                        fx = fx_new
                        improved = True
                        step_sizes[i] = min(step_sizes[i] * grow_factor, max_step)
                        total_stagnation = 0
                        break
                    else:
                        step_sizes[i] = max(step_sizes[i] * shrink_factor, min_step)

            if not improved:
                total_stagnation += 1
                # If all coordinates are stuck, try a random perturbation
                if total_stagnation >= self.dim:
                    # Random jump
                    jump_size = 2.0
                    x = np.clip(x + rng.normal(0, jump_size, self.dim), -5, 5)
                    fx = evaluate(x)
                    step_sizes = np.ones(self.dim) * 1.0
                    total_stagnation = 0

        return best_x, best_f
