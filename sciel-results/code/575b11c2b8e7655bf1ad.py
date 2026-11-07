"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Hybrid optimizer: (1+1)-ES with adaptive step size, multiple restarts,
    and a time guard to stay within the 300s wall-clock limit.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 270.0  # Safety margin before 300s
        n = self.dim
        total_budget = self.budget

        best_x = None
        best_f = np.inf

        # We'll do multiple restarts. Each restart gets a portion of the budget.
        # Early progress is heavily rewarded by AOCC, so we want to find
        # good solutions quickly. We'll use a (1+1)-ES with aggressive
        # step size adaptation and restarts.

        # Strategy: 
        # 1. Start with a few random samples to get an initial idea.
        # 2. Use (1+1)-ES with sigma adaptation.
        # 3. If stagnation detected (no improvement for K steps), restart
        #    from the best point with a different initial sigma or from a
        #    new random point.

        rng = np.random.default_rng(12345)

        # Phase 1: Initial exploration - sample a few points
        n_init = min(10, total_budget)
        for i in range(n_init):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, n)
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(n)
            best_f = func(best_x)

        # Phase 2: (1+1)-ES with restarts
        calls_made = n_init
        remaining = total_budget - calls_made

        # We'll do up to 5 restarts, each with a portion of remaining budget
        max_restarts = 5
        budget_per_restart = max(1, remaining // max_restarts)

        for restart in range(max_restarts):
            if calls_made >= total_budget or time.time() > deadline:
                break

            # Initialize from best point with some perturbation
            if restart == 0:
                x = best_x.copy()
                sigma = 1.0
            else:
                # Restart from a new random point near the best
                x = best_x + rng.normal(0, 0.5, n)
                x = np.clip(x, -5, 5)
                sigma = 0.5 + 0.5 * rng.random()

            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            # Run (1+1)-ES for budget_per_restart steps
            stagnation = 0
            max_stagnation = 30

            for step in range(budget_per_restart):
                if calls_made >= total_budget or time.time() > deadline:
                    break

                y = x + sigma * rng.standard_normal(n)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_made += 1

                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.3
                    stagnation = 0
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.85
                    stagnation += 1

                # If stagnation is high, break early to restart
                if stagnation >= max_stagnation:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
