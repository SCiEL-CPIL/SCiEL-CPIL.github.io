"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-phase optimizer: coarse global search, local CMA-ES-like refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 285.0  # safety margin before 330s kill
        rng = np.random.default_rng(123)

        # State
        best_x = None
        best_f = float('inf')
        calls = 0

        def eval_func(x):
            nonlocal calls, best_x, best_f
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Coarse global search with (1+1)-ES style exploration
        # Use a few restarts to cover the landscape
        # Budget allocation: ~40% for global, ~60% for local refinement
        phase1_budget = int(budget * 0.4)
        phase2_budget = budget - phase1_budget

        # Phase 1: Multiple short (1+1)-ES runs from random starts
        num_starts = 5
        per_start_budget = phase1_budget // num_starts

        for s in range(num_starts):
            if calls >= phase1_budget or time.time() > deadline - 5.0:
                break
            x = rng.uniform(-5, 5, dim)
            fx = eval_func(x)
            sigma = 1.5
            local_best_x = x.copy()
            local_best_f = fx
            remaining = per_start_budget - 1
            for _ in range(remaining):
                if calls >= phase1_budget or time.time() > deadline - 5.0:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = eval_func(y)
                if fy < fx:
                    x = y
                    fx = fy
                    sigma *= 1.2
                    if fx < local_best_f:
                        local_best_f = fx
                        local_best_x = x.copy()
                else:
                    sigma *= 0.85
            # Update global best
            if local_best_f < best_f:
                best_f = local_best_f
                best_x = local_best_x.copy()

        # Phase 2: Local refinement around best point
        if best_x is not None and calls < budget and time.time() < deadline - 2.0:
            # Use a simpler local search with decreasing step size
            x = best_x.copy()
            fx = best_f
            sigma = 0.5
            remaining_calls = budget - calls
            for _ in range(remaining_calls):
                if time.time() > deadline - 1.0:
                    break
                # Try multiple directions
                improved = False
                for _attempt in range(3):
                    y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                    fy = eval_func(y)
                    if fy < fx:
                        x = y
                        fx = fy
                        sigma *= 1.1
                        improved = True
                        break
                if not improved:
                    sigma *= 0.7
                    if sigma < 1e-6:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
