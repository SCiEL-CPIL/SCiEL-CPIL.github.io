"""Rewritten optimizer for BBOB-style f15 (Rastrigin rotated)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart (1+1)-ES with adaptive step size and time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_used = 0

        # Multi-restart (1+1)-ES
        # Strategy: allocate budget across multiple restarts.
        # Each restart uses a portion of the budget. If a restart stagnates
        # (no improvement for many steps), we restart from a new random point.
        
        # We'll use a simple approach: try several restarts, each with a sub-budget.
        # The number of restarts and sub-budget can be adaptive.
        
        # For Rastrigin, the function has many local minima. A single (1+1)-ES
        # might get stuck. Multi-restart helps escape.
        
        # Let's do: up to 10 restarts, each with budget/10 evaluations,
        # but with early termination if no improvement for 20% of sub-budget.
        
        max_restarts = 10
        sub_budget = max(10, budget // max_restarts)
        stagnation_limit = max(5, sub_budget // 5)

        for restart in range(max_restarts):
            # Check time
            if time.time() > deadline:
                break
            if calls_used >= budget:
                break

            # Initialize
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            calls_used += 1
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            sigma = 1.0
            no_improve = 0
            sub_calls = 0
            max_sub_calls = min(sub_budget, budget - calls_used)

            for step in range(max_sub_calls):
                # Check time
                if time.time() > deadline:
                    break
                if calls_used >= budget:
                    break

                # Generate trial
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_used += 1
                sub_calls += 1

                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                    no_improve = 0
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.9
                    no_improve += 1
                    if no_improve >= stagnation_limit:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
