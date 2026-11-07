"""Improved optimizer for BBOB f19 (Griewank-Rosenbrock)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES style optimizer with adaptive step size and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit

        dim = self.dim
        budget = self.budget

        # Track best solution
        best_x = np.full(dim, 0.0)
        best_f = float('inf')
        call_count = 0

        # Safe function call wrapper
        def safe_call(x):
            nonlocal call_count, best_x, best_f
            if call_count >= budget:
                return best_f
            x = np.clip(x, -5.0, 5.0)
            f = func(x)
            call_count += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Global exploration with random restarts
        # Use 20% of budget for global search
        n_global = max(10, int(0.2 * budget))
        rng = np.random.default_rng(42)

        candidates = []
        for i in range(n_global):
            if call_count >= budget:
                break
            if time.time() > deadline:
                break
            x = rng.uniform(-5.0, 5.0, dim)
            f = safe_call(x)
            candidates.append((f, x.copy()))

        # Sort candidates by fitness
        candidates.sort(key=lambda t: t[0])

        # Phase 2: Local refinement using (1+1)-ES with adaptive sigma
        # Refine top K candidates
        K = min(5, len(candidates))
        remaining_budget = budget - call_count
        budget_per_local = max(10, remaining_budget // K)

        for i in range(K):
            if call_count >= budget:
                break
            if time.time() > deadline:
                break

            x = candidates[i][1].copy()
            fx = candidates[i][0]

            # Adaptive (1+1)-ES
            sigma = 1.0
            # Initial sigma based on distance from origin or fixed
            sigma = 0.5

            for step in range(budget_per_local):
                if call_count >= budget:
                    break
                if time.time() > deadline:
                    break

                # Generate mutant
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5.0, 5.0)

                fy = safe_call(y)

                if fy < fx:
                    x = y.copy()
                    fx = fy
                    sigma *= 1.25
                else:
                    sigma *= 0.9

                # Adaptive sigma based on success rate (simple heuristic)
                # Keep sigma bounded
                sigma = np.clip(sigma, 0.01, 10.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
