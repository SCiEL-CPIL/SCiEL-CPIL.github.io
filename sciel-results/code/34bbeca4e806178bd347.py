"""Black-box continuous optimizer on BBOB f11 (Discus)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Global map via random restarts + L-BFGS-B local refinement, with deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 300.0  # 300s wall-clock limit; keep margin

        rng = np.random.default_rng(42)
        best_x = None
        best_f = np.inf
        calls_used = 0

        # We have budget=500 evaluations. We'll use a mix of global exploration and local refinement.
        # Strategy:
        # 1. Do a set of random initial points (global exploration)
        # 2. For each promising point, run L-BFGS-B with a limited number of function calls
        # 3. Keep track of best found
        # 4. Use remaining budget for more random starts or refinement

        # Phase 1: Random initial points to find good basins
        # Use about 30% of budget for random exploration
        n_random = 50  # 50 random points
        for i in range(n_random):
            if calls_used >= self.budget:
                break
            if time.time() > deadline - 10:
                break

            x = rng.uniform(-5, 5, self.dim)
            fx = func(x)
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: Local refinement from best points using L-BFGS-B
        # We'll do refinement from the best point found, and possibly a few others
        # Each L-BFGS-B call uses multiple function evaluations
        # We need to be careful about the budget

        # Collect candidate starting points: best found + a few random ones
        candidates = [best_x.copy()] if best_x is not None else []

        # Add a few more random candidates for diversity
        for i in range(5):
            if calls_used >= self.budget:
                break
            x = rng.uniform(-5, 5, self.dim)
            candidates.append(x)

        # For each candidate, run L-BFGS-B with limited budget
        # Total remaining budget for local refinement
        remaining_budget = self.budget - calls_used

        # Allocate budget across candidates
        n_refine = min(len(candidates), 10)
        budget_per_refine = max(1, remaining_budget // n_refine)

        for i in range(n_refine):
            if calls_used >= self.budget:
                break
            if time.time() > deadline - 20:
                break

            x0 = candidates[i]
            fx0 = func(x0)
            calls_used += 1
            if fx0 < best_f:
                best_f = fx0
                best_x = x0.copy()

            # Run L-BFGS-B from this point
            # Use a wrapper to track calls
            call_count = [0]

            def wrapped_func(x):
                nonlocal calls_used
                if calls_used >= self.budget:
                    raise Exception("Budget exceeded")
                if time.time() > deadline - 5:
                    raise Exception("Time exceeded")
                x_clipped = np.clip(x, -5, 5)
                f_val = func(x_clipped)
                calls_used += 1
                call_count[0] += 1
                return f_val

            try:
                res = minimize(
                    wrapped_func,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * self.dim,
                    options={
                        'maxiter': budget_per_refine,
                        'maxfun': budget_per_refine,
                        'gtol': 1e-8,
                    }
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5, 5).copy()
            except Exception:
                pass

        # Phase 3: If we still have budget, do more random exploration + refinement
        if calls_used < self.budget:
            n_more_random = min(30, self.budget - calls_used)
            for i in range(n_more_random):
                if calls_used >= self.budget:
                    break
                if time.time() > deadline - 10:
                    break

                x = rng.uniform(-5, 5, self.dim)
                fx = func(x)
                calls_used += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

            # Final refinement from the best point
            if best_x is not None and calls_used < self.budget:
                remaining = self.budget - calls_used
                if remaining > 5:
                    call_count = [0]

                    def wrapped_func2(x):
                        nonlocal calls_used
                        if calls_used >= self.budget:
                            raise Exception("Budget exceeded")
                        if time.time() > deadline - 5:
                            raise Exception("Time exceeded")
                        x_clipped = np.clip(x, -5, 5)
                        f_val = func(x_clipped)
                        calls_used += 1
                        call_count[0] += 1
                        return f_val

                    try:
                        res = minimize(
                            wrapped_func2,
                            best_x,
                            method='L-BFGS-B',
                            bounds=[(-5, 5)] * self.dim,
                            options={
                                'maxiter': remaining,
                                'maxfun': remaining,
                                'gtol': 1e-10,
                            }
                        )
                        if res.fun < best_f:
                            best_f = res.fun
                            best_x = np.clip(res.x, -5, 5).copy()
                    except Exception:
                        pass

        if best_x is None:
            best_x = np.zeros(self.dim)
            best_f = func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
