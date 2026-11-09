"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart (1+1)-ES with adaptive sigma for rotated Rastrigin."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        deadline = time.time() + 280  # Safety margin before 300s

        best_x = None
        best_f = np.inf
        call_count = 0

        def safe_func(x):
            nonlocal call_count
            if call_count >= budget:
                return np.inf
            call_count += 1
            try:
                val = func(np.asarray(x, dtype=float).reshape(dim))
                if np.isfinite(val):
                    return float(val)
                return 1e10
            except Exception:
                return 1e10

        def run_es(start_x, max_calls, initial_sigma=1.5):
            nonlocal best_x, best_f
            x = np.clip(start_x, -5, 5).copy()
            fx = safe_func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            sigma = initial_sigma
            used = 1  # Count the initial evaluation
            stagnation = 0
            max_stagnation = 40

            while used < max_calls and call_count < budget and time.time() < deadline:
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = safe_func(y)
                used += 1

                if fy < fx:
                    x, fx = y, fy
                    sigma = min(sigma * 1.5, 5.0)
                    stagnation = 0
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.87
                    stagnation += 1
                    if stagnation >= max_stagnation:
                        break

            return x, fx

        # Phase 1: Start from origin (often good for centered landscapes)
        if call_count < budget and time.time() < deadline:
            start1 = np.zeros(dim)
            budget1 = int(budget * 0.35)
            run_es(start1, budget1, initial_sigma=2.0)

        # Phase 2: Start from a random point in the central region
        if call_count < budget and time.time() < deadline:
            start2 = rng.uniform(-4, 4, dim)
            budget2 = int(budget * 0.35)
            run_es(start2, budget2, initial_sigma=2.0)

        # Phase 3: Polish the best solution found so far with smaller sigma
        if call_count < budget and time.time() < deadline:
            if best_x is not None:
                start3 = best_x.copy()
            else:
                start3 = rng.uniform(-5, 5, dim)
            budget3 = budget - call_count
            run_es(start3, budget3, initial_sigma=0.5)

        return best_x, best_f
# EVOLVE-BLOCK-END
