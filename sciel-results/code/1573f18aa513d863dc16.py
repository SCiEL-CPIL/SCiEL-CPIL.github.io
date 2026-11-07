"""Optimizer for BBOB f4 (Büche-Rastrigin) on [-5,5]^5."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Multi-restart (1+1)-ES with adaptive step size and stagnation-based restarts."""

    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)
        t0 = time.time()
        deadline = t0 + 285.0

        calls = 0
        best_x = None
        best_f = float('inf')

        def clip(x):
            return np.clip(x, -5.0, 5.0)

        def check_deadline():
            nonlocal calls
            return calls >= budget or time.time() > deadline

        # --- Phase 1: Coarse exploration with LHS-like sampling ---
        # Use stratified sampling to cover the space well
        n_phase1 = 80
        # Create stratified samples: divide each dimension into chunks
        n_strata = 4
        stride = 10.0 / n_strata
        # Generate stratified points
        phase1_points = []
        for i in range(n_phase1):
            # Use a quasi-random approach: golden ratio based
            x = np.zeros(d)
            for j in range(d):
                # Interleave different dimensions with different offsets
                frac = (i * (j + 1) + 1) / n_phase1
                # Add stratification
                stratum = int(frac * n_strata)
                within = frac * n_strata - stratum
                x[j] = -5.0 + (stratum + within) * stride
            phase1_points.append(x)

        # Also add some random points for diversity
        for _ in range(n_phase1 // 2):
            phase1_points.append(rng.uniform(-5.0, 5.0, d))

        # Evaluate phase 1 points
        candidates = []
        for x in phase1_points:
            if check_deadline():
                break
            x = clip(x)
            fx = float(func(x))
            calls += 1
            candidates.append((fx, x.copy()))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Sort by fitness
        candidates.sort(key=lambda item: item[0])

        # --- Phase 2: Multi-restart (1+1)-ES from top candidates ---
        remaining = budget - calls
        if remaining <= 0:
            return best_x, best_f

        # Select top K starting points
        n_restarts = min(5, len(candidates))
        budget_per_restart = remaining // n_restarts

        for k in range(n_restarts):
            if check_deadline():
                break

            start_x = candidates[k][1].copy()
            start_f = candidates[k][0]

            # (1+1)-ES parameters
            sigma = 1.5
            x = start_x.copy()
            fx = start_f

            # Track progress for stagnation detection
            no_improve_count = 0
            max_no_improve = 30

            for step in range(budget_per_restart):
                if check_deadline():
                    break

                # Generate candidate with adaptive sigma
                y = clip(x + sigma * rng.standard_normal(d))
                fy = float(func(y))
                calls += 1

                if fy < fx:
                    # Success: move and increase sigma
                    x = y.copy()
                    fx = fy
                    sigma *= 1.15
                    no_improve_count = 0

                    # Update global best
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    # Failure: decrease sigma
                    sigma *= 0.9
                    no_improve_count += 1

                # Clamp sigma
                sigma = np.clip(sigma, 1e-5, 5.0)

                # Stagnation-based restart: if stuck for too long, break early
                if no_improve_count >= max_no_improve:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
