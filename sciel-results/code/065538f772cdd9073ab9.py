"""Rewritten optimizer for BBOB f24 (Lunacek bi-Rastrigin)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Fast adaptive (1+1)-ES with periodic coarse scans and restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start = time.time()
        deadline = start + 295.0  # safety margin before 300s limit

        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)

        # Counters and best
        calls = 0
        best_x = None
        best_f = np.inf

        def record(x, f):
            nonlocal calls, best_x, best_f
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Helper: evaluate a point safely
        def eval_pt(x):
            x = np.clip(x, -5.0, 5.0)
            return float(func(x))

        # ---- Phase 1: Coarse random + grid scan to find promising basins ----
        # Use a portion of the budget for exploration
        # Lunacek f24 has two basins; a few well-chosen points help
        # Try origin and some structured points
        candidates = []
        # Origin
        candidates.append(np.zeros(n))
        # Some structured points in [-4,4]^5
        for _ in range(30):
            candidates.append(rng.uniform(-4.5, 4.5, n))
        # Axis-aligned points
        for i in range(n):
            for s in [-3.0, 3.0, -1.5, 1.5]:
                p = np.zeros(n)
                p[i] = s
                candidates.append(p)
        # Diagonal points
        for s in [-2.0, 2.0, -1.0, 1.0]:
            p = np.full(n, s)
            candidates.append(p)

        # Evaluate all candidates
        for x in candidates:
            if calls >= budget:
                break
            fx = eval_pt(x)
            record(x, fx)

        # Check time
        if time.time() > deadline:
            return best_x, best_f

        # ---- Phase 2: Local (1+1)-ES from best point ----
        x = best_x.copy()
        fx = best_f
        sigma = 1.5
        no_improve = 0
        max_no_improve = 30

        while calls < budget:
            if time.time() > deadline:
                break

            y = x + sigma * rng.standard_normal(n)
            y = np.clip(y, -5.0, 5.0)
            fy = eval_pt(y)
            record(y, fy)

            if fy < fx:
                x = y
                fx = fy
                sigma *= 1.2
                no_improve = 0
            else:
                sigma *= 0.8
                no_improve += 1

            # Restart if stuck
            if no_improve > max_no_improve:
                x = best_x.copy()
                fx = best_f
                sigma = 1.5
                no_improve = 0

                # Try a few more random restarts
                for _ in range(5):
                    if calls >= budget or time.time() > deadline:
                        break
                    xr = rng.uniform(-4.5, 4.5, n)
                    fr = eval_pt(xr)
                    record(xr, fr)
                    if fr < fx:
                        x = xr
                        fx = fr
                        break

                max_no_improve = 25  # Tighten after restart

            # Occasionally do a small coarse scan to find other basins
            if no_improve > 10 and calls < budget * 0.8:
                for _ in range(3):
                    if calls >= budget or time.time() > deadline:
                        break
                    xr = rng.uniform(-4.5, 4.5, n)
                    fr = eval_pt(xr)
                    record(xr, fr)
                    if fr < fx:
                        x = xr
                        fx = fr
                        sigma = 1.5
                        no_improve = 0
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
