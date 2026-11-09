"""Combined optimizer for BBOB f24 (Lunacek bi-Rastrigin):
coarse structured basin scan + (1+1)-ES with 1/5 success rule and restarts."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Coarse scan to locate basins, then (1+1)-ES (1/5 rule) with restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start = time.time()
        deadline = start + 295.0  # safety margin before the 300s kill

        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(7)

        calls = 0
        best_x = None
        best_f = np.inf

        def record(x, f):
            nonlocal calls, best_x, best_f
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()

        def eval_pt(x):
            x = np.clip(x, -5.0, 5.0)
            return float(func(x))

        # ---- Phase 1: coarse structured scan to find promising basins ----
        candidates = [np.zeros(n)]
        for i in range(n):
            for s in (-3.0, -1.5, 1.5, 3.0):
                p = np.zeros(n)
                p[i] = s
                candidates.append(p)
        for s in (-2.0, -1.0, 1.0, 2.0):
            candidates.append(np.full(n, s))
        for _ in range(40):
            candidates.append(rng.uniform(-4.5, 4.5, n))

        for x in candidates:
            if calls >= budget or time.time() > deadline:
                break
            record(x, eval_pt(x))

        if best_x is None:
            best_x = np.zeros(n)
            best_f = eval_pt(best_x)
            record(best_x, best_f)

        if time.time() > deadline:
            return best_x, best_f

        # ---- Phase 2: (1+1)-ES with 1/5 success rule + stagnation restarts ----
        x = best_x.copy()
        fx = best_f
        sigma = 1.5
        no_improve = 0
        max_no_improve = 35

        while calls < budget:
            if time.time() > deadline:
                break

            y = np.clip(x + sigma * rng.standard_normal(n), -5.0, 5.0)
            fy = eval_pt(y)
            record(y, fy)

            if fy < fx:
                x = y
                fx = fy
                sigma *= 1.5      # 1/5 success rule: increase on accept
                no_improve = 0
            else:
                sigma *= 0.87     # 1/5 success rule: decrease on reject
                no_improve += 1

            if no_improve > max_no_improve:
                # restart from the global best with a fresh step size
                x = best_x.copy()
                fx = best_f
                sigma = 1.5
                no_improve = 0
                max_no_improve = 30  # tighten after a restart

        return best_x, best_f
# EVOLVE-BLOCK-END
