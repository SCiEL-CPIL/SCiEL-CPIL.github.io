"""Rewritten optimizer: multi-start adaptive ES with deadline guard."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Multi-start adaptive (1+1)-ES with 1/5 rule and a hard wall-clock deadline."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t0 = time.time()
        deadline = t0 + 295.0  # 5 s safety margin before the 300 s eval limit

        def elapsed_ok():
            return time.time() < deadline

        # ---------- Phase 1: coarse random sampling to locate the basin ----------
        # Spend a fraction of the budget on independent random starts.
        n_starts = max(8, budget // 60)
        n_starts = min(n_starts, 40)
        per_start = max(1, (budget - n_starts) // n_starts)

        best_x = None
        best_f = np.inf

        rng = np.random.default_rng(12345)

        def clip(v):
            return np.clip(v, -5.0, 5.0)

        # Random candidates spread over the box (the minimizer is in [-4,4]^5).
        candidates = rng.uniform(-4.5, 4.5, size=(n_starts, dim))
        # Also include a few points near the origin and corners for robustness.
        extra = np.zeros((min(4, dim + 2), dim))
        candidates = np.vstack([candidates, extra])
        n_starts = len(candidates)
        per_start = max(1, (budget - n_starts) // n_starts)

        evaluated = 0
        start_xs = []
        start_fs = []
        for i in range(n_starts):
            if not elapsed_ok() or evaluated >= budget:
                break
            x = clip(candidates[i])
            f = func(x)
            evaluated += 1
            start_xs.append(x)
            start_fs.append(f)
            if f < best_f:
                best_f, best_x = f, x.copy()

        # ---------- Phase 2: adaptive (1+1)-ES from the best starts ----------
        # Sort starts by fitness and run a short ES from the top ones.
        order = np.argsort(start_fs)
        top_k = min(len(order), max(1, n_starts // 2))
        top_indices = order[:top_k]

        # Per-start budget, leaving room for a final polish.
        es_budget_each = max(1, (budget - evaluated - dim) // max(1, top_k))

        for idx in top_indices:
            if not elapsed_ok() or evaluated >= budget:
                break
            x = start_xs[idx].copy()
            fx = start_fs[idx]
            sigma = 1.0
            successes = 0
            window = 0
            WIN = 5

            for _ in range(es_budget_each):
                if not elapsed_ok() or evaluated >= budget:
                    break
                y = clip(x + sigma * rng.standard_normal(dim))
                fy = func(y)
                evaluated += 1
                window += 1
                if fy < fx:
                    x, fx = y, fy
                    successes += 1
                    if fx < best_f:
                        best_f, best_x = fx, x.copy()
                else:
                    # no improvement
                    pass
                if window >= WIN:
                    # 1/5 success rule
                    if successes / window > 1.0 / 5.0:
                        sigma *= 1.3
                    else:
                        sigma *= 0.7
                    sigma = float(np.clip(sigma, 1e-4, 5.0))
                    successes = 0
                    window = 0

        # ---------- Phase 3: final fine polish from the overall best ----------
        remaining = budget - evaluated
        if remaining > 0 and best_x is not None and elapsed_ok():
            x = best_x.copy()
            fx = best_f
            sigma = 0.05
            successes = 0
            window = 0
            WIN = 5
            for _ in range(remaining):
                if not elapsed_ok() or evaluated >= budget:
                    break
                y = clip(x + sigma * rng.standard_normal(dim))
                fy = func(y)
                evaluated += 1
                window += 1
                if fy < fx:
                    x, fx = y, fy
                    successes += 1
                    if fx < best_f:
                        best_f, best_x = fx, x.copy()
                if window >= WIN:
                    if successes / window > 1.0 / 5.0:
                        sigma *= 1.3
                    else:
                        sigma *= 0.7
                    sigma = float(np.clip(sigma, 1e-6, 5.0))
                    successes = 0
                    window = 0

        return best_x, best_f
# EVOLVE-BLOCK-END
