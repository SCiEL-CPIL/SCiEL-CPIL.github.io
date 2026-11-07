"""Search/portfolio optimizer for BBOB f4 (Büche-Rastrigin)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Portfolio: run multiple diverse (1+1)-ES variants from varied starts,
    always tracking the global best. Explicit wall-clock deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 295.0  # safety margin vs 300s wall-clock

        def remaining():
            return self.budget > 0 and time.time() < deadline

        # Pre-compute a deterministic set of candidate starting points:
        # origin, grid corners, and a few random points.
        rng = np.random.default_rng(12345)
        starts = []
        starts.append(np.zeros(self.dim))
        for axis in range(self.dim):
            v1 = np.zeros(self.dim); v1[axis] = -3.0
            v2 = np.zeros(self.dim); v2[axis] = 3.0
            starts.append(v1); starts.append(v2)
        # Random starts
        for _ in range(10):
            starts.append(rng.uniform(-4, 4, self.dim))

        best_x = None
        best_f = np.inf

        def evaluate(x):
            nonlocal best_x, best_f
            x = np.clip(x, -5.0, 5.0)
            f = float(func(x))
            self.budget -= 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: cheap multi-start scan to find a good basin.
        for s in starts:
            if not remaining():
                break
            evaluate(s)

        # Phase 2: run several (1+1)-ES runs with different sigma schedules
        # and step sizes, each restarting from the current best or a random point.
        strategies = [
            # (sigma_init, sigma_up, sigma_down)
            (2.0, 1.5, 0.87),
            (1.0, 1.4, 0.85),
            (3.0, 1.6, 0.80),
            (0.5, 1.3, 0.90),
        ]

        for strat_i, (sig0, up, down) in enumerate(strategies):
            if not remaining():
                break
            # Start from best if available, else random
            if best_x is not None and strat_i % 2 == 0:
                x = best_x.copy()
                fx = best_f
            else:
                x = np.clip(rng.uniform(-4, 4, self.dim), -5, 5)
                fx = evaluate(x)
                if not remaining():
                    break
            sigma = sig0
            # Run until budget/time exhausted
            while remaining():
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = evaluate(y)
                if not remaining():
                    break
                if fy < fx:
                    x, fx = y, fy
                    sigma *= up
                else:
                    sigma *= down

        # Phase 3: local fine-tuning from the global best with small sigma.
        if best_x is not None and remaining():
            x = best_x.copy()
            fx = best_f
            sigma = 0.05
            while remaining():
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = evaluate(y)
                if not remaining():
                    break
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.1
                else:
                    sigma *= 0.85
                    if sigma < 1e-6:
                        break

        # Phase 4: final high-precision polish pass with even smaller sigma.
        if best_x is not None and remaining():
            x = best_x.copy()
            fx = best_f
            sigma = 0.01
            while remaining():
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = evaluate(y)
                if not remaining():
                    break
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.05
                else:
                    sigma *= 0.95
                    if sigma < 1e-8:
                        break

        if best_x is None:
            best_x = np.zeros(self.dim)
        return best_x, best_f
# EVOLVE-BLOCK-END
