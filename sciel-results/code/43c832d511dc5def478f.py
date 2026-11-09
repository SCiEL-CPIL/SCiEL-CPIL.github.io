"""Optimizer for BBOB-style f6 (Attractive sector) with deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Global coarse-to-fine search + adaptive (1+1)-ES refiner with deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t0 = time.time()
        DEADLINE = 300.0 - 15.0  # safety margin before 330s kill
        rng = np.random.default_rng(0)

        def remaining():
            return budget - self.calls

        self.calls = 0
        self.best = None
        self.best_f = np.inf

        def record(x, f):
            self.calls += 1
            if f < self.best_f:
                self.best_f = f
                self.best = x.copy()

        # ---- Phase 1: coarse-to-fine deterministic grid + LHS for global basin ----
        # Coarse grid: 3^5 = 243 points
        coarse = 3
        pts = np.array(np.meshgrid(*[np.linspace(-5, 5, coarse) for _ in range(dim)]),
                       dtype=float).reshape(-1, dim)
        for x in pts:
            if self.calls >= budget:
                break
            f = func(x)
            record(x, f)

        # Fine grid around the coarse best: 2 levels of 5^5 = 3125 (capped by budget)
        if self.best is not None:
            for span in (2.0, 0.5):
                c = 5
                center = self.best
                lo = np.clip(center - span, -5, 5)
                hi = np.clip(center + span, -5, 5)
                if np.any(hi <= lo):
                    continue
                pts = np.array(np.meshgrid(*[np.linspace(lo[i], hi[i], c) for i in range(dim)]),
                               dtype=float).reshape(-1, dim)
                for x in pts:
                    if self.calls >= budget or time.time() > DEADLINE:
                        break
                    f = func(x)
                    record(x, f)

        # A few LHS exploratory points to hedge against a missed basin
        lhs_n = min(20, max(0, budget - self.calls))
        if lhs_n > 0:
            # simple Latin hypercube
            grid = np.array([np.linspace(0, 1, lhs_n) for _ in range(dim)])
            perm = [rng.permutation(lhs_n) for _ in range(dim)]
            lhs = (grid.T[perm] * 10.0 - 5.0)  # map to [-5,5]
            for x in lhs:
                if self.calls >= budget or time.time() > DEADLINE:
                    break
                f = func(x)
                record(x, f)

        # ---- Phase 2: adaptive (1+1)-ES refiner from the global best ----
        if self.best is None:
            self.best = np.zeros(dim)
            self.best_f = func(self.best)
            self.calls += 1

        x = self.best.copy()
        fx = self.best_f
        sigma = 1.0
        last_improve = 0
        stuck = 0

        while self.calls < budget and time.time() < DEADLINE:
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            record(y, fy)
            if fy < fx - 1e-12:
                x, fx = y.copy(), fy
                sigma *= 1.5
                last_improve = self.calls
                stuck = 0
            else:
                sigma *= 0.87
                # stagnation jump: if no improvement for a while, take a big random step
                if self.calls - last_improve > 40:
                    stuck += 1
                    if stuck > 5:
                        # jump toward a fresh random point but keep box
                        jump = np.clip(x + 3.0 * rng.standard_normal(dim), -5, 5)
                        fj = func(jump)
                        record(jump, fj)
                        if fj < fx:
                            x, fx = jump.copy(), fj
                        stuck = 0
                        last_improve = self.calls
                        sigma = 2.0

        return self.best, self.best_f
# EVOLVE-BLOCK-END
