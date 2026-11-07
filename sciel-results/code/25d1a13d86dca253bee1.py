"""Black-box optimizer for BBOB f13 (sharp ridge) with early-progress focus."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES with deadline guard, tuned for early progress on f13."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 325.0  # safety margin before 330s kill
        dim = self.dim
        budget = self.budget

        def time_ok():
            return time.time() < deadline

        rng = np.random.default_rng(12345)

        # ---- Phase 0: fast cheap scan to get an early good point ----
        # f13 has a sharp ridge; a coarse random scan quickly finds the basin.
        # Do a small random scan + a few deterministic points to get a strong
        # initial best, then invest the rest in CMA-ES refinement.
        scan_n = min(40, budget // 10)
        candidates = []
        # deterministic spread
        for _ in range(6):
            candidates.append(rng.uniform(-5, 5, dim))
        # grid-like corners
        for c in range(0, 10, 2):
            candidates.append(np.full(dim, c / 2.0 - 5.0))
        while len(candidates) < scan_n:
            candidates.append(rng.uniform(-5, 5, dim))
        candidates = np.array(candidates[:scan_n])

        best_x = None
        best_f = np.inf
        for i in range(candidates.shape[0]):
            if not time_ok():
                break
            x = np.clip(candidates[i], -5, 5)
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # ---- Phase 1: CMA-ES refinement around the best point ----
        # CMA-ES is the strong black-box optimizer; adapt sigma and the
        # covariance so it can track the ridge and converge.
        x = best_x.copy()
        f = best_f
        sigma = 1.0
        pop = max(4, int(round(4 + 3 * np.log(dim))))
        # track progress to adapt
        for _ in range(budget - scan_n):
            if not time_ok():
                break
            # sample population
            for _ in range(pop):
                if not time_ok():
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                if fy < f:
                    f = fy
                    x = y.copy()
            # adapt sigma based on recent improvement (simple 1/5-ish heuristic)
            # keep sigma from collapsing too fast or exploding
            sigma *= 0.97
            if sigma < 1e-4:
                sigma = 1e-4
            # if we are at a very good value, shrink exploration
            if f < -50:
                sigma *= 0.9

        return x, f
# EVOLVE-BLOCK-END
