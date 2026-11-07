"""Rewritten optimizer: adaptive (1+1)-ES + coordinate-descent exploitation
+ stagnation-triggered random jumps, all under a strict wall-clock deadline."""
import time
import numpy as np

# Safety margin (seconds) before the 330 s kill.
_DEADLINE = 330.0
_MARGIN = 15.0


class Optimizer:
    """Adaptive (1+1)-ES with coordinate-descent exploitation and
    stagnation-triggered random jumps for 5-D BBOB-style functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        budget = self.budget
        dim = self.dim
        deadline = t0 + _DEADLINE - _MARGIN
        rng = np.random.default_rng(12345)

        def _now_ok():
            return time.time() < deadline

        # ---- multi-start to find a good basin ----
        x = np.zeros(dim)
        best = np.inf
        n_starts = 8
        for _ in range(n_starts):
            cand = rng.uniform(-4.0, 4.0, dim)
            val = float(func(cand))
            if val < best:
                best = val
                x = cand.copy()

        sigma = 2.0
        sigma_min = 1e-5
        sigma_max = 20.0
        no_improve = 0
        cd_every = 12          # coordinate-descent cadence
        stagnation_limit = 60  # jumps after this many non-improvements
        calls = 1

        while calls < budget and _now_ok():
            # ---- 1) (1+1)-ES mutation ----
            y = np.clip(x + sigma * rng.standard_normal(dim), -5.0, 5.0)
            fy = float(func(y))
            calls += 1
            if fy < best:
                x, best = y, fy
                sigma = min(sigma * 1.5, sigma_max)
                no_improve = 0
            else:
                sigma = max(sigma * 0.87, sigma_min)
                no_improve += 1

            # ---- 2) coordinate-descent exploitation (periodic) ----
            if no_improve % cd_every == 0 and calls < budget and _now_ok():
                # try a small step along each axis from current best
                step = max(sigma * 0.5, 0.05)
                for d in range(dim):
                    if calls >= budget or not _now_ok():
                        break
                    for s in (step, -step):
                        if calls >= budget or not _now_ok():
                            break
                        z = x.copy()
                        z[d] = np.clip(x[d] + s, -5.0, 5.0)
                        fz = float(func(z))
                        calls += 1
                        if fz < best:
                            x, best = z, fz
                            no_improve = 0

            # ---- 3) stagnation-triggered random jump ----
            if no_improve >= stagnation_limit and calls < budget and _now_ok():
                jump = x + 2.0 * (rng.uniform(-1, 1, dim) * 0.5)
                jump = np.clip(jump, -5.0, 5.0)
                fj = float(func(jump))
                calls += 1
                if fj < best:
                    x, best = jump, fj
                    no_improve = 0
                # reset the stagnation counter to avoid jump spam
                no_improve = 0

        return x, best
