"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """CMA-ES black-box continuous optimizer for the rotated Discus (f11)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before the 330s kill

        rng = np.random.default_rng(0)

        # Best-so-far bookkeeping (guards against exceptions / early stop).
        x0 = rng.uniform(-4.0, 4.0, dim)
        fx0 = float(func(x0))
        best_x, best_f = x0, fx0
        calls = 1

        def timed_func(x):
            nonlocal calls, best_x, best_f
            fx = float(func(x))
            calls += 1
            if fx < best_f:
                best_x, best_f = x, fx
            return fx

        # Primary path: CMA-ES (designed for smooth rotated spheres like Discus).
        try:
            from cma import CMA
        except Exception:
            CMA = None

        if CMA is not None:
            try:
                opt = CMA(
                    best_x,
                    2.5,
                    {
                        "bounds": [(-5.0, 5.0)] * dim,
                        "maxiter": 10**9,
                        "verbose": -9,
                        "seed": 1,
                    },
                )
                while not opt.stop() and calls < budget and time.time() < deadline:
                    xs = opt.ask()
                    fs = []
                    for x in xs:
                        if calls >= budget or time.time() > deadline:
                            break
                        fs.append(timed_func(x))
                    if fs:
                        opt.tell(xs[: len(fs)], fs)
            except Exception:
                pass

        # Final high-precision polish: tiny-step (1+1) refinement of the best point.
        try:
            sigma = 0.05
            for _ in range(50):
                if calls >= budget or time.time() > deadline:
                    break
                y = np.clip(best_x + sigma * rng.standard_normal(dim), -5.0, 5.0)
                fy = timed_func(y)
                if fy < best_f:
                    sigma *= 1.2
                else:
                    sigma *= 0.7
        except Exception:
            pass

        return best_x, best_f
# EVOLVE-BLOCK-END
