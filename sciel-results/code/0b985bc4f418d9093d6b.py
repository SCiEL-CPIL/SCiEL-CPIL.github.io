"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start robust optimizer: adaptive (1+1)-ES + coordinate-descent polish,
    with a wall-clock deadline guard and a strict evaluation budget."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_wall = time.time()
        deadline = start_wall + 285.0  # safety margin before the 330s kill

        def time_ok():
            return (time.time() - start_wall) < deadline

        # strict call accounting
        calls = [0]

        def f(x):
            x = np.asarray(x, dtype=float)
            x = np.clip(x, -5.0, 5.0)
            calls[0] += 1
            v = func(x)
            return float(v)

        rng = np.random.default_rng(12345)

        # ---- global best tracking ----
        best_x = None
        best_f = np.inf

        def consider(x, fx):
            nonlocal best_x, best_f
            if fx < best_f:
                best_f = fx
                best_x = np.array(x, dtype=float)
                return True
            return False

        # ---- coordinate-descent line polish (axis-aligned, fast early progress) ----
        def ccd(x, fx, max_steps=60):
            x = np.array(x, dtype=float)
            fx = float(fx)
            for _ in range(max_steps):
                if not time_ok() or calls[0] >= budget:
                    break
                improved = False
                for d in range(dim):
                    if not time_ok() or calls[0] >= budget:
                        break
                    # simple line search along axis d: try a few step sizes
                    base = x.copy()
                    best_local = fx
                    best_step = 0.0
                    for step in (0.5, 0.25, 0.1, 0.05, -0.5, -0.25, -0.1, -0.05):
                        cand = base.copy()
                        cand[d] = np.clip(cand[d] + step, -5.0, 5.0)
                        fc = f(cand)
                        if fc < best_local:
                            best_local = fc
                            best_step = step
                    if best_step != 0.0:
                        x = base.copy()
                        x[d] = np.clip(x[d] + best_step, -5.0, 5.0)
                        fx = best_local
                        improved = True
                if not improved:
                    break
            consider(x, fx)
            return x, fx

        # ---- adaptive (1+1)-ES (rotation-robust, noise-robust) ----
        def es(x0, fx0, max_steps=40):
            x = np.array(x0, dtype=float)
            fx = float(fx0)
            sigma = 1.0
            successes = 0
            for _ in range(max_steps):
                if not time_ok() or calls[0] >= budget:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5.0, 5.0)
                fy = f(y)
                if fy < fx:
                    x, fx = y, fy
                    successes += 1
                    sigma *= 1.25
                else:
                    sigma *= 0.85
                consider(x, fx)
            return x, fx

        # ---- initial candidates: random + structured ----
        starts = []
        # a few random points
        for _ in range(4):
            starts.append(rng.uniform(-5.0, 5.0, dim))
        # center-ish and a few offset points (Lunacek has a secondary minimum)
        starts.append(np.zeros(dim))
        starts.append(np.full(dim, 2.5))
        starts.append(np.full(dim, -2.5))
        starts.append(rng.uniform(-4.0, 4.0, dim))

        # ---- phase 1: quick CCD from each start to grab early progress ----
        for s in starts:
            if not time_ok() or calls[0] >= budget:
                break
            fx = f(s)
            consider(s, fx)
            # short CCD to get on a basin
            ccd(s, fx, max_steps=25)

        # ---- phase 2: alternate ES + CCD from the current best until budget/time ----
        while time_ok() and calls[0] < budget:
            if best_x is None:
                break
            # ES from best to explore/escape
            x, fx = es(best_x, best_f, max_steps=25)
            # CCD polish
            x, fx = ccd(x, fx, max_steps=40)
            # occasionally re-seed from a fresh random point to find the other basin
            if calls[0] < budget - 30 and rng.random() < 0.25:
                s = rng.uniform(-5.0, 5.0, dim)
                fx2 = f(s)
                consider(s, fx2)
                ccd(s, fx2, max_steps=20)

        # ---- final polish with remaining budget ----
        if best_x is not None and time_ok() and calls[0] < budget:
            ccd(best_x, best_f, max_steps=40)

        return best_x, best_f
# EVOLVE-BLOCK-END
