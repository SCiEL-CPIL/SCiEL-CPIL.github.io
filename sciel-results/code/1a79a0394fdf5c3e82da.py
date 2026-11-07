"""Derivative-free trust-region optimizer for BBOB f13 (sharp ridge)."""
import time
import numpy as np


class Optimizer:
    """DFTR: build local quadratic model, step on trust-region boundary."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        DEADLINE = 290.0          # safety margin before the 330 s wall-clock kill
        dim = self.dim
        budget = self.budget
        calls = [0]

        def f(x):
            calls[0] += 1
            return float(func(x))

        def elapsed():
            return time.time() - t0

        def within_budget():
            return calls[0] < budget and elapsed() < DEADLINE

        rng = np.random.default_rng(12345)

        # --- initial point (center is a strong prior for shifted box) ---
        x = np.zeros(dim)
        fx = f(x)
        best_x = x.copy()
        best_fx = fx

        def update_best(p, fp):
            nonlocal best_x, best_fx
            if fp < best_fx:
                best_fx = fp
                best_x = p.copy()

        # --- finite-difference model builder ---
        def build_model(c, fc):
            h = max(1e-4, 1e-3 * np.linalg.norm(c))
            g = np.zeros(dim)
            for i in range(dim):
                xp = c.copy(); xp[i] += h
                xm = c.copy(); xm[i] -= h
                fp = f(xp)
                fm = f(xm)
                g[i] = (fp - fm) / (2 * h)
            return g

        # --- trust-region quadratic step (boundary-constrained) ---
        def tr_step(c, fc, g, delta):
            """Minimize q(d)=fc+g'd+0.5 d'H d subject to ||d||<=delta.
            Ridge case: take a full boundary step along the most negative
            gradient direction (robust when H is ill-conditioned)."""
            ng = np.linalg.norm(g)
            if ng < 1e-10:
                return c, fc, 0.0
            # most-negative gradient direction, full boundary step
            d = -delta * g / ng
            xc = c + d
            fc_new = f(xc)
            return xc, fc_new, delta

        # --- adaptive trust-region main loop ---
        delta = 2.0
        min_delta = 1e-4
        max_delta = 4.0
        stall = 0

        while within_budget():
            g = build_model(x, fx)
            x_new, fx_new, step = tr_step(x, fx, g, delta)
            improvement = fx - fx_new
            ratio = improvement / (step * np.linalg.norm(g)) if step > 0 else 0.0

            if fx_new < fx:
                x, fx = x_new, fx_new
                update_best(x, fx)
                # expand trust region on good progress
                delta = min(max_delta, delta * 1.6)
                stall = 0
            else:
                # shrink trust region; if too small, jitter to escape
                delta *= 0.5
                if delta < min_delta:
                    delta = min_delta
                    stall += 1
                    if stall >= 3:
                        # random restart nudge to escape local flatness
                        x = x + delta * 0.5 * rng.standard_normal(dim)
                        x = np.clip(x, -5, 5)
                        fx = f(x)
                        update_best(x, fx)
                        delta = 0.5
                        stall = 0
            if not within_budget():
                break

        # --- final high-precision polish: tiny isotropic (1+1)-ES ---
        if within_budget():
            sigma = 1e-3
            for _ in range(40):
                if not within_budget():
                    break
                y = np.clip(best_x + sigma * rng.standard_normal(dim), -5, 5)
                fy = f(y)
                if fy < best_fx:
                    best_x, best_fx = y, fy
                    sigma *= 1.2
                else:
                    sigma *= 0.7

        return best_x, best_fx
