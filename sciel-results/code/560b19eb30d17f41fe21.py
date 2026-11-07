"""Rewritten optimizer exploiting the quadratic structure of BBOB f1 (Sphere)."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Quadratic-structure estimator + (1+1)-ES polish for Sphere f1."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start = time.time()
        deadline = start + 320.0  # safety margin before the 330 s kill

        # ---- Phase 1: estimate minimizer and per-axis curvature ----
        # f(x) = offset + sum_i a_i (x_i - x*_i)^2
        c = np.zeros(dim)
        fc = func(c)
        best_x = c.copy()
        best_f = float(fc)

        e = np.eye(dim)
        xs = np.zeros((dim, 2, dim))
        fs = np.zeros((dim, 2))
        for i in range(dim):
            xs[i, 0] = c + e[i]
            xs[i, 1] = c - e[i]
            fs[i, 0] = func(xs[i, 0])
            fs[i, 1] = func(xs[i, 1])

        # a_i = (f_{+} + f_{-} - 2 f_c)/2 ; x*_i = c_i - (f_{+} - f_{-})/(2 a_i)
        a = 0.5 * (fs[:, 0] + fs[:, 1] - 2.0 * fc)
        a = np.where(np.abs(a) < 1e-12, np.sign(a) * 1e-12 + 1e-12, a)
        xstar = c - (fs[:, 0] - fs[:, 1]) / (2.0 * a)
        xstar = np.clip(xstar, -5.0, 5.0)

        fx = func(xstar)
        if fx < best_f:
            best_f, best_x = fx, xstar.copy()

        # ---- Phase 2: polish with a (1+1)-ES from x* ----
        x = best_x.copy()
        fx = best_f
        sigma = 0.05
        rng = np.random.default_rng(0)

        for _ in range(budget - (1 + 2 * dim)):
            if time.time() >= deadline:
                break
            y = np.clip(x + sigma * rng.standard_normal(dim), -5.0, 5.0)
            fy = func(y)
            if fy < fx:
                x, fx = y, fy
                if fy < best_f:
                    best_f, best_x = fy, x.copy()
                sigma *= 1.3
                if sigma > 0.5:
                    sigma = 0.5
            else:
                sigma *= 0.85
                if sigma < 1e-5:
                    sigma = 1e-5

        return best_x, best_f
# EVOLVE-BLOCK-END
