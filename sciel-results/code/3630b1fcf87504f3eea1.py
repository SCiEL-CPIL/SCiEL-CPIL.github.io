"""Black-box continuous optimizer for the BBOB-style f13 (Sharp ridge)."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """CMA-like (success-rule) covariance adaptation + coordinate refinement.

    Designed for a rotated, shifted, offset sharp ridge in 5D with 500
    evaluations.  The covariance matrix is adapted by a 1/5-like success rule
    (increase on improvement, decay otherwise) so it learns the narrow ridge
    direction and spreads along it, giving fast early progress (AOCC reward).
    A cheap coordinate-descent pass exploits the narrow ridge for additional
    gains.  A wall-clock deadline guard guarantees a clean return.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t0 = time.perf_counter()
        deadline = t0 + 270.0          # safety margin vs 300 s kill
        rng = np.random.default_rng(12345)

        # ---- initial point (centre of box) ----
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_f = fx
        calls = 1

        # ---- state ----
        C = np.eye(dim)                 # covariance (starts isotropic)
        sigma = 1.0
        min_sigma, max_sigma = 1e-3, 6.0

        def remaining():
            return budget - calls

        def try_point(p, f):
            nonlocal x, fx, best_x, best_f, calls
            if f < fx:
                x, fx = p, f
            if f < best_f:
                best_f, best_x = f, p.copy()

        def coord_descent(cx, cf):
            # one sweep of coordinate line search (cheap, exploits narrow ridge)
            nonlocal calls, best_f, best_x
            for _ in range(2):
                for d in range(dim):
                    step = 0.5
                    for _ in range(6):
                        if remaining() <= 0:
                            break
                        for sgn in (1.0, -1.0):
                            p = cx.copy()
                            p[d] = np.clip(p[d] + sgn * step, -5.0, 5.0)
                            f = func(p)
                            calls += 1
                            if f < cf:
                                cx, cf = p, f
                        if remaining() <= 0:
                            break
                    step *= 0.5
            if cf < best_f:
                best_f, best_x = cf, cx.copy()
            return cx, cf

        while remaining() > 0 and time.perf_counter() < deadline:
            # ---- sample a move from N(0, C) ----
            z = rng.standard_normal(dim)
            move = C @ z
            y = np.clip(x + sigma * move, -5.0, 5.0)
            fy = func(y)
            calls += 1
            improved = fy < fx

            if improved:
                # success rule: grow along the direction that worked
                C = C + 0.5 * np.outer(move, move)
                sigma *= 1.15
            else:
                C = 0.9 * C
                sigma *= 0.9
            sigma = min(max(sigma, min_sigma), max_sigma)

            # keep C well-conditioned / bounded
            C = 0.5 * (C + C.T)
            tr = np.trace(C)
            if not np.isfinite(tr) or tr <= 0:
                C = np.eye(dim)
                tr = float(dim)
            C = C / tr * dim            # normalise scale (isotropic when round)
            # clip eigenvalues to keep it stable
            try:
                w, V = np.linalg.eigh(C)
                w = np.clip(w, 1e-3, None)
                C = (V * w) @ V.T
            except np.linalg.LinAlgError:
                C = np.eye(dim)

            try_point(y, fy)

            # ---- periodic cheap coordinate refinement ----
            if calls % 25 == 0 and remaining() > 10:
                x, fx = coord_descent(x, fx)

        # ---- final polish ----
        if remaining() > 0:
            x, fx = coord_descent(best_x, best_f)

        return best_x, best_f
# EVOLVE-BLOCK-END
