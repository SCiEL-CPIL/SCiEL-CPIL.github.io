"""Fresh design: anisotropy-aware coordinate descent for Bent Cigar (f12)."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Coordinate descent with golden-section 1D refinement + random restarts.

    Exploits the Bent Cigar's anisotropy: a few coordinates carry large
    quadratic weights while one (the 'cigar' axis) is nearly flat. We cheaply
    identify the important axes, then refine each with a dense 1D search,
    and use fresh random restarts to defeat the hidden shift/rotation.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        deadline = time.time() + 290.0  # safety margin before the 300 s kill
        calls = [0]

        def F(x):
            calls[0] += 1
            return func(np.asarray(x, dtype=float))

        def golden(x, i, a, b, f0, fa, fb):
            """Refine coordinate i in [a, b] given f0/fa/fb at a/x/b."""
            phi = 0.6180339887498949
            for _ in range(20):
                if calls[0] >= self.budget:
                    break
                if fa < fb:
                    b, fb = x[i], fb
                    x[i] = b - phi * (b - a)
                    fa = F(x)
                else:
                    a, fa = x[i], fa
                    x[i] = a + phi * (b - a)
                    fb = F(x)
            # pick the better of the two retained points
            if fa <= fb:
                x[i] = a
                return fa
            else:
                x[i] = b
                return fb

        best_x = None
        best_f = np.inf

        def try_point(x0):
            nonlocal best_x, best_f
            x = np.array(x0, dtype=float)
            fx = F(x)
            if fx < best_f:
                best_x, best_f = x.copy(), fx
            # cheap axis probe: does moving each coordinate help?
            step = 1.0
            useful = []
            for i in range(d):
                if calls[0] >= self.budget:
                    break
                x[i] = min(5.0, x[i] + step)
                fp = F(x)
                x[i] = max(-5.0, x[i] - 2 * step)
                fm = F(x)
                x[i] = x0[i]
                if fp < fx or fm < fx:
                    useful.append(i)
            # dense golden refinement on the axes that reacted
            for i in useful:
                if calls[0] >= self.budget:
                    break
                a = max(-5.0, x[i] - 3.0)
                b = min(5.0, x[i] + 3.0)
                if b - a < 1e-9:
                    continue
                x[i] = a
                fa = F(x)
                x[i] = b
                fb = F(x)
                fx = golden(x, i, a, b, fx, fa, fb)
            # final polish pass over all axes
            for i in range(d):
                if calls[0] >= self.budget:
                    break
                a = max(-5.0, x[i] - 1.0)
                b = min(5.0, x[i] + 1.0)
                if b - a < 1e-9:
                    continue
                x[i] = a
                fa = F(x)
                x[i] = b
                fb = F(x)
                fx = golden(x, i, a, b, fx, fa, fb)
            if fx < best_f:
                best_x, best_f = x.copy(), fx

        # diverse random restarts to escape the hidden shift
        rng = np.random.default_rng(0)
        while calls[0] < self.budget and time.time() < deadline:
            x0 = rng.uniform(-5.0, 5.0, d)
            try_point(x0)

        return best_x, best_f
# EVOLVE-BLOCK-END
