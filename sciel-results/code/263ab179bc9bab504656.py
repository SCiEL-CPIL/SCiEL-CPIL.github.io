"""Rewritten optimizer: coordinate + Nelder-Mead hybrid with restarts and a time guard."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Hybrid coordinate-descent / Nelder-Mead with restarts and a wall-clock guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = int(self.budget)
        t0 = time.time()
        deadline = t0 + 275.0          # safety margin vs. 330 s kill
        max_calls = budget

        calls = [0]

        def f(x):
            calls[0] += 1
            val = float(func(x))
            if not np.isfinite(val):
                return 1e18
            return val

        def remaining():
            return deadline - time.time()

        def nelder_mead(x, fx, max_calls, max_iter):
            """Return best (x, fx) using Nelder-Mead with per-axis step scaling."""
            n = dim
            # Build initial simplex: unit steps along each axis
            simplex = [np.array(x, dtype=float)]
            for i in range(n):
                p = np.array(x, dtype=float)
                p[i] += 0.5
                simplex.append(p)
            fvals = [fx] + [f(p) for p in simplex[1:]]
            it = 0
            while it < max_iter and calls[0] < max_calls and remaining() > 5.0:
                it += 1
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]
                worst = simplex[-1]
                best = simplex[0]
                centroid = np.mean(simplex[:-1], axis=0)
                # Reflection
                xr = centroid + 1.0 * (centroid - worst)
                fr = f(xr)
                if fvals[0] <= fr < fvals[-2]:
                    simplex[-1] = xr
                    fvals[-1] = fr
                    continue
                # Expansion
                if fr < fvals[0]:
                    xe = centroid + 2.0 * (centroid - worst)
                    fe = f(xe)
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                    continue
                # Contraction
                if fr < fvals[-1]:
                    xc = centroid + 0.5 * (worst - centroid)
                    fc = f(xc)
                    if fc < fvals[-1]:
                        simplex[-1] = xc
                        fvals[-1] = fc
                        continue
                else:
                    xc = centroid - 0.5 * (worst - centroid)
                    fc = f(xc)
                    if fc < fvals[-1]:
                        simplex[-1] = xc
                        fvals[-1] = fc
                        continue
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = best + 0.5 * (simplex[i] - best)
                    fvals[i] = f(simplex[i])
            order = np.argsort(fvals)
            return simplex[order[0]], fvals[order[0]]

        def coordinate_polish(x, fx, max_calls, max_iter):
            """Golden-section line search along each coordinate, cyclic."""
            n = dim
            for _ in range(max_iter):
                if calls[0] >= max_calls or remaining() < 5.0:
                    break
                improved = False
                for i in range(n):
                    if calls[0] >= max_calls or remaining() < 5.0:
                        break
                    # Search along axis i within [-5, 5]
                    lo = -5.0
                    hi = 5.0
                    # Start with a small window around current value
                    base = x[i]
                    step = 0.5
                    a = max(lo, base - 2.0)
                    b = min(hi, base + 2.0)
                    # Golden-section on [a, b]
                    gr = (np.sqrt(5.0) - 1.0) / 2.0
                    c = b - gr * (b - a)
                    d = a + gr * (b - a)
                    x[i] = c
                    fc = f(x)
                    x[i] = d
                    fd = f(x)
                    for _ in range(40):
                        if calls[0] >= max_calls or remaining() < 5.0:
                            break
                        if fc < fd:
                            b = d
                            d = c
                            fd = fc
                            c = b - gr * (b - a)
                            x[i] = c
                            fc = f(x)
                        else:
                            a = c
                            c = d
                            fc = fd
                            d = a + gr * (b - a)
                            x[i] = d
                            fd = f(x)
                        if abs(b - a) < 1e-6:
                            break
                    # Pick the better of c, d
                    if fc < fd:
                        x[i] = c
                        fx = fc
                    else:
                        x[i] = d
                        fx = fd
                    if fx < fx - 1e-12:
                        improved = True
            return x, fx

        best_x = None
        best_fx = 1e18

        # Multi-start: a few deterministic starting points
        rng = np.random.default_rng(0)
        starts = []
        starts.append(np.zeros(dim))
        starts.append(rng.uniform(-4, 4, dim))
        starts.append(rng.uniform(-4, 4, dim))
        starts.append(rng.uniform(-2, 2, dim))

        # Budget split: give each start a chunk of the call budget
        per_start = max(50, budget // len(starts))

        for si, s in enumerate(starts):
            if calls[0] >= max_calls or remaining() < 10.0:
                break
            s = np.clip(s, -5, 5)
            fx = f(s)
            # Phase 1: Nelder-Mead
            x, fx = nelder_mead(s, fx, max_calls, max_iter=200)
            # Phase 2: coordinate polish
            x, fx = coordinate_polish(x, fx, max_calls, max_iter=10)
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            # If we improved a lot, do another polish
            if remaining() > 10.0 and calls[0] < max_calls:
                x, fx = coordinate_polish(x, fx, max_calls, max_iter=5)
                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(dim)
            best_fx = f(best_x)

        return best_x, best_fx
# EVOLVE-BLOCK-END
