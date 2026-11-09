"""Optimizer for BBOB f4 (Büche-Rastrigin) on [-5,5]^5."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _time_ok(self, t0, margin=15.0):
        return (time.time() - t0) < 330.0 - margin

    def _local_polish(self, func, x0, fx0, calls, budget, t0, margin=15.0):
        # Adaptive step-size local search (Nelder-Mead-like + step shrink)
        x = np.array(x0, dtype=float)
        fx = fx0
        n = self.dim
        # Build simplex
        simplex = [x.copy()]
        fvals = [fx]
        for i in range(n):
            v = x.copy()
            v[i] += 0.05
            v = np.clip(v, -5, 5)
            fv = func(v)
            calls[0] += 1
            simplex.append(v)
            fvals.append(fv)
        max_iter = max(1, budget - calls[0])
        for _ in range(max_iter):
            if not self._time_ok(t0, margin):
                break
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]
            worst = simplex[-1]
            fworst = fvals[-1]
            centroid = np.mean(simplex[:-1], axis=0)
            # Reflection
            alpha = 1.0
            xr = centroid + alpha * (centroid - worst)
            xr = np.clip(xr, -5, 5)
            fr = func(xr)
            calls[0] += 1
            if fr < fvals[0]:
                # Expansion
                xe = centroid + 2.0 * (centroid - worst)
                xe = np.clip(xe, -5, 5)
                fe = func(xe)
                calls[0] += 1
                if fe < fr:
                    simplex[-1] = xe
                    fvals[-1] = fe
                else:
                    simplex[-1] = xr
                    fvals[-1] = fr
            elif fr < fworst:
                simplex[-1] = xr
                fvals[-1] = fr
            else:
                # Contraction
                gamma = 0.5
                xc = centroid + gamma * (worst - centroid)
                xc = np.clip(xc, -5, 5)
                fc = func(xc)
                calls[0] += 1
                if fc < fworst:
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        si = simplex[i] + 0.5 * (simplex[0] - simplex[i])
                        si = np.clip(si, -5, 5)
                        simplex[i] = si
                        fvals[i] = func(si)
                        calls[0] += 1
            # Track best
            bidx = int(np.argmin(fvals))
            if fvals[bidx] < fx:
                fx = fvals[bidx]
                x = simplex[bidx].copy()
        return x, fx

    def __call__(self, func):
        t0 = time.time()
        n = self.dim
        budget = self.budget
        calls = [0]
        rng = np.random.default_rng(12345)

        def eval(x):
            x = np.clip(np.asarray(x, dtype=float), -5, 5)
            calls[0] += 1
            return func(x)

        best_x = None
        best_f = float('inf')

        def consider(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = np.array(x, dtype=float)

        # Phase 1: Global random sampling (Büche-Rastrigin is multimodal)
        n_global = min(200, budget // 3)
        for _ in range(n_global):
            if calls[0] >= budget or not self._time_ok(t0):
                break
            x = rng.uniform(-5, 5, n)
            f = eval(x)
            consider(x, f)

        # Phase 2: CMA-ES-like from a few good starting points
        # Gather top starting points from global sampling
        # (we only kept best; do a small re-sampled set)
        # Use a simple population-based evolution
        pop_size = 8
        if calls[0] < budget and self._time_ok(t0):
            # Initialize population around best + random
            pop = [best_x.copy() if best_x is not None else rng.uniform(-5, 5, n)]
            for _ in range(pop_size - 1):
                if best_x is not None and rng.random() < 0.5:
                    p = best_x + 1.0 * rng.standard_normal(n)
                else:
                    p = rng.uniform(-5, 5, n)
                p = np.clip(p, -5, 5)
                pop.append(p)
            pf = [eval(p) for p in pop]
            for p, f in zip(pop, pf):
                consider(p, f)

            sigma = 1.0
            max_gen = max(1, (budget - calls[0]) // pop_size)
            for gen in range(max_gen):
                if calls[0] >= budget or not self._time_ok(t0):
                    break
                # Select parents (top half)
                idx = np.argsort(pf)
                parents = [pop[i] for i in idx[:pop_size // 2]]
                new_pop = []
                new_pf = []
                for _ in range(pop_size):
                    if not self._time_ok(t0):
                        break
                    pi = parents[rng.integers(len(parents))]
                    child = pi + sigma * rng.standard_normal(n)
                    child = np.clip(child, -5, 5)
                    fc = eval(child)
                    new_pop.append(child)
                    new_pf.append(fc)
                    consider(child, fc)
                # Combine with best of old, keep top pop_size
                combined = list(zip(pop, pf)) + list(zip(new_pop, new_pf))
                combined.sort(key=lambda t: t[1])
                pop = [c[0] for c in combined[:pop_size]]
                pf = [c[1] for c in combined[:pop_size]]
                # Adapt sigma
                improved = sum(1 for c in combined[:pop_size] if c[1] < min(pf))
                if improved > 0:
                    sigma *= 1.25
                else:
                    sigma *= 0.8
                if sigma < 1e-4:
                    sigma = 1e-4

        # Phase 3: Local polish from best point
        if best_x is not None and calls[0] < budget and self._time_ok(t0):
            remaining = budget - calls[0]
            if remaining > 10:
                x, f = self._local_polish(func, best_x, best_f, calls, remaining, t0)
                consider(x, f)

        # Phase 4: If budget remains, do a finer local step search
        if best_x is not None and calls[0] < budget and self._time_ok(t0):
            remaining = budget - calls[0]
            if remaining > 5:
                x = best_x.copy()
                fx = best_f
                step = 0.05
                for _ in range(remaining):
                    if not self._time_ok(t0):
                        break
                    improved = False
                    for i in range(n):
                        for sign in (1.0, -1.0):
                            v = x.copy()
                            v[i] += sign * step
                            v = np.clip(v, -5, 5)
                            fv = eval(v)
                            if fv < fx:
                                x, fx = v, fv
                                improved = True
                                consider(x, fx)
                    if not improved:
                        step *= 0.5
                        if step < 1e-5:
                            break
                consider(x, fx)

        return best_x, best_f
# EVOLVE-BLOCK-END
