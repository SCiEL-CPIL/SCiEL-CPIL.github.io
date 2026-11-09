"""Fresh design: multi-start global scan + adaptive local search + Nelder-Mead."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Multi-start (coarse scan + adaptive ES + Nelder-Mead) for f21."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        d = self.dim
        deadline = time.time() + 275.0   # safety margin under 300s wall-clock
        budget = self.budget
        rng = np.random.default_rng(0)

        # --- coarse global scan: spread 64 points across the box ---
        grid = np.array(np.meshgrid(*[np.linspace(-5, 5, 8) for _ in range(d)]),
                        dtype=float).reshape(d, -1).T
        n_scan = min(64, grid.shape[0])
        idx = np.arange(grid.shape[0])
        rng.shuffle(idx)
        idx = idx[:n_scan]

        best_x, best_f = None, np.inf
        for i in idx:
            x = grid[i].copy()
            fx = func(x)
            budget -= 1
            if fx < best_f:
                best_x, best_f = x.copy(), fx
            if budget <= 0 or time.time() > deadline:
                break

        if best_x is None:
            best_x = rng.uniform(-5, 5, d)
            best_f = func(best_x)
            budget -= 1

        # --- local search from the best scan point ---
        best_x, best_f = self._local(func, best_x, best_f, budget, deadline, rng)
        budget -= self._used  # track consumed budget

        # --- extra restarts from remaining budget ---
        restarts = 0
        while budget > 200 and restarts < 10 and time.time() < deadline:
            # random restart, biased toward the current best basin
            if rng.random() < 0.5:
                x0 = best_x + 1.5 * rng.standard_normal(d)
            else:
                x0 = rng.uniform(-5, 5, d)
            x0 = np.clip(x0, -5, 5)
            f0 = func(x0)
            budget -= 1
            used = self._local(func, x0, f0, budget, deadline, rng)
            budget -= used
            restarts += 1

        return best_x, best_f

    # ---- local search: adaptive ES then Nelder-Mead ----
    def _local(self, func, x, fx, budget, deadline, rng):
        d = self.dim
        self._used = 0

        # adaptive (1+1)-ES with success-rate step control
        sigma = 1.0
        n_es = min(budget, 120)
        for _ in range(n_es):
            if self._used >= budget or time.time() > deadline:
                break
            y = np.clip(x + sigma * rng.standard_normal(d), -5, 5)
            fy = func(y)
            self._used += 1
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.35
            else:
                sigma *= 0.8
            if sigma < 1e-5:
                break

        # Nelder-Mead polish
        x, fx = self._nelder(func, x, fx, budget, deadline, rng)
        self._used += self._nm_used
        return x, fx

    def _nelder(self, func, x, fx, budget, deadline, rng):
        d = self.dim
        self._nm_used = 0
        step = max(0.2, 1e-4)
        simplex = [x.copy()]
        fvals = [fx]
        for i in range(d):
            v = x.copy()
            v[i] += step
            v = np.clip(v, -5, 5)
            fv = func(v)
            self._nm_used += 1
            simplex.append(v)
            fvals.append(fv)
        simplex = np.array(simplex)
        fvals = np.array(fvals)

        iters = 0
        max_it = 100
        while iters < max_it and self._nm_used < budget and time.time() < deadline:
            iters += 1
            order = np.argsort(fvals)
            simplex = simplex[order]
            fvals = fvals[order]
            if self._nm_used >= budget or time.time() > deadline:
                break
            centroid = np.mean(simplex[:-1], axis=0)
            # reflect
            xr = centroid + 1.0 * (centroid - simplex[-1])
            xr = np.clip(xr, -5, 5)
            fr = func(xr); self._nm_used += 1
            if fr < fvals[0]:
                # expand
                xe = centroid + 2.0 * (centroid - simplex[-1])
                xe = np.clip(xe, -5, 5)
                fe = func(xe); self._nm_used += 1
                if fe < fr:
                    simplex[-1] = xe; fvals[-1] = fe
                else:
                    simplex[-1] = xr; fvals[-1] = fr
            elif fr < fvals[-2]:
                simplex[-1] = xr; fvals[-1] = fr
            else:
                # contract
                xc = centroid + 0.5 * (simplex[-1] - centroid)
                xc = np.clip(xc, -5, 5)
                fc = func(xc); self._nm_used += 1
                if fc < fvals[-1]:
                    simplex[-1] = xc; fvals[-1] = fc
                else:
                    # shrink toward best
                    for i in range(1, d + 1):
                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        fvals[i] = func(simplex[i]); self._nm_used += 1
                    if self._nm_used >= budget or time.time() > deadline:
                        break
        order = np.argmin(fvals)
        return simplex[order].copy(), float(fvals[order])
# EVOLVE-BLOCK-END
