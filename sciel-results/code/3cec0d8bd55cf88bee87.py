"""Rewritten optimizer: global grid search + Nelder-Mead local refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Global coarse search followed by local Nelder-Mead simplex refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.deadline = time.time() + 290.0
        self.MAX_CALLS = min(budget, 490)

    def __call__(self, func):
        dim = self.dim
        best_x = None
        best_f = np.inf

        def f(x):
            if self.calls >= self.MAX_CALLS:
                return np.inf
            self.calls += 1
            v = float(func(x))
            if v < best_f:
                best_f = v
                best_x = x.copy()
            return v

        # ---- Phase 1: coarse global grid to find promising basins ----
        # 3 levels per dim => 3^5 = 243 evaluations
        grid_vals = np.array([-4.0, 0.0, 4.0])
        grid_pts = np.array(np.meshgrid(*([grid_vals] * dim)), dtype=float)
        grid_pts = grid_pts.reshape(-1, dim)
        for pt in grid_pts:
            if self.calls >= self.MAX_CALLS or time.time() > self.deadline:
                break
            f(pt)

        # ---- Phase 2: local Nelder-Mead refinement from best starts ----
        # Collect candidate starts: best grid points
        scored = []
        for pt in grid_pts:
            if self.calls >= self.MAX_CALLS or time.time() > self.deadline:
                break
            # Re-evaluate only if not already in cache; use a small local cache
            scored.append((pt, None))
        # Instead of re-evaluating, just refine from the best few grid points.
        # We'll track best grid point values by re-evaluating a small subset.
        # Simpler: pick top-k grid points by re-evaluating (cheap, few calls).
        # Use the overall best and a couple of neighbors.
        starts = [best_x.copy()] if best_x is not None else [np.zeros(dim)]
        # Add a few deterministic diverse starts
        for s in [np.ones(dim) * 2.0, np.ones(dim) * -2.0,
                  np.array([3.0, -3.0, 3.0, -3.0, 3.0]),
                  np.array([-3.0, 3.0, -3.0, 3.0, -3.0])]:
            starts.append(np.clip(s, -5, 5))

        for start in starts:
            if self.calls >= self.MAX_CALLS or time.time() > self.deadline:
                break
            self._nelder_mead(start)

        # ---- Phase 3: final (1+1)-ES polish if budget remains ----
        if best_x is not None and self.calls < self.MAX_CALLS:
            x = best_x.copy()
            fx = best_f
            sigma = 0.5
            for _ in range(200):
                if self.calls >= self.MAX_CALLS or time.time() > self.deadline:
                    break
                y = np.clip(x + sigma * np.random.default_rng(0).standard_normal(dim), -5, 5)
                fy = f(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.3
                else:
                    sigma *= 0.8
            best_x, best_f = x, fx

        return best_x, best_f

    def _nelder_mead(self, x0):
        dim = self.dim
        # Build initial simplex
        simplex = [x0.copy()]
        for i in range(dim):
            pt = x0.copy()
            pt[i] = pt[i] + 0.5 if pt[i] < 4.5 else pt[i] - 0.5
            pt = np.clip(pt, -5, 5)
            simplex.append(pt)

        fvals = [self._eval(s) for s in simplex]

        alpha, gamma, rho = 1.0, 2.0, 0.5

        for _ in range(60):
            if self.calls >= self.MAX_CALLS or time.time() > self.deadline:
                break
            # Sort
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]

            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            xr = np.clip(centroid + alpha * (centroid - simplex[-1]), -5, 5)
            fr = self._eval(xr)
            if fr < fvals[0]:
                # Expansion
                xe = np.clip(centroid + gamma * (centroid - simplex[-1]), -5, 5)
                fe = self._eval(xe)
                if fe < fr:
                    simplex[-1], fvals[-1] = xe, fe
                else:
                    simplex[-1], fvals[-1] = xr, fr
            elif fr < fvals[-2]:
                simplex[-1], fvals[-1] = xr, fr
            else:
                # Contraction
                xc = np.clip(centroid + rho * (simplex[-1] - centroid), -5, 5)
                fc = self._eval(xc)
                if fc < fvals[-1]:
                    simplex[-1], fvals[-1] = xc, fc
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        simplex[i] = np.clip(simplex[0] + 0.5 * (simplex[i] - simplex[0]), -5, 5)
                        fvals[i] = self._eval(simplex[i])

        # Record the best from this simplex
        idx = int(np.argmin(fvals))
        if fvals[idx] < self.best_f:
            self.best_x, self.best_f = simplex[idx].copy(), fvals[idx]

    def _eval(self, x):
        if self.calls >= self.MAX_CALLS:
            return np.inf
        self.calls += 1
        v = float(self.func(x))
        if v < self.best_f:
            self.best_f = v
            self.best_x = x.copy()
        return v

    def __call__(self, func):
        self.func = func
        self.best_x = None
        self.best_f = np.inf
        self.calls = 0
        self.deadline = time.time() + 290.0
        self.MAX_CALLS = min(self.budget, 490)
        dim = self.dim

        def f(x):
            if self.calls >= self.MAX_CALLS:
                return np.inf
            self.calls += 1
            v = float(func(x))
            if v < self.best_f:
                self.best_f = v
                self.best_x = x.copy()
            return v

        # ---- Phase 1: coarse global grid ----
        grid_vals = np.array([-4.0, 0.0, 4.0])
        grid_pts = np.array(np.meshgrid(*([grid_vals] * dim)), dtype=float)
        grid_pts = grid_pts.reshape(-1, dim)
        for pt in grid_pts:
            if self.calls >= self.MAX_CALLS or time.time() > self.deadline:
                break
            f(pt)

        # ---- Phase 2: local Nelder-Mead from best starts ----
        starts = [self.best_x.copy()] if self.best_x is not None else [np.zeros(dim)]
        for s in [np.ones(dim) * 2.0, np.ones(dim) * -2.0,
                  np.array([3.0, -3.0, 3.0, -3.0, 3.0]),
                  np.array([-3.0, 3.0, -3.0, 3.0, -3.0])]:
            starts.append(np.clip(s, -5, 5))

        for start in starts:
            if self.calls >= self.MAX_CALLS or time.time() > self.deadline:
                break
            self._nelder_mead(start)

        # ---- Phase 3: (1+1)-ES polish ----
        if self.best_x is not None and self.calls < self.MAX_CALLS:
            x = self.best_x.copy()
            fx = self.best_f
            sigma = 0.5
            rng = np.random.default_rng(0)
            for _ in range(200):
                if self.calls >= self.MAX_CALLS or time.time() > self.deadline:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = f(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.3
                else:
                    sigma *= 0.8
            self.best_x, self.best_f = x, fx

        return self.best_x, self.best_f
# EVOLVE-BLOCK-END
