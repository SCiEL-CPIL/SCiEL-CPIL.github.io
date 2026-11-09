"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc


class Optimizer:
    """Global Latin-hypercube exploration feeding adaptive multi-start Nelder-Mead,
    with a Rosenbrock-valley-aware final polish. All phases share one call/time budget."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def _nelder_mead(self, func, x0, remaining_budget, deadline):
        """Run Nelder-Mead from x0, using at most remaining_budget calls.
        Returns (best_x, best_f, calls_used)."""
        dim = self.dim
        n = dim + 1
        simplex = np.zeros((n, dim))
        simplex[0] = x0
        for i in range(1, n):
            simplex[i] = x0.copy()
            simplex[i, i - 1] = x0[i - 1] + 0.5

        fvals = np.zeros(n)
        calls_used = 0
        for i in range(n):
            if time.time() > deadline or calls_used >= remaining_budget:
                break
            fvals[i] = func(simplex[i])
            calls_used += 1

        best_idx = int(np.argmin(fvals))
        best_x = simplex[best_idx].copy()
        best_f = fvals[best_idx]

        alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5

        while calls_used < remaining_budget and time.time() < deadline:
            order = np.argsort(fvals)
            simplex = simplex[order]
            fvals = fvals[order]
            worst_idx = n - 1
            second_worst_idx = n - 2

            if np.abs(fvals[worst_idx] - fvals[0]) < 1e-12:
                break

            centroid = np.mean(simplex[:n - 1], axis=0)

            xr = np.clip(centroid + alpha * (centroid - simplex[worst_idx]), -5, 5)
            fr = func(xr)
            calls_used += 1

            if fr < fvals[0]:
                xe = np.clip(centroid + gamma * (xr - centroid), -5, 5)
                fe = func(xe)
                calls_used += 1
                if fe < fr:
                    simplex[worst_idx], fvals[worst_idx] = xe, fe
                else:
                    simplex[worst_idx], fvals[worst_idx] = xr, fr
            elif fr < fvals[second_worst_idx]:
                simplex[worst_idx], fvals[worst_idx] = xr, fr
            else:
                if fr < fvals[worst_idx]:
                    xc = centroid + rho * (xr - centroid)
                else:
                    xc = centroid - rho * (xr - centroid)
                xc = np.clip(xc, -5, 5)
                fc = func(xc)
                calls_used += 1
                if fc <= fr:
                    simplex[worst_idx], fvals[worst_idx] = xc, fc
                else:
                    for i in range(1, n):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        fvals[i] = func(simplex[i])
                        calls_used += 1
                        if calls_used >= remaining_budget:
                            break

            ci = int(np.argmin(fvals))
            if fvals[ci] < best_f:
                best_f = fvals[ci]
                best_x = simplex[ci].copy()

        return best_x, best_f, calls_used

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 20s safety margin under the 300s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        total_budget = self.budget

        def guard():
            return time.time() < deadline

        best_x = None
        best_f = np.inf
        calls_made = 0

        def record(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # ---- Phase 1: global exploration via deterministic Latin-hypercube sample ----
        # Cheap, well-spread points to locate the basin of the hidden-shifted minimum.
        exp_budget = min(120, max(40, int(total_budget * 0.24)))
        m = exp_budget
        try:
            sampler = qmc.LatinHypercube(d=dim, seed=42)
            pts = sampler.random(m) * 10.0 - 5.0  # map [0,1)^d -> [-5,5)^d
        except Exception:
            pts = rng.uniform(-5, 5, size=(m, dim))

        cand = []
        for i in range(m):
            if calls_made >= total_budget or not guard():
                break
            x = np.clip(pts[i], -5, 5)
            f = func(x)
            calls_made += 1
            record(x, f)
            cand.append((f, x.copy()))

        # Fallback if exploration produced nothing.
        if best_x is None:
            x = np.clip(np.zeros(dim), -5, 5)
            f = func(x)
            calls_made += 1
            record(x, f)
            cand.append((f, x.copy()))

        # ---- Phase 2: adaptive multi-start Nelder-Mead from the best explorers ----
        cand.sort(key=lambda t: t[0])
        seen = [best_x]
        for f0, x0 in cand[:6]:
            if calls_made >= total_budget - 15 or not guard():
                break
            remaining = total_budget - calls_made
            if remaining < 15:
                break
            # skip near-duplicates to avoid wasting a restart
            if any(np.linalg.norm(x0 - s) < 1e-6 for s in seen):
                continue
            seen.append(x0.copy())
            nx, nf, used = self._nelder_mead(func, np.clip(x0, -5, 5), remaining, deadline)
            calls_made += used
            record(nx, nf)

        # ---- Phase 3: Rosenbrock-valley-aware polish ----
        # The Rosenbrock valley is x[i+1] ~ x[i]^2; a short line search along it
        # exploits the known geometry to squeeze out the final drop.
        if guard() and calls_made < total_budget - 5 and best_x is not None:
            base = best_x.copy()
            for _ in range(8):
                if not guard() or calls_made >= total_budget:
                    break
                y = base.copy()
                for i in range(dim - 1):
                    y[i + 1] = y[i] * y[i]
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_made += 1
                if fy < best_f - 1e-12:
                    best_f = fy
                    best_x = y.copy()
                    base = y.copy()
                else:
                    break

        # ---- Phase 4: local random-walk finish on the remaining budget ----
        if guard() and best_x is not None:
            sigma = 0.2
            while calls_made < total_budget and guard():
                y = np.clip(best_x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_made += 1
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                    sigma *= 0.7
                else:
                    sigma *= 0.9
                    if sigma < 1e-4:
                        break

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls_made += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
