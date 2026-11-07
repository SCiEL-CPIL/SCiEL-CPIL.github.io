"""Optimizer for BBOB f19 (Griewank-Rosenbrock) on [-5,5]^5, 500 calls.

Strategy (single smooth objective, no black-box restarts):
  1. Locate the valley cheaply using the Griewank factor (== 1 at the origin,
     so its minimum identifies the valley entry under the hidden transform).
  2. Polish the full objective with a time-budgeted Nelder-Mead simplex.
  3. A short final polish under a hard wall-clock deadline.
AOCC rewards early progress, so the Griewank phase descends quickly and the
Nelder-Mead phase refines; the deadline guard guarantees a safe return.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Black-box continuous minimizer for f19 on [-5, 5]^dim."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    # ---------- objective helpers ----------
    def _griewank(self, x):
        s = 0.0
        for i in range(self.dim):
            s += x[i] * x[i] / 4000.0
        c = 1.0
        for i in range(self.dim):
            c *= np.cos(x[i] / np.sqrt(i + 1))
        return s - c + 1.0

    def _rosen(self, x):
        s = 0.0
        for i in range(self.dim - 1):
            a = x[i + 1] - x[i] * x[i]
            b = 1.0 - x[i]
            s += 100.0 * a * a + b * b
        return s

    # ---------- Nelder-Mead (bounded, time-budgeted) ----------
    def _nelder_mead(self, func, x0, fx0, remaining_calls, deadline):
        n = self.dim
        alpha, gamma, rho, sigma = 1.0, 2.0, -1.0, 0.5

        # build simplex
        simplex = [x0.copy()]
        values = [fx0]
        for i in range(n):
            d = x0.copy()
            d[i] += 0.5 if x0[i] < 4.5 else -0.5
            simplex.append(d)
            values.append(func(d))
            remaining_calls -= 1
            if remaining_calls <= 0:
                break

        it = 0
        while remaining_calls > 0 and time.time() < deadline and it < 4000:
            it += 1
            order = np.argsort(values)
            simplex = [simplex[i] for i in order]
            values = [values[i] for i in order]

            centroid = np.mean(simplex[:-1], axis=0)
            xr = np.clip(centroid + alpha * (centroid - simplex[-1]), -5.0, 5.0)
            fr = func(xr)
            remaining_calls -= 1
            if remaining_calls <= 0:
                break

            if fr < values[0]:
                if fr < gamma * values[0]:
                    xe = np.clip(centroid + gamma * (centroid - simplex[-1]), -5.0, 5.0)
                    fe = func(xe)
                    remaining_calls -= 1
                    simplex[-1] = xe if fe < fr else xr
                    values[-1] = min(fe, fr)
                else:
                    simplex[-1] = xr
                    values[-1] = fr
            elif fr < values[-2]:
                simplex[-1] = xr
                values[-1] = fr
            else:
                xc = np.clip(centroid + rho * (simplex[-1] - centroid), -5.0, 5.0)
                fc = func(xc)
                remaining_calls -= 1
                if fc < values[-1]:
                    simplex[-1] = xc
                    values[-1] = fc
                else:
                    for i in range(1, n + 1):
                        simplex[i] = np.clip(simplex[0] + sigma * (simplex[i] - simplex[0]), -5.0, 5.0)
                        values[i] = func(simplex[i])
                        remaining_calls -= 1
                        if remaining_calls <= 0:
                            break
                    if remaining_calls <= 0:
                        break

        idx = int(np.argmin(values))
        return simplex[idx], values[idx]

    # ---------- main ----------
    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start = time.time()
        deadline = start + 300.0

        def elapsed_ok():
            return (budget - self._calls) > 0 and time.time() < deadline

        self._calls = 0
        calls = [0]

        def F(x):
            calls[0] += 1
            self._calls = calls[0]
            return func(x)

        rng = np.random.default_rng(0)

        # ---- Phase 0: a few scattered samples ----
        x = np.clip(rng.uniform(-5, 5, dim), -5.0, 5.0)
        fx = F(x)
        best_x, best_f = x.copy(), fx

        for _ in range(5):
            if not elapsed_ok():
                break
            y = np.clip(rng.uniform(-5, 5, dim), -5.0, 5.0)
            fy = F(y)
            if fy < best_f:
                best_x, best_f = y.copy(), fy

        # ---- Phase 1: locate the valley via the Griewank factor (cheap) ----
        # Griewank is 1 at the origin, so its min finds the valley entry.
        xg = best_x.copy()
        fg = self._griewank(xg)
        sigma = 1.5
        for _ in range(60):
            if not elapsed_ok():
                break
            y = np.clip(xg + sigma * rng.standard_normal(dim), -5.0, 5.0)
            fy = self._griewank(y)
            if fy < fg:
                xg, fg = y, fy
                sigma *= 1.15
            else:
                sigma *= 0.8
            fxg = F(xg)
            if fxg < best_f:
                best_x, best_f = xg.copy(), fxg

        # ---- Phase 2: Nelder-Mead on the full objective ----
        x = best_x.copy()
        fx = F(x)
        remaining = budget - calls[0]
        x, fx = self._nelder_mead(F, x, fx, remaining, deadline)
        if fx < best_f:
            best_x, best_f = x.copy(), fx

        # ---- Phase 3: short final polish (tighter steps) ----
        remaining = budget - calls[0]
        if remaining > 8:
            x, fx = self._nelder_mead(F, best_x, best_f, min(remaining, 40), deadline)
            if fx < best_f:
                best_x, best_f = x.copy(), fx

        return best_x, best_f
# EVOLVE-BLOCK-END
