"""Paraboloid basin-fitting with direct basin jumps + Nelder-Mead / coordinate-descent polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Deterministic two-basin search: fit a local quadratic, jump to its center,
    then polish with Nelder-Mead and fine coordinate descent."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _halton(self, index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r

    def _halton_sequence(self, n, dim):
        pts = []
        for i in range(1, n + 1):
            p = [self._halton(i, b) for b in range(2, dim + 2)]
            p = [10.0 * q - 5.0 for q in p]
            pts.append(np.array(p, dtype=float))
        return pts

    def _fit_paraboloid(self, center, step, f, calls, budget, deadline):
        """Fit a quadratic to center + 2n axis points; return predicted minimum."""
        n = self.dim
        # center value
        c = np.clip(center, -5, 5)
        if calls[0] >= budget or time.time() > deadline:
            return None, None
        fc = f(c)

        pts = [c]
        vals = [fc]
        for d in range(n):
            for s in (1.0, -1.0):
                if calls[0] >= budget or time.time() > deadline:
                    return None, None
                x = c.copy()
                x[d] = np.clip(x[d] + s * step, -5, 5)
                if np.allclose(x, c, atol=1e-12):
                    continue
                pts.append(x)
                vals.append(f(x))

        # Need at least 2n+1 points
        if len(pts) < 2 * n + 1:
            return None, None

        A = np.array(pts)
        b = np.array(vals)
        # Least-squares fit of quadratic: b = A^T h + A^2 c + k
        # Build design matrix
        D = np.hstack([A, A ** 2, np.ones((len(A), 1))])
        try:
            coef, *_ = np.linalg.lstsq(D, b, rcond=None)
        except Exception:
            return None, None
        h = coef[:n]
        c_q = coef[n:2 * n]
        # Minimum of quadratic: x* = -0.5 * inv(2*C) * h  where C = diag(c_q)
        # For diagonal quadratic, x*_d = -h_d / (2*c_q_d)
        x_star = np.zeros(n)
        ok = True
        for d in range(n):
            if abs(c_q[d]) < 1e-8:
                ok = False
                break
            x_star[d] = -h[d] / (2.0 * c_q[d])
        if not ok:
            return None, None
        x_star = np.clip(x_star, -5, 5)
        return x_star, fc

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        calls = [0]
        dim = self.dim
        budget = self.budget

        def f(x):
            if calls[0] >= budget:
                return float('inf')
            if time.time() > deadline:
                return float('inf')
            calls[0] += 1
            x_clipped = np.clip(x, -5, 5)
            return func(x_clipped)

        best_x = None
        best_f = float('inf')

        def update_best(x, fx):
            nonlocal best_x, best_f
            if fx < best_f:
                best_x = x.copy()
                best_f = fx

        # ---------- Phase 1: coarse diverse exploration ----------
        halton_points = self._halton_sequence(15, dim)
        candidate_starts = list(halton_points)
        candidate_starts.append(np.zeros(dim))
        candidate_starts.append(np.full(dim, 1.0))
        candidate_starts.append(np.full(dim, -1.0))
        candidate_starts.append(np.array([1.0, -1.0, 1.0, -1.0, 1.0][:dim]))
        candidate_starts.append(np.array([-1.0, 1.0, -1.0, 1.0, -1.0][:dim]))

        evaluated_starts = []
        for x0 in candidate_starts:
            if calls[0] >= budget or time.time() > deadline:
                break
            x0 = np.clip(x0, -5, 5)
            fx = f(x0)
            update_best(x0, fx)
            evaluated_starts.append((x0, fx))

        # ---------- Phase 2: paraboloid basin jumps from several restarts ----------
        # Sort starts by fitness; for each of the top few, fit a paraboloid and jump.
        evaluated_starts.sort(key=lambda t: t[1])
        jump_targets = []
        for x0, fx0 in evaluated_starts[:6]:
            if calls[0] >= budget or time.time() > deadline:
                break
            # Fit paraboloid centered at x0 with a moderate step
            x_star, _ = self._fit_paraboloid(x0, 1.0, f, calls, budget, deadline)
            if x_star is not None:
                jump_targets.append(x_star)
            # Also try a larger step to catch the far basin
            x_star2, _ = self._fit_paraboloid(x0, 2.5, f, calls, budget, deadline)
            if x_star2 is not None:
                jump_targets.append(x_star2)

        # Evaluate jump targets and keep the best
        for xt in jump_targets:
            if calls[0] >= budget or time.time() > deadline:
                break
            xt = np.clip(xt, -5, 5)
            fx = f(xt)
            update_best(xt, fx)
            evaluated_starts.append((xt, fx))

        # ---------- Phase 3: Nelder-Mead from best and a few diverse points ----------
        if best_x is not None and calls[0] < budget and time.time() < deadline:
            remaining = budget - calls[0]
            maxfev = min(remaining // 2, 150)
            try:
                res = minimize(f, best_x, method='Nelder-Mead',
                               options={'maxiter': 80, 'maxfev': maxfev,
                                        'xatol': 1e-8, 'fatol': 1e-8})
                if res.fun < best_f:
                    update_best(res.x, res.fun)
            except Exception:
                pass

        # Nelder-Mead from a few diverse points far from the current best
        if calls[0] < budget and time.time() < deadline:
            evaluated_starts.sort(key=lambda t: t[1])
            for x0, fx0 in evaluated_starts[:8]:
                if calls[0] >= budget or time.time() > deadline:
                    break
                if best_x is not None and np.linalg.norm(x0 - best_x) < 0.5:
                    continue
                remaining = budget - calls[0]
                maxfev = min(remaining // 3, 80)
                if maxfev < 15:
                    break
                try:
                    res = minimize(f, x0, method='Nelder-Mead',
                                   options={'maxiter': 40, 'maxfev': maxfev,
                                            'xatol': 1e-8, 'fatol': 1e-8})
                    if res.fun < best_f:
                        update_best(res.x, res.fun)
                except Exception:
                    pass

        # ---------- Phase 4: fine coordinate-descent polish ----------
        if calls[0] < budget and time.time() < deadline and best_x is not None:
            current_x = best_x.copy()
            current_f = best_f
            fine_steps = [1e-2, 5e-3, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4, 5e-5, 2e-5, 1e-5]
            for sigma in fine_steps:
                if calls[0] >= budget or time.time() > deadline:
                    break
                improved = True
                while improved and calls[0] < budget and time.time() < deadline:
                    improved = False
                    for d in range(dim):
                        if calls[0] >= budget or time.time() > deadline:
                            break
                        for sign in (1, -1):
                            if calls[0] >= budget or time.time() > deadline:
                                break
                            new_x = current_x.copy()
                            new_x[d] = np.clip(new_x[d] + sign * sigma, -5, 5)
                            if np.allclose(new_x, current_x, atol=1e-12):
                                continue
                            fx = f(new_x)
                            if fx < current_f:
                                current_x = new_x
                                current_f = fx
                                update_best(current_x, current_f)
                                improved = True

        return best_x, best_f
# EVOLVE-BLOCK-END
