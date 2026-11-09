"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Coarse-to-fine global grid + adaptive accelerated gradient + restart escape."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 270  # 30s safety margin before 300s limit
        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(0)

        calls = 0
        best_fx = np.inf
        best_x = np.zeros(d)

        def eval(x):
            nonlocal calls, best_fx, best_x
            fx = func(np.asarray(x, dtype=float))
            calls += 1
            if fx < best_fx:
                best_fx = fx
                best_x = np.asarray(x, dtype=float).copy()
            return fx

        # ---- Phase 1: coarse-to-fine global grid (guarantees early monotonic progress) ----
        base = [np.array([-4.0, -2.0, 0.0, 2.0, 4.0])]
        for k in range(1, 4):
            lo, hi = base[0][0], base[0][-1]
            step = (hi - lo) / (2 * k - 1)
            base.append(np.linspace(lo, hi, 2 * k))
        for axis in range(1, d):
            new = []
            for a in base[axis - 1]:
                for b in base[0]:
                    v = a.copy()
                    v[axis] = b
                    new.append(v)
            base = new
        for v in base:
            if calls >= budget:
                break
            if time.time() > deadline:
                break
            eval(v)

        x = best_x.copy()
        fx = best_fx

        # ---- Phase 2: adaptive accelerated (Nesterov) gradient descent ----
        # Step size auto-adapts via backtracking line search (no tuned constant needed).
        eps = 1e-5
        alpha = 1.0
        momentum = 0.9
        t = 1.0
        v = np.zeros(d)
        x_old = x.copy()
        gamma = 1.0
        gamma_max = 5.0
        gamma_min = 0.5
        stall = 0
        stall_max = 140

        while calls < budget and time.time() <= deadline:
            y = x + gamma * v
            fy = eval(y)
            if fy > fx + 1e-12:
                gamma *= gamma_min
                if gamma < 1e-3:
                    gamma = 1.0
                continue

            g = np.zeros(d)
            for i in range(d):
                yp = y.copy(); yp[i] += eps
                ym = y.copy(); ym[i] -= eps
                g[i] = (eval(yp) - eval(ym)) / (2.0 * eps)
            if calls >= budget or time.time() > deadline:
                break

            t_new = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
            beta = (t_new - 1.0) / t_new
            v = beta * v + (1.0 - beta) * (y - x_old)
            x_old = x.copy()

            # adaptive step: grow if the step was accepted cleanly, shrink via line search
            step = alpha * 1.05
            if step > gamma_max:
                step = gamma_max
            z = y
            fz = fy
            for _ in range(12):
                zc = z + step * g
                fzc = eval(zc)
                if fzc < fz - 1e-10:
                    break
                step *= 0.5
                if step < 1e-6:
                    break
            if fzc < fz:
                z, fz = zc, fzc
                alpha = step
                if alpha > gamma_max:
                    alpha = gamma_max
                stall = 0
            else:
                alpha = max(step, 1e-6)
                stall += 1

            x = z
            fx = fz
            if fx < best_fx - 1e-12:
                best_x = x.copy()

            # ---- Phase 3: coarse random-restart escape (keeps global best, never regresses) ----
            if stall >= stall_max:
                stall = 0
                alpha = 1.0
                momentum = 0.9
                t = 1.0
                v = np.zeros(d)
                x_old = x.copy()
                gamma = 1.0
                for _ in range(3):
                    if calls >= budget or time.time() > deadline:
                        break
                    cand = np.clip(best_x + 2.5 * rng.standard_normal(d), -5.0, 5.0)
                    fc = eval(cand)
                    if fc < fx:
                        x = cand
                        fx = fc
                        x_old = x.copy()
                        v = np.zeros(d)
                        t = 1.0

        return best_x, best_fx
# EVOLVE-BLOCK-END
