"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Derivative-free trust-region BFGS (DF-TBFGS) for f14."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0

        def time_ok():
            return time.time() < deadline

        n = self.dim
        budget_left = [self.budget]

        def make_eval():
            def evaluate(x):
                budget_left[0] -= 1
                return func(x)
            return evaluate

        eval_func = make_eval()
        rng = np.random.default_rng(42)

        # --- Phase 1: Multi-start coarse search ---
        best_x = None
        best_f = np.inf

        starts = [np.zeros(n)]
        for d in range(n):
            v = np.zeros(n)
            v[d] = 3.0
            starts.append(v)
            v[d] = -3.0
            starts.append(v)
        for _ in range(10):
            starts.append(rng.uniform(-4, 4, n))

        per_start_budget = max(8, self.budget // (len(starts) + 3))

        for s in starts:
            if budget_left[0] < 4 or not time_ok():
                break
            x = np.clip(s.copy(), -5, 5)
            fx = eval_func(x)
            sigma = 1.5
            local_budget = min(per_start_budget, budget_left[0])
            for _ in range(local_budget):
                if not time_ok() or budget_left[0] <= 0:
                    break
                y = np.clip(x + sigma * rng.standard_normal(n), -5, 5)
                fy = eval_func(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                else:
                    sigma *= 0.75
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(n)
            best_f = eval_func(best_x)

        # --- Phase 2: DF-TBFGS local refinement ---
        x = best_x.copy()
        fx = best_f

        def fd_grad(x0):
            g = np.zeros(n)
            h = 1e-5
            for i in range(n):
                xp = x0.copy()
                xp[i] += h
                xp = np.clip(xp, -5, 5)
                fp = eval_func(xp)
                xm = x0.copy()
                xm[i] -= h
                xm = np.clip(xm, -5, 5)
                fm = eval_func(xm)
                g[i] = (fp - fm) / (2 * h)
            return g

        delta = 1.0
        delta_max = 8.0
        delta_min = 1e-6
        H = np.eye(n)
        consecutive_failures = 0

        while budget_left[0] > 2 * n + 2 and time_ok():
            g = fd_grad(x)
            if budget_left[0] <= 0:
                break

            gnorm = np.linalg.norm(g)
            if gnorm < 1e-8:
                break

            d = -H @ g
            dnorm = np.linalg.norm(d)
            if dnorm < 1e-12:
                break
            if dnorm > delta:
                d = d * (delta / dnorm)
                dnorm = delta

            max_step = delta
            for i in range(n):
                if d[i] > 1e-12:
                    max_step = min(max_step, (5.0 - x[i]) / d[i])
                elif d[i] < -1e-12:
                    max_step = min(max_step, (-5.0 - x[i]) / d[i])
            if max_step <= 1e-12:
                max_step = 1e-8
            if max_step < dnorm:
                d = d * (max_step / dnorm)
                dnorm = max_step

            x_trial = np.clip(x + d, -5, 5)
            f_trial = eval_func(x_trial)

            if budget_left[0] <= 0:
                if f_trial < fx:
                    x, fx = x_trial, f_trial
                break

            if f_trial < fx:
                x_new = x_trial
                f_new = f_trial

                s = x_new - x
                g_new = fd_grad(x_new)
                if budget_left[0] <= 0:
                    x, fx = x_new, f_new
                    break
                y = g_new - g

                sy = s @ y
                if sy > 1e-12:
                    Hy = H @ y
                    yHy = y @ Hy
                    H = H + (np.outer(s, s) / sy) * (1 + yHy / sy) - (np.outer(Hy, Hy) / sy)

                x, fx = x_new, f_new
                consecutive_failures = 0
                delta = min(2.0 * delta, delta_max)
            else:
                consecutive_failures += 1
                delta = max(0.5 * delta, delta_min)

                if consecutive_failures > 5:
                    perturbation = rng.normal(0, 0.5, n)
                    x_restart = np.clip(x + perturbation, -5, 5)
                    f_restart = eval_func(x_restart)
                    if budget_left[0] <= 0:
                        if f_restart < fx:
                            x, fx = x_restart, f_restart
                        break
                    if f_restart < fx:
                        x, fx = x_restart, f_restart
                        delta = 0.5
                        H = np.eye(n)
                        consecutive_failures = 0

        # --- Phase 3: Final polish ---
        if budget_left[0] > 0 and time_ok():
            sigma = 1e-4
            while budget_left[0] > 0 and time_ok():
                y = np.clip(x + sigma * rng.standard_normal(n), -5, 5)
                fy = eval_func(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 0.8
                else:
                    sigma *= 0.7

        return x, fx
# EVOLVE-BLOCK-END
