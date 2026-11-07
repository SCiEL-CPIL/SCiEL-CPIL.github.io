"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid optimizer: (1+1)-ES with adaptive step size and local Hessian Newton polishing."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # safety margin before 300s limit
        rng = np.random.default_rng(12345)

        def clip(x):
            return np.clip(x, -5.0, 5.0)

        # --- Phase 1: (1+1)-ES with 1/5 success rule ---
        x = rng.uniform(-5, 5, dim)
        fx = func(x)
        calls_made = 1
        x_best = x.copy()
        f_best = fx
        sigma = 2.0

        # Run (1+1)-ES for a portion of the budget to find a good basin
        es_budget = int(budget * 0.6)  # use 60% of budget for ES
        max_es_iter = es_budget - 1

        for i in range(max_es_iter):
            if calls_made >= budget or time.time() >= deadline:
                break

            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls_made += 1

            if fy < fx:
                x, fx = y, fy
                sigma *= 1.5
                if fx < f_best:
                    f_best = fx
                    x_best = x.copy()
            else:
                sigma *= 0.87

        # --- Phase 2: Local Hessian Newton steps to polish the solution ---
        # This exploits the quadratic structure of Discus
        calls_left = budget - calls_made
        if calls_left > 10 and time.time() < deadline:
            x_curr = x_best.copy()
            f_curr = f_best

            # Adaptive Hessian fitting
            # Estimate gradient and Hessian using finite differences
            h = 0.05

            for newton_round in range(5):
                if calls_left <= 2 * dim + dim * (dim + 1) // 2 or time.time() >= deadline:
                    break

                # Estimate gradient
                grad = np.zeros(dim)
                for i in range(dim):
                    xp = x_curr.copy()
                    xp[i] += h
                    xm = x_curr.copy()
                    xm[i] -= h
                    xp = clip(xp)
                    xm = clip(xm)
                    fpx = func(xp)
                    fxm = func(xm)
                    calls_made += 2
                    calls_left -= 2
                    grad[i] = (fpx - fxm) / (2 * h)

                if calls_left <= dim * (dim + 1) // 2 or time.time() >= deadline:
                    break

                # Check if gradient is small enough
                if np.linalg.norm(grad) < 1e-10:
                    break

                # Estimate Hessian
                H = np.zeros((dim, dim))
                for i in range(dim):
                    for j in range(i, dim):
                        if i == j:
                            xp = x_curr.copy()
                            xp[i] += h
                            xm = x_curr.copy()
                            xm[i] -= h
                            xp = clip(xp)
                            xm = clip(xm)
                            fpp = func(xp)
                            fxm = func(xm)
                            calls_made += 2
                            calls_left -= 2
                            H[i, i] = (fpp - 2 * f_curr + fxm) / (h * h)
                        else:
                            xpp = x_curr.copy()
                            xpp[i] += h
                            xpp[j] += h
                            xpm = x_curr.copy()
                            xpm[i] += h
                            xpm[j] -= h
                            xmp = x_curr.copy()
                            xmp[i] -= h
                            xmp[j] += h
                            xmm = x_curr.copy()
                            xmm[i] -= h
                            xmm[j] -= h
                            xpp = clip(xpp)
                            xpm = clip(xpm)
                            xmp = clip(xmp)
                            xmm = clip(xmm)
                            fpp = func(xpp)
                            fpm = func(xpm)
                            fmp = func(xmp)
                            fxm = func(xmm)
                            calls_made += 4
                            calls_left -= 4
                            H[i, j] = (fpp - fpm - fmp + fxm) / (4 * h * h)
                            H[j, i] = H[i, j]

                if calls_left <= 0 or time.time() >= deadline:
                    break

                # Add small regularization for numerical stability
                H_reg = H + 1e-4 * np.eye(dim)

                # Solve for Newton step
                try:
                    step_newton = np.linalg.solve(H_reg, grad)
                except np.linalg.LinAlgError:
                    step_newton = -grad * 0.1

                # Line search to ensure descent
                alpha_ls = 1.0
                improved = False
                for ls_iter in range(10):
                    if calls_left <= 0 or time.time() >= deadline:
                        break

                    x_trial = clip(x_curr - alpha_ls * step_newton)
                    f_trial = func(x_trial)
                    calls_made += 1
                    calls_left -= 1

                    if f_trial < f_curr:
                        x_curr = x_trial
                        f_curr = f_trial
                        improved = True
                        if f_curr < f_best:
                            f_best = f_curr
                            x_best = x_curr.copy()
                        break
                    else:
                        alpha_ls *= 0.5

                if not improved:
                    break

        return x_best, f_best
# EVOLVE-BLOCK-END
