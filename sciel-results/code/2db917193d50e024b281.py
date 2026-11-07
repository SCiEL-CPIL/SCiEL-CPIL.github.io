"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Linear-aware optimizer: estimate gradient, march to best corner, then refine."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()

        def time_ok():
            return (time.time() - start_time) < 285.0

        # ---- Phase 1: gradient estimation via boundary probes ----
        # Probe at a central-ish point and at +/- h along each axis.
        # Using large h (near boundary) maximizes signal-to-noise for linear f.
        x0 = np.zeros(dim)
        f0 = func(x0)
        calls = 1

        h = 4.0  # near boundary for max signal
        grad = np.zeros(dim)
        for i in range(dim):
            xp = x0.copy(); xp[i] += h
            xm = x0.copy(); xm[i] -= h
            f_p = func(xp)
            f_m = func(xm)
            calls += 2
            grad[i] = (f_p - f_m) / (2.0 * h)

        # ---- Phase 2: march to the optimal corner ----
        # For a linear function, min is at the corner where each coord
        # goes opposite to the gradient sign.
        best_x = np.zeros(dim)
        for i in range(dim):
            if grad[i] > 0:
                best_x[i] = -5.0
            else:
                best_x[i] = 5.0
        best_f = func(best_x)
        calls += 1

        # Verify with neighbor corners (guard against sign errors)
        # Check corners that differ by one coordinate
        for i in range(dim):
            if not time_ok():
                break
            alt = best_x.copy()
            alt[i] = -alt[i]  # flip to the other boundary
            f_alt = func(alt)
            calls += 1
            if f_alt < best_f:
                best_f = f_alt
                best_x = alt

        # ---- Phase 3: polish with fine local search ----
        # Use a small-step random walk / coordinate descent to refine.
        sigma = 0.5
        for _ in range(budget - calls):
            if not time_ok():
                break
            # Coordinate descent: try moving each coordinate toward interior
            improved = False
            for i in range(dim):
                if not time_ok():
                    break
                # Try a few step sizes
                for step in (sigma, sigma * 0.5, sigma * 0.25):
                    trial = best_x.copy()
                    trial[i] = np.clip(best_x[i] - step * np.sign(grad[i]), -5, 5)
                    f_trial = func(trial)
                    calls += 1
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial
                        improved = True
                        break
                if improved:
                    break
            if improved:
                sigma = min(sigma * 1.2, 5.0)
            else:
                sigma *= 0.7
                if sigma < 1e-4:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
