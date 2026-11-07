"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Linear function minimizer: determine gradient signs via bidirectional probing, then go to optimal vertex."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        dim = self.dim
        budget = self.budget

        best_x = np.zeros(dim)
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_x, best_f = x.copy(), f

        # Phase 1: Probe at origin and at ±1*e_i for each axis (2*dim + 1 evals)
        # Every probe updates best-so-far, improving early AOCC.
        x0 = np.zeros(dim)
        f0 = func(x0)
        update_best(x0, f0)

        signs = np.zeros(dim)
        for i in range(dim):
            if time.time() > deadline:
                break
            xp = np.zeros(dim)
            xp[i] = 1.0
            fp = func(xp)
            update_best(xp, fp)

            xm = np.zeros(dim)
            xm[i] = -1.0
            fm = func(xm)
            update_best(xm, fm)

            # If f(-e_i) < f(+e_i), gradient is positive -> minimum at x_i = -5
            if fm < fp:
                signs[i] = 1.0
            else:
                signs[i] = -1.0

        # Phase 2: Jump to the optimal vertex
        x_opt = -5.0 * signs
        f_opt = func(x_opt)
        update_best(x_opt, f_opt)

        # Phase 3: Local refinement with remaining budget
        remaining = budget - (2 * dim + 2)
        if remaining > 0 and time.time() < deadline:
            rng = np.random.default_rng(42)
            sigma = 0.5
            for _ in range(min(remaining, 200)):
                if time.time() > deadline:
                    break
                y = np.clip(best_x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                if fy < best_f:
                    best_x, best_f = y, fy
                    sigma *= 1.1
                else:
                    sigma *= 0.85

        return best_x, best_f
# EVOLVE-BLOCK-END
