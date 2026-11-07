"""Rewritten optimizer: adaptive per-dimension line search (coordinate descent)
with periodic global perturbations and an explicit wall-clock deadline guard."""
import time
import numpy as np


class Optimizer:
    """Adaptive per-dimension line-search (coordinate descent) for Schwefel f20.

    Strategy:
      * Each axis has its own step size and direction.
      * A small geometric line search per axis reduces wasted evaluations and
        enables fine convergence near the optimum.
      * Periodic global perturbations (full-space step) prevent getting trapped
        in local minima of the rotated landscape.
      * An explicit deadline guard returns the best result with a safety margin.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        start = time.time()
        deadline = start + 295.0  # safety margin under the 300 s wall-clock limit

        # ---- initial point ----
        x = rng.uniform(-5, 5, self.dim)
        fx = func(x)
        best_x, best_f = x.copy(), fx
        calls = 1

        # ---- per-dimension state ----
        steps = np.full(self.dim, 1.0)
        dirs = np.ones(self.dim)

        # line-search parameters
        base_ratio = 0.5
        ls_steps = 4  # number of geometric shrink steps per axis
        min_step = 1e-5

        # periodic global perturbation every N coordinate sweeps
        pert_every = 15
        pert_sigma = 1.5

        sweep = 0
        no_improve = 0

        while calls < self.budget and time.time() < deadline:
            # ---- periodic global perturbation (escape local minima) ----
            if sweep % pert_every == 0 and sweep > 0:
                cand = np.clip(x + pert_sigma * rng.standard_normal(self.dim), -5, 5)
                cf = func(cand)
                calls += 1
                if cf < fx:
                    x, fx = cand, cf
                    if fx < best_f:
                        best_x, best_f = x.copy(), fx
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    no_improve += 1
                    # dampen perturbation if it keeps failing
                    pert_sigma *= 0.8

            # ---- coordinate descent over each axis ----
            for d in range(self.dim):
                if calls >= self.budget or time.time() >= deadline:
                    break
                step = steps[d]
                direction = dirs[d]
                improved_this_axis = False

                # geometric line search: try shrinking steps
                for k in range(ls_steps):
                    cand = x.copy()
                    cand[d] = np.clip(x[d] + direction * step, -5, 5)
                    cf = func(cand)
                    calls += 1
                    if cf < fx:
                        x, fx = cand, cf
                        if fx < best_f:
                            best_x, best_f = x.copy(), fx
                            no_improve = 0
                        else:
                            no_improve += 1
                        improved_this_axis = True
                        break
                    else:
                        step *= base_ratio
                        # reverse direction on failure to explore other side
                        if k == ls_steps - 1:
                            dirs[d] = -dirs[d]

                if improved_this_axis:
                    steps[d] = min(step * 1.3, 5.0)  # grow step after success
                else:
                    steps[d] = max(step, min_step)  # keep at least min_step

            sweep += 1

        return best_x, best_f
