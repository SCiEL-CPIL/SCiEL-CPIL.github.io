"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart adaptive (1+1)-ES portfolio + Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def _adaptive_es(self, func, rng, x0, fx0, n_evals, deadline, global_best):
        """Run an adaptive (1+1)-ES trajectory with 1/5 success-rate adaptation."""
        x = x0.copy()
        fx = fx0
        sigma = 1.6
        d = self.dim
        successes = 0
        window = 0
        for _ in range(n_evals):
            if time.time() >= deadline:
                break
            y = x + sigma * rng.standard_normal(d)
            y = np.clip(y, -5.0, 5.0)
            fy = func(y)
            if fy < fx:
                x = y
                fx = fy
                successes += 1
                window += 1
                if window >= 20:
                    sr = successes / window
                    # 1/5 success rule: push sigma up if too many successes, down if too few
                    if sr > 0.2:
                        sigma *= 1.25
                    elif sr < 0.2:
                        sigma *= 0.8
                    successes = 0
                    window = 0
            else:
                window += 1
                if window >= 20:
                    sr = successes / window
                    if sr > 0.2:
                        sigma *= 1.25
                    elif sr < 0.2:
                        sigma *= 0.8
                    successes = 0
                    window = 0
            if fx < global_best[0]:
                global_best[0] = fx
                global_best[1] = x.copy()
        return x, fx

    def _nelder_mead(self, func, x0, fx0, n_evals, deadline, global_best):
        """Deterministic Nelder-Mead local polish on the incumbent."""
        d = self.dim
        # build initial simplex
        simplex = [x0.copy() + fx0]
        base_step = 0.5
        for i in range(d):
            v = x0.copy()
            v[i] += base_step
            simplex.append(v + func(np.clip(v, -5.0, 5.0)))
        # sort
        simplex.sort(key=lambda p: p[-1])
        centroid = np.mean([p[:-1] for p in simplex], axis=0)
        for _ in range(n_evals):
            if time.time() >= deadline:
                break
            # reflection
            xr = centroid + 1.0 * (centroid - simplex[0][:-1])
            xr = np.clip(xr, -5.0, 5.0)
            fxr = func(xr)
            if simplex[0][-1] <= fxr < simplex[-1][-1]:
                simplex[-1] = xr + fxr
            elif fxr < simplex[0][-1]:
                # expansion
                xe = centroid + 2.0 * (centroid - simplex[0][:-1])
                xe = np.clip(xe, -5.0, 5.0)
                fxe = func(xe)
                if fxe < fxr:
                    simplex[-1] = xe + fxe
                else:
                    simplex[-1] = xr + fxr
            else:
                # contraction
                if fxr < simplex[-1][-1]:
                    xc = centroid + 0.5 * (simplex[0][:-1] - centroid)
                    fxc = func(np.clip(xc, -5.0, 5.0))
                    if fxc <= fxr:
                        simplex[-1] = xc + fxc
                    else:
                        # shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = (simplex[0][:-1] + 0.5 * (simplex[i][:-1] - simplex[0][:-1])) + \
                                          func(np.clip(simplex[0][:-1] + 0.5 * (simplex[i][:-1] - simplex[0][:-1]), -5.0, 5.0))
                else:
                    # shrink
                    for i in range(1, len(simplex)):
                        simplex[i] = (simplex[0][:-1] + 0.5 * (simplex[i][:-1] - simplex[0][:-1])) + \
                                      func(np.clip(simplex[0][:-1] + 0.5 * (simplex[i][:-1] - simplex[0][:-1]), -5.0, 5.0))
            simplex.sort(key=lambda p: p[-1])
            centroid = np.mean([p[:-1] for p in simplex], axis=0)
            if simplex[0][-1] < global_best[0]:
                global_best[0] = simplex[0][-1]
                global_best[1] = simplex[0][:-1].copy()
        return simplex[0][:-1], simplex[0][-1]

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin before 300s limit
        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(0)

        global_best = [float('inf'), None]

        # Reserve a small budget for Nelder-Mead polish
        nm_budget = d + 1  # simplex construction + a few iterations
        es_budget = budget - nm_budget - 1  # -1 for initial evaluation
        n_restarts = 10
        per_restart = max(1, es_budget // n_restarts)

        # Initial evaluation
        x0 = rng.uniform(-5.0, 5.0, d)
        fx0 = func(x0)
        global_best[0] = fx0
        global_best[1] = x0.copy()

        # Run portfolio of restarts
        for r in range(n_restarts):
            if time.time() >= deadline:
                break
            # New starting point: mix of random and perturbation of best
            if r == 0:
                xs = x0.copy()
            elif r % 2 == 0:
                xs = rng.uniform(-5.0, 5.0, d)
            else:
                # Perturb the current best
                xs = global_best[1] + 2.0 * rng.standard_normal(d)
                xs = np.clip(xs, -5.0, 5.0)
            fx_s = func(xs)
            if fx_s < global_best[0]:
                global_best[0] = fx_s
                global_best[1] = xs.copy()
            self._adaptive_es(func, rng, xs, fx_s, per_restart, deadline, global_best)

        # Nelder-Mead polish on the best incumbent
        if global_best[1] is not None and time.time() < deadline:
            self._nelder_mead(func, global_best[1].copy(), global_best[0], nm_budget, deadline, global_best)

        return global_best[1], global_best[0]
# EVOLVE-BLOCK-END
