"""Combined optimizer for BBOB f22 (Gallagher 21 peaks) with stratified exploration and L-BFGS-B refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Stratified exploration + L-BFGS-B basin refinement for BBOB f22."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit

        dim = self.dim
        budget = self.budget
        calls = 0
        best_f = float('inf')
        best_x = np.zeros(dim)
        rng = np.random.default_rng(12345)

        def eval_and_track(x):
            nonlocal calls, best_f, best_x
            if calls >= budget or time.time() > deadline:
                return best_f
            calls += 1
            try:
                x_clipped = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
                val = float(func(x_clipped))
                if val < best_f:
                    best_f = val
                    best_x = x_clipped.copy()
                return val
            except Exception:
                return best_f

        def time_ok():
            return time.time() < deadline and calls < budget

        def budget_ok():
            return calls < budget

        def lbfgsb(x0, maxiter):
            """Run L-BFGS-B from x0 with an early-termination guard."""
            if not time_ok():
                return
            x0 = np.clip(np.asarray(x0, dtype=float), -5.0, 5.0)
            state = {'stop': False}

            def obj(x):
                if state['stop']:
                    return best_f
                v = eval_and_track(x)
                if not time_ok() or not budget_ok():
                    state['stop'] = True
                return v

            try:
                minimize(
                    obj,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={'maxiter': maxiter, 'maxls': 40, 'gtol': 1e-4},
                )
            except Exception:
                pass

        # Phase 1: Broad exploration to locate promising basins.
        # f22 has 21 peaks; we need a diverse set of starting points so at
        # least one lands in (or near) the global basin. Use stratified
        # sampling (2 bins per dim -> 32 strata) plus structured points.
        n_strata = 2 ** dim
        n_phase1 = min(120, budget)
        points_per_stratum = max(1, n_phase1 // n_strata)

        initial_points = []
        for s in range(n_strata):
            bins = []
            temp = s
            for d in range(dim):
                bins.append(temp % 2)
                temp //= 2
            bins = np.array(bins)
            for _ in range(points_per_stratum):
                pt = np.zeros(dim)
                for d in range(dim):
                    if bins[d] == 0:
                        pt[d] = rng.uniform(-5.0, 0.0)
                    else:
                        pt[d] = rng.uniform(0.0, 5.0)
                initial_points.append(pt)

        # Center + axis-aligned structured points
        initial_points.append(np.zeros(dim))
        for d in range(dim):
            for v in [-3.5, -1.75, 0.0, 1.75, 3.5]:
                pt = np.zeros(dim)
                pt[d] = v
                initial_points.append(pt)

        for x in initial_points:
            if not time_ok():
                break
            eval_and_track(x)

        if not time_ok():
            return best_x, best_f

        # Phase 2: Rapid local refinement with L-BFGS-B.
        # This is where early AOCC progress comes from: L-BFGS-B exploits
        # local smoothness to descend fast once inside a basin, far cheaper
        # per call than coordinate sweeps.
        lbfgsb(best_x.copy(), maxiter=200)

        if not time_ok():
            return best_x, best_f

        # Phase 3: Multi-start refinement around the best basin with
        # diverse, progressively tighter restarts to escape shallow local
        # minima and probe neighbouring basins.
        sigmas = [1.5, 1.0, 0.7, 0.5, 0.35, 0.25, 1.2, 0.8, 0.6, 0.4]
        n_starts = len(sigmas)
        for i in range(n_starts):
            if not time_ok():
                break
            sigma = sigmas[i]
            if i % 2 == 0:
                x0 = best_x + rng.normal(0.0, sigma, size=dim)
            else:
                x0 = best_x + rng.uniform(-sigma, sigma, size=dim)
            x0 = np.clip(x0, -5.0, 5.0)
            lbfgsb(x0, maxiter=120)

        if not time_ok():
            return best_x, best_f

        # Phase 4: Final polish on the global best.
        lbfgsb(best_x.copy(), maxiter=150)

        return best_x, best_f
# EVOLVE-BLOCK-END
