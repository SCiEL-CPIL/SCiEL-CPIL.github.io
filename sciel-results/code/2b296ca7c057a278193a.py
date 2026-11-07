"""Optimizer for BBOB-style Sphere f1 on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time

try:
    import cma
    _HAS_CMA = True
except Exception:
    _HAS_CMA = False


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290.0  # safety margin before 300s kill

        rng = np.random.default_rng(12345)
        dim = self.dim
        budget = self.budget
        calls = 0

        def clip(v):
            return np.clip(v, -5.0, 5.0)

        # ---- Phase 1: coarse grid search to locate the shifted minimizer ----
        # Sphere is smooth and convex; a coarse grid quickly finds the region.
        grid_vals = np.linspace(-5.0, 5.0, 7)  # 7^5 = 16801, too many -> use 5^5=3125
        grid_vals = np.linspace(-5.0, 5.0, 5)
        best_x = None
        best_f = np.inf
        # iterate grid in a way that spends calls; stop at budget/3
        grid_budget = min(len(grid_vals) ** dim, budget // 3)
        for gx in np.ndindex(*([len(grid_vals)] * dim)):
            if calls >= grid_budget:
                break
            if time.time() > deadline:
                break
            x = np.array([grid_vals[i] for i in gx], dtype=float)
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # ---- Phase 2: (1+1)-ES local refinement from the best grid point ----
        x = best_x.copy()
        fx = best_f
        sigma = 1.0
        while calls < budget:
            if time.time() > deadline:
                break
            y = clip(x + sigma * rng.standard_normal(dim))
            fy = func(y)
            calls += 1
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.3
            else:
                sigma *= 0.8
            if sigma < 1e-6:
                break

        # ---- Phase 3: CMA-ES final polish if available and time remains ----
        if _HAS_CMA and calls < budget and time.time() < deadline - 5.0:
            try:
                remaining = budget - calls
                # CMA-ES with limited budget; use the current best as start
                x0 = x.copy()
                # initial sigma estimate
                s0 = max(sigma, 0.1)
                opts = {'bounds': [-5.0] * dim, 'maxiter': remaining,
                        'seed': 0, 'verbose': -9, 'tolopt': 1e-10,
                        'tolx': 1e-10, 'tolfun': 1e-12,
                        'save_history': False, 'CMA_verb': -1000000}
                es = cma.CMAEvolutionStrategy(x0, s0, opts)
                while not es.stop() and calls < budget:
                    if time.time() > deadline - 2.0:
                        break
                    solutions = es.ask()
                    for sol in solutions:
                        if calls >= budget:
                            break
                        sol_c = clip(sol)
                        fval = func(sol_c)
                        calls += 1
                        es.tell(sol, fval)
                # get best
                cma_best = es.result.xbest
                cma_best = clip(cma_best)
                cma_f = func(cma_best)
                calls += 1
                if cma_f < fx:
                    x, fx = cma_best, cma_f
            except Exception:
                pass

        return x, fx
# EVOLVE-BLOCK-END
