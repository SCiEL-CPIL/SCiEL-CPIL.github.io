"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """LHS global search + L-BFGS-B local exploitation for Lunacek bi-Rastrigin."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 270  # safety margin before 300s limit

        global_best = None
        global_fx = np.inf
        calls_made = 0

        def evaluate(x):
            nonlocal calls_made, global_best, global_fx
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            fx = float(func(x))
            calls_made += 1
            if fx < global_fx:
                global_fx = fx
                global_best = x.copy()
            return fx

        rng = np.random.default_rng(12345)

        # Phase 1: Dense Latin Hypercube Sampling to find the best basin
        n_lhs = 150
        lhs_points = np.zeros((n_lhs, dim))
        for d in range(dim):
            edges = np.linspace(-4.0, 4.0, n_lhs + 1)
            for i in range(n_lhs):
                lhs_points[i, d] = rng.uniform(edges[i], edges[i+1])
        perm = rng.permutation(n_lhs)
        lhs_points = lhs_points[perm]

        for i in range(n_lhs):
            if calls_made >= budget - 100 or time.time() > deadline - 30:
                break
            evaluate(lhs_points[i])

        # Phase 2: L-BFGS-B polish from the best point
        if global_best is not None and calls_made < budget - 20 and time.time() < deadline - 20:
            x0 = global_best.copy()
            remaining = budget - calls_made

            # L-BFGS-B with finite differences (default)
            # maxfun controls max function evaluations in scipy
            maxfun = min(remaining - 10, 300)
            try:
                result = minimize(
                    evaluate,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={'maxfun': maxfun, 'maxiter': 100}
                )
            except Exception:
                pass

        # Phase 3: If budget remains, do a second LHS round with different seed
        if calls_made < budget - 50 and time.time() < deadline - 20:
            rng2 = np.random.default_rng(54321)
            n_lhs2 = min(80, budget - calls_made - 30)
            if n_lhs2 > 10:
                lhs_points2 = np.zeros((n_lhs2, dim))
                for d in range(dim):
                    edges = np.linspace(-4.0, 4.0, n_lhs2 + 1)
                    for i in range(n_lhs2):
                        lhs_points2[i, d] = rng2.uniform(edges[i], edges[i+1])
                perm2 = rng2.permutation(n_lhs2)
                lhs_points2 = lhs_points2[perm2]

                for i in range(n_lhs2):
                    if calls_made >= budget - 30 or time.time() > deadline - 15:
                        break
                    evaluate(lhs_points2[i])

                # L-BFGS-B polish from new best if improved
                if global_best is not None and calls_made < budget - 20 and time.time() < deadline - 15:
                    x0 = global_best.copy()
                    remaining = budget - calls_made
                    maxfun = min(remaining - 5, 200)
                    try:
                        result = minimize(
                            evaluate,
                            x0,
                            method='L-BFGS-B',
                            bounds=[(-5.0, 5.0)] * dim,
                            options={'maxfun': maxfun, 'maxiter': 80}
                        )
                    except Exception:
                        pass

        return global_best, global_fx
# EVOLVE-BLOCK-END
