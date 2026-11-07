"""Multi-start Nelder-Mead with aggressive early sampling for BBOB f22 (Gallagher 21 peaks)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math


class Optimizer:
    """Multi-start optimizer with aggressive early sampling and Nelder-Mead refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        dim = self.dim
        budget = self.budget
        calls = 0
        best_f = float('inf')
        best_x = np.zeros(dim)

        def eval_and_track(x):
            nonlocal calls, best_f, best_x
            if calls >= budget or time.time() > deadline:
                return best_f
            calls += 1
            try:
                val = func(np.asarray(x, dtype=float))
                if not np.isfinite(val):
                    val = best_f
                if val < best_f:
                    best_f = val
                    best_x = np.asarray(x, dtype=float).copy()
                return val
            except Exception:
                return best_f

        rng = np.random.default_rng(7)

        # Phase 1: Aggressive random sampling for early AOCC progress
        # Spend ~40% of budget on random points to establish good initial best_f
        n_random = max(15, int(budget * 0.4))
        candidates = []

        for i in range(n_random):
            if calls >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, size=dim)
            f_val = eval_and_track(x)
            candidates.append((f_val, x.copy()))

        # Sort by fitness
        candidates.sort(key=lambda t: t[0])

        # Phase 2: Nelder-Mead refinement on top candidates
        top_k = min(6, len(candidates))

        for i in range(top_k):
            if calls >= budget or time.time() > deadline:
                break

            x0 = candidates[i][1].copy()
            remaining = budget - calls
            if remaining < 20:
                break

            # Allocate budget: earlier candidates get more budget
            cma_budget = max(30, remaining // (top_k - i + 1) * 2)
            cma_budget = min(cma_budget, remaining)

            try:
                from scipy.optimize import minimize
                res = minimize(
                    eval_and_track,
                    x0,
                    method='Nelder-Mead',
                    options={
                        'maxiter': max(50, cma_budget),
                        'maxfev': cma_budget,
                        'xatol': 1e-7,
                        'fatol': 1e-9,
                        'adaptive': True,
                    },
                    bounds=[(-5.0, 5.0)] * dim,
                )
            except Exception:
                pass

        # Phase 3: Adaptive local search around the best point
        if calls < budget and time.time() < deadline:
            step = 0.3
            min_step = 0.01
            max_step = 1.5
            patience = 0
            max_patience = 30

            while calls < budget and time.time() < deadline:
                # Generate candidate around best point
                x = best_x + rng.normal(0, step, size=dim)
                x = np.clip(x, -5, 5)
                f_val = eval_and_track(x)

                if f_val < best_f:
                    patience = 0
                    step = max(min_step, step * 0.85)
                else:
                    patience += 1
                    if patience > max_patience:
                        # Restart with larger step
                        step = min(max_step, step * 1.4)
                        patience = 0
                        # Occasionally jump to a random point for exploration
                        if rng.random() < 0.15:
                            x = rng.uniform(-5, 5, size=dim)
                            f_val = eval_and_track(x)
                            if f_val < best_f:
                                step = 0.3

        return best_x, best_f
# EVOLVE-BLOCK-END
