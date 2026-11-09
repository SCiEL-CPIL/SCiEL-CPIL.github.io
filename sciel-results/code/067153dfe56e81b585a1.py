"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start (1+1)-ES with adaptive allocation for Rastrigin."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 280s safety margin from 330s limit
        rng = np.random.default_rng(42)

        dim = self.dim
        total_budget = self.budget
        calls_made = 0
        best_val = np.inf
        best_x = None

        # Phase 1: Diverse initial sampling using Latin Hypercube
        n_init = 40
        if total_budget <= n_init + 10:
            n_init = max(5, total_budget // 2)

        pts = np.zeros((n_init, dim))
        for d in range(dim):
            perm = rng.permutation(n_init)
            pts[:, d] = (perm + rng.uniform(size=n_init)) / n_init
            pts[:, d] = pts[:, d] * 10.0 - 5.0

        init_results = []
        for i in range(n_init):
            if time.time() > deadline:
                break
            if calls_made >= total_budget:
                break
            val = func(pts[i])
            calls_made += 1
            init_results.append((val, pts[i].copy()))
            if val < best_val:
                best_val = val
                best_x = pts[i].copy()

        if calls_made >= total_budget or not init_results:
            return best_x, best_val

        # Sort by value, pick top candidates for refinement
        init_results.sort(key=lambda x: x[0])
        n_top = min(4, len(init_results))
        top_starts = [r[1] for r in init_results[:n_top]]
        top_vals = [r[0] for r in init_results[:n_top]]

        # Phase 2: Allocate remaining budget to top starts
        remaining = total_budget - calls_made
        if remaining <= 0:
            return best_x, best_val

        # Weighted allocation: better starts get more budget
        weights = np.array([1.0 / (i + 1) for i in range(n_top)])
        weights /= weights.sum()
        budgets = np.maximum(10, (remaining * weights).astype(int))
        diff = remaining - budgets.sum()
        budgets[0] += diff
        budgets = np.maximum(0, budgets)

        # Run (1+1)-ES on each top start
        for idx in range(n_top):
            if time.time() > deadline:
                break
            if calls_made >= total_budget:
                break
            if budgets[idx] <= 0:
                continue

            x = top_starts[idx].copy()
            fx = top_vals[idx]
            sigma = 1.5
            budget_i = budgets[idx]

            for _ in range(budget_i):
                if time.time() > deadline:
                    break
                if calls_made >= total_budget:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_made += 1

                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.3
                else:
                    sigma *= 0.85

                if fx < best_val:
                    best_val = fx
                    best_x = x.copy()

                if sigma < 1e-4:
                    break

        return best_x, best_val
# EVOLVE-BLOCK-END
