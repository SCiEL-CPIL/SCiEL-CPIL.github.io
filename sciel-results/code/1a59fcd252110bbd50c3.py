"""Rewritten optimizer for BBOB f21 (Gallagher 101 peaks)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math


class Optimizer:
    """Global-scan then multi-restart (1+1)-ES with adaptive sigma and deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget

        # Phase 1: Global scan – mixed stratified + random sampling
        n_scan = min(30, budget // 5)
        n_refine_candidates = 3

        # Generate stratified grid points (3^5 = 243 cells)
        n_grid = 3
        grid_points = []
        for i in range(n_grid ** dim):
            coords = []
            tmp = i
            for d in range(dim):
                coords.append(tmp % n_grid)
                tmp //= n_grid
            x = np.array([(-4 + 8 * (c + 0.5) / n_grid) + rng.uniform(-0.4, 0.4) * 8 / n_grid for c in coords])
            x = np.clip(x, -5, 5)
            grid_points.append(x)

        # Mix: 70% stratified, 30% uniform random
        n_strat = int(0.7 * n_scan)
        n_rand = n_scan - n_strat
        idx = rng.permutation(len(grid_points))[:n_strat]
        scan_points = [grid_points[i].copy() for i in idx]
        scan_points.extend(rng.uniform(-4.5, 4.5, size=(n_rand, dim)).tolist())

        # Evaluate scan points
        evaluated = []
        for x in scan_points:
            if time.time() > deadline - 5.0:
                break
            fx = func(x)
            evaluated.append((fx, np.array(x, dtype=float).copy()))

        if not evaluated:
            x0 = rng.uniform(-4, 4, dim)
            fx0 = func(x0)
            return x0, fx0

        # Sort and take top candidates
        evaluated.sort(key=lambda t: t[0])
        top_candidates = evaluated[:n_refine_candidates]

        best_fx = top_candidates[0][0]
        best_x = top_candidates[0][1].copy()

        # Phase 2: Refine with adaptive (1+1)-ES
        remaining_budget = budget - len(evaluated)
        # Allocate: 65% to best, 25% to second, 10% to third
        weights = [0.65, 0.25, 0.10]
        refine_budgets = [max(15, int(remaining_budget * w)) for w in weights]

        for rank, (fx_init, x_init) in enumerate(top_candidates):
            if time.time() > deadline - 10.0 or remaining_budget <= 0:
                break

            x = x_init.copy()
            fx = fx_init
            sigma = 2.0  # Larger initial sigma to escape broad local minima
            n_steps = min(refine_budgets[rank], remaining_budget)
            stall_count = 0

            for step in range(n_steps):
                if time.time() > deadline - 2.0 or remaining_budget <= 0:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                remaining_budget -= 1

                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.5  # Larger growth rate to escape local minima
                    stall_count = 0
                    if fx < best_fx:
                        best_x, best_fx = x.copy(), fx
                else:
                    sigma *= 0.75
                    stall_count += 1
                    if sigma < 0.1:
                        sigma = 0.1
                        stall_count = 0

        # Phase 3: Fine-tune the best solution with very small sigma
        if remaining_budget > 10 and time.time() < deadline - 5.0:
            x = best_x.copy()
            fx = best_fx
            sigma = 0.05
            n_fine = min(remaining_budget, 50)
            for step in range(n_fine):
                if time.time() > deadline - 1.0 or remaining_budget <= 0:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                remaining_budget -= 1
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.05
                    if fx < best_fx:
                        best_x, best_fx = x.copy(), fx
                else:
                    sigma *= 0.9

        return best_x, best_fx
# EVOLVE-BLOCK-END
