"""Multi-start (1+1)-ES with basin-aware seeding and coordinate-descent refinement
for Lunacek bi-Rastrigin (BBOB f24)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start ES with basin-aware seeding and coordinate refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 20s safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_fx = np.inf
        calls = 0

        def time_ok():
            return time.time() < deadline

        def es_search(x0, sigma0, momentum_decay=0.9, max_iter=None):
            nonlocal best_x, best_fx, calls
            x = x0.copy()
            fx = func(x)
            calls += 1
            if fx < best_fx:
                best_x = x.copy()
                best_fx = fx

            sigma = sigma0
            momentum = np.zeros(dim)
            no_improve = 0
            max_no_improve = 25 * dim

            while calls < budget and time_ok():
                if max_iter is not None and max_iter <= 0:
                    break
                if max_iter is not None:
                    max_iter -= 1

                # Generate candidate with momentum
                step = rng.normal(0, sigma, dim)
                new_x = x + momentum + step
                new_x = np.clip(new_x, -5, 5)

                fy = func(new_x)
                calls += 1

                if fy < fx:
                    # Success: update momentum and x
                    momentum = momentum_decay * momentum + (1 - momentum_decay) * (new_x - x)
                    x = new_x
                    fx = fy
                    no_improve = 0
                    sigma *= 1.05
                    if fx < best_fx:
                        best_x = x.copy()
                        best_fx = fx
                else:
                    # Failure: shrink sigma, keep momentum
                    no_improve += 1
                    sigma *= 0.9
                    if no_improve > max_no_improve:
                        break

                if sigma < 1e-8:
                    break

            return x, fx

        def coordinate_refinement(x, fx, n_passes=3, n_points=9, span=1.5):
            """Coordinate descent refinement around a point."""
            nonlocal best_x, best_fx, calls
            for _ in range(n_passes):
                if not time_ok() or calls >= budget:
                    break
                improved = False
                for d in range(dim):
                    if not time_ok() or calls >= budget:
                        break
                    center = x[d]
                    lo = max(-5.0, center - span / 2)
                    hi = min(5.0, center + span / 2)
                    if lo >= hi:
                        continue
                    grid = np.linspace(lo, hi, n_points)
                    best_val = fx
                    best_pt = x[d]
                    for val in grid:
                        if not time_ok() or calls >= budget:
                            break
                        trial = x.copy()
                        trial[d] = val
                        f_val = func(trial)
                        calls += 1
                        if f_val < best_val:
                            best_val = f_val
                            best_pt = val
                    if best_pt != x[d]:
                        x[d] = best_pt
                        fx = best_val
                        improved = True
                if not improved:
                    break
            if fx < best_fx:
                best_x = x.copy()
                best_fx = fx
            return x, fx

        # Phase 1: Basin-aware multi-start ES
        # Lunacek bi-Rastrigin has a global min near origin and a local min offset
        # Seed both basins directly
        
        # Start 1: Center (likely global basin)
        x0 = np.zeros(dim)
        x1, fx1 = es_search(x0, sigma0=1.0, max_iter=budget // 5)
        coordinate_refinement(x1, fx1, n_passes=2, n_points=7, span=2.0)

        # Start 2: Offset point (likely local basin)
        x0 = np.full(dim, 2.5)
        x2, fx2 = es_search(x0, sigma0=0.8, max_iter=budget // 5)
        coordinate_refinement(x2, fx2, n_passes=2, n_points=7, span=2.0)

        # Start 3: Negative offset
        x0 = np.full(dim, -2.5)
        x3, fx3 = es_search(x0, sigma0=0.8, max_iter=budget // 5)
        coordinate_refinement(x3, fx3, n_passes=2, n_points=7, span=2.0)

        # Start 4-6: Random diverse points
        for _ in range(3):
            if calls >= budget or not time_ok():
                break
            x0 = rng.uniform(-3.5, 3.5, dim)
            xi, fxi = es_search(x0, sigma0=0.6, max_iter=budget // 8)
            coordinate_refinement(xi, fxi, n_passes=1, n_points=7, span=1.5)

        # Phase 2: Fine polish from global best
        if calls < budget and time_ok() and best_x is not None:
            remaining = budget - calls
            if remaining > 30:
                # Fine ES polish
                es_search(best_x, sigma0=0.05, max_iter=remaining // 2)
                # Coordinate refinement with fine grid
                if calls < budget and time_ok():
                    coordinate_refinement(best_x, best_fx, n_passes=3, n_points=11, span=0.5)

        return best_x, best_fx
# EVOLVE-BLOCK-END
