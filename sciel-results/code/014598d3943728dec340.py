import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin before 300s limit
        n = self.dim
        budget = self.budget
        calls = 0
        best_x = np.zeros(n)
        best_f = np.inf
        evaluated_points = []

        def _counted_func(x):
            nonlocal calls, best_x, best_f, evaluated_points
            if time.time() > deadline:
                return np.inf
            if calls >= budget:
                return np.inf
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            fx = float(func(x))
            calls += 1
            evaluated_points.append((x.copy(), fx))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Global Exploration using Halton Sequence for better coverage
        # Weierstrass is oscillatory; good initial spread is crucial.
        # 40 points is a reasonable trade-off: enough to find a basin, leaves 460 for local search.
        n_init = 40
        if n_init > budget:
            n_init = max(1, budget // 2)
            
        # Simple Halton sequence generator for low-discrepancy sampling
        def halton_sequence(index, base):
            f = 1.0
            r = 0.0
            i = index
            while i > 0:
                f = f / base
                r = r + f * (i % base)
                i = i // base
            return r

        init_points = []
        for i in range(1, n_init + 1):
            point = []
            for d in range(n):
                # Use distinct prime bases for each dimension
                # Primes: 2, 3, 5, 7, 11
                base = [2, 3, 5, 7, 11][d % 5]
                point.append(halton_sequence(i, base))
            # Map from [0, 1] to [-5, 5]
            x = np.array(point) * 10.0 - 5.0
            init_points.append(x)

        for x in init_points:
            if time.time() > deadline or calls >= budget:
                break
            _counted_func(x)

        if calls == 0:
            return best_x, best_f

        # Sort evaluated points by fitness to get the best starting points
        evaluated_points.sort(key=lambda item: item[1])
        
        # Select top unique starting points for local optimization
        # We want distinct basins, so we check distance between candidates
        start_points = []
        min_dist = 1.0  # Minimum distance to consider points distinct
        for x, f in evaluated_points:
            is_distinct = True
            for sp in start_points:
                if np.linalg.norm(x - sp) < min_dist:
                    is_distinct = False
                    break
            if is_distinct:
                start_points.append(x)
            if len(start_points) >= 5: # Max 5 restarts
                break

        # Phase 2: Multi-start Nelder-Mead
        # Nelder-Mead is derivative-free and robust for non-convex, oscillatory functions.
        remaining_budget = budget - calls
        if remaining_budget > 0 and start_points:
            # Distribute remaining budget among restarts
            # Give more budget to earlier/better restarts? Or equal?
            # Equal is safer.
            budget_per_restart = max(10, remaining_budget // len(start_points))
            
            for i, sp in enumerate(start_points):
                if time.time() > deadline or calls >= budget:
                    break
                
                # Adjust maxfev to not exceed total budget
                maxfev_allowed = min(budget_per_restart, budget - calls)
                if maxfev_allowed < 10:
                    break

                try:
                    # Nelder-Mead options
                    # adaptive=True helps with scaling issues
                    result = minimize(
                        _counted_func,
                        sp,
                        method='Nelder-Mead',
                        options={
                            'maxiter': max(20, maxfev_allowed // 5),
                            'maxfev': maxfev_allowed,
                            'xatol': 1e-6,
                            'fatol': 1e-8,
                            'adaptive': True
                        }
                    )
                    if result.fun < best_f:
                        best_f = result.fun
                        best_x = np.clip(result.x, -5.0, 5.0).copy()
                except Exception:
                    pass

        # Phase 3: Final Polish with Coordinate Descent
        # If budget remains, do a fine-grained coordinate descent around the best point
        if calls < budget - 5 and time.time() < deadline:
            step = 0.05
            # Try a few iterations of coordinate descent
            for _ in range(20):
                if calls >= budget or time.time() > deadline:
                    break
                improved = False
                for d in range(n):
                    if calls >= budget or time.time() > deadline:
                        break
                    for sign in [1.0, -1.0]:
                        if calls >= budget or time.time() > deadline:
                            break
                        x_p = best_x.copy()
                        x_p[d] += sign * step
                        x_p = np.clip(x_p, -5.0, 5.0)
                        fx_p = _counted_func(x_p)
                        if fx_p < best_f:
                            best_x = x_p.copy()
                            best_f = fx_p
                            improved = True
                if not improved:
                    step *= 0.5
                    if step < 1e-5:
                        break

        return best_x, best_f
