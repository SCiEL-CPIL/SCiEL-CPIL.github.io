"""Black-box continuous optimizer on BBOB-style f22 (Gallagher 21 peaks)."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Multi-start random + L-BFGS-B + Nelder-Mead polish for 21-peak function."""

    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 295.0  # safety margin before 330s kill
        D = self.dim
        budget = self.budget
        calls = 0

        def f(x):
            nonlocal calls
            if calls >= budget:
                return float('inf')
            calls += 1
            return float(func(np.asarray(x, dtype=float)))

        best_x = np.zeros(D)
        best_f = float('inf')

        def update_best(x, fx):
            nonlocal best_x, best_f
            if fx < best_f:
                best_f = fx
                best_x = np.array(x, dtype=float)

        # Phase 1: Random sampling with storage
        random_points = []
        n_random = min(150, budget // 4)
        for i in range(n_random):
            if calls >= budget or time.time() > deadline - 15:
                break
            x = np.random.uniform(-5.0, 5.0, D)
            fx = f(x)
            random_points.append((x, fx))
            update_best(x, fx)

        # Sort by fitness, select diverse starting points
        random_points.sort(key=lambda t: t[1])
        
        # Greedy diverse selection
        start_points = [random_points[0][0]]
        for i in range(1, min(30, len(random_points))):
            min_dist = min(np.linalg.norm(random_points[i][0] - sp) for sp in start_points)
            if min_dist > 1.0:
                start_points.append(random_points[i][0])
            if len(start_points) >= 12:
                break

        # Ensure we have at least a few start points
        if len(start_points) < 5:
            for x, fx in random_points[:5]:
                if not any(np.allclose(x, sp) for sp in start_points):
                    start_points.append(x)

        # Phase 2: L-BFGS-B from each diverse starting point
        for sp in start_points:
            if calls >= budget - 10 or time.time() > deadline - 10:
                break
            try:
                res = minimize(
                    lambda x: f(np.clip(x, -5.0, 5.0)),
                    sp,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * D,
                    options={'maxiter': 50, 'maxls': 20}
                )
                fx = res.fun
                if fx < float('inf'):
                    update_best(res.x, fx)
            except Exception:
                pass

        # Phase 3: Nelder-Mead polish from best point
        if calls < budget - 5 and time.time() < deadline - 5:
            try:
                res = minimize(
                    lambda x: f(np.clip(x, -5.0, 5.0)),
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
                )
                fx = res.fun
                if fx < float('inf'):
                    update_best(res.x, fx)
            except Exception:
                pass

        # Phase 4: Use remaining budget for more random exploration + local opt
        remaining = budget - calls
        if remaining > 30 and time.time() < deadline - 10:
            n_extra = min(remaining // 3, 60)
            new_points = []
            for i in range(n_extra):
                if calls >= budget - 10 or time.time() > deadline - 5:
                    break
                x = np.random.uniform(-5.0, 5.0, D)
                fx = f(x)
                new_points.append((x, fx))
                update_best(x, fx)
            
            # L-BFGS-B from best new point
            if new_points and calls < budget - 10 and time.time() < deadline - 5:
                new_points.sort(key=lambda t: t[1])
                try:
                    res = minimize(
                        lambda x: f(np.clip(x, -5.0, 5.0)),
                        new_points[0][0],
                        method='L-BFGS-B',
                        bounds=[(-5.0, 5.0)] * D,
                        options={'maxiter': 50, 'maxls': 20}
                    )
                    fx = res.fun
                    if fx < float('inf'):
                        update_best(res.x, fx)
                except Exception:
                    pass

        return best_x, best_f


# EVOLVE-BLOCK-END
