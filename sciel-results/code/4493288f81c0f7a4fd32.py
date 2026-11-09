"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start Nelder-Mead approach for multimodal optimization."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 290.0  # safety margin before 330s kill
        rng = np.random.default_rng(42)
        dim = self.dim
        calls = 0
        best_x = None
        best_f = float('inf')

        def wrapped_func(x):
            nonlocal calls, best_x, best_f
            if calls >= self.budget:
                return best_f
            if time.time() - start_time > time_limit:
                return best_f
            x_clipped = np.clip(x, -5, 5)
            v = func(x_clipped)
            calls += 1
            if v < best_f:
                best_f = v
                best_x = x_clipped.copy()
            return v

        def nm_run(seed, maxiter, xatol=1e-6, fatol=1e-9):
            """Run Nelder-Mead with safety checks."""
            nonlocal best_f, best_x
            if calls >= self.budget or time.time() - start_time > time_limit - 2.0:
                return
            try:
                res = minimize(wrapped_func, seed, method='Nelder-Mead',
                               options={'maxiter': maxiter, 'xatol': xatol, 'fatol': fatol,
                                        'adaptive': True, 'initial_simplex': None})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5, 5)
            except Exception:
                pass

        # 1. Initial diverse sampling - more points for better coverage
        n_init = 30
        init_points = []
        init_points.append(np.zeros(dim))
        for _ in range(n_init - 1):
            init_points.append(rng.uniform(-4.5, 4.5, dim))

        init_values = []
        for p in init_points:
            p = np.clip(p, -5, 5)
            v = wrapped_func(p)
            init_values.append(v)

        # Sort init points by value
        sorted_indices = np.argsort(init_values)

        # 2. Multi-start Nelder-Mead from top seeds with adaptive maxiter
        # Allocate more iterations to better seeds
        top_k = min(8, len(sorted_indices))
        for rank, idx in enumerate(sorted_indices[:top_k]):
            if calls >= self.budget - 30:
                break
            if time.time() - start_time > time_limit - 5.0:
                break

            seed = init_points[idx].copy()
            # Better seeds get more iterations
            if rank < 3:
                maxiter = 200
            elif rank < 5:
                maxiter = 100
            else:
                maxiter = 50

            nm_run(seed, maxiter)

        # 3. Random restarts with short NM runs, prioritizing near best
        remaining_budget = self.budget - calls
        n_restarts = max(5, remaining_budget // 40)

        for _ in range(n_restarts):
            if calls >= self.budget - 10:
                break
            if time.time() - start_time > time_limit - 5.0:
                break

            # Mix of random and local perturbations around best
            if best_x is not None and rng.random() < 0.5:
                # Local search around best
                scale = max(0.1, (best_f - (init_values[sorted_indices[0]])) * 0.1)
                new_p = best_x + rng.normal(0, scale, dim)
            else:
                new_p = rng.uniform(-4.5, 4.5, dim)

            new_p = np.clip(new_p, -5, 5)
            v = wrapped_func(new_p)

            # Only refine if it's reasonably good
            if v < best_f + 5.0:
                nm_run(new_p, maxiter=30)

        # 4. Final high-precision polish on the best solution
        if best_x is not None and calls < self.budget - 5 and time.time() - start_time < time_limit - 3.0:
            nm_run(best_x.copy(), maxiter=300, xatol=1e-8, fatol=1e-10)

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
