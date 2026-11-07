"""Portfolio search optimizer for BBOB f20 (Schwefel) with explicit deadline guard."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """
    Hybrid optimizer using random sampling and L-BFGS-B for BBOB f20 (Schwefel).
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        rng = np.random.default_rng(0)

        best_x = None
        best_f = np.inf
        call_count = 0
        bounds = [(-5.0, 5.0)] * self.dim

        def try_func(x):
            nonlocal best_x, best_f, call_count
            if call_count >= self.budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            try:
                fx = func(x)
                call_count += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                return fx
            except Exception:
                return np.inf

        # Phase 1: Initial random sampling
        n_init = 20
        candidates = []
        for _ in range(n_init):
            if call_count >= self.budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, self.dim)
            fx = try_func(x)
            if np.isfinite(fx):
                candidates.append((fx, x))

        if not candidates:
            return best_x, best_f

        candidates.sort(key=lambda t: t[0])

        # Phase 2: Local optimization on top 3 candidates
        top_k = 3
        max_fun_per_opt = 100

        for i in range(min(top_k, len(candidates))):
            if call_count >= self.budget or time.time() > deadline - 10:
                break

            start_x = candidates[i][1]

            def local_func(x):
                return try_func(x)

            try:
                res = minimize(local_func, start_x, method='L-BFGS-B', bounds=bounds,
                               options={'maxiter': 50, 'maxfun': max_fun_per_opt})
            except Exception:
                pass

        # Phase 3: More random sampling
        if call_count < self.budget - 100 and time.time() < deadline - 10:
            n_init2 = 20
            for _ in range(n_init2):
                if call_count >= self.budget or time.time() > deadline:
                    break
                x = rng.uniform(-5, 5, self.dim)
                fx = try_func(x)

        # Phase 4: Local optimization on best found so far
        if call_count < self.budget - 50 and time.time() < deadline - 5:
            if best_x is not None:
                def local_func2(x):
                    return try_func(x)
                try:
                    res = minimize(local_func2, best_x, method='L-BFGS-B', bounds=bounds,
                                   options={'maxiter': 50, 'maxfun': 50})
                except Exception:
                    pass

        # Phase 5: Final random sampling
        if call_count < self.budget and time.time() < deadline - 1:
            n_init3 = 10
            for _ in range(n_init3):
                if call_count >= self.budget or time.time() > deadline:
                    break
                x = rng.uniform(-5, 5, self.dim)
                fx = try_func(x)

        # Phase 6: Final polish
        if call_count < self.budget - 10 and time.time() < deadline - 1:
            if best_x is not None:
                def local_func3(x):
                    return try_func(x)
                try:
                    res = minimize(local_func3, best_x, method='L-BFGS-B', bounds=bounds,
                                   options={'maxiter': 20, 'maxfun': 10})
                except Exception:
                    pass

        return best_x, best_f
# EVOLVE-BLOCK-END
