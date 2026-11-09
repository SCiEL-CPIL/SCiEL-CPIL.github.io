import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        calls = [0]
        best_x = [None]
        best_f = [float('inf')]

        def evaluate(x):
            if calls[0] >= budget or time.time() > deadline:
                return None
            x_clipped = np.clip(x, -5, 5)
            val = func(x_clipped)
            calls[0] += 1
            if val < best_f[0]:
                best_f[0] = val
                best_x[0] = x_clipped.copy()
            return val

        # 1. Initial exploration: diverse sampling
        initial_candidates = []
        # Center
        initial_candidates.append(np.zeros(n))
        # Random points
        for _ in range(30):
            initial_candidates.append(rng.uniform(-4, 4, n))
        # Points near typical Rosenbrock minima (shifted/rotated)
        for _ in range(15):
            initial_candidates.append(rng.uniform(0.5, 1.5, n))
        for _ in range(15):
            initial_candidates.append(rng.uniform(-1.5, -0.5, n))
        # Some structured points
        for i in range(n):
            x = np.zeros(n)
            x[i] = 1.0
            initial_candidates.append(x)
            x[i] = -1.0
            initial_candidates.append(x)

        for x in initial_candidates:
            if calls[0] >= budget or time.time() > deadline:
                break
            evaluate(x)

        if best_x[0] is None:
            best_x[0] = np.zeros(n)
            evaluate(best_x[0])
            return best_x[0], best_f[0]

        # 2. Hybrid local search with adaptive strategy
        # Use Nelder-Mead for local refinement, with restarts on stagnation
        def local_search(x_start, max_iter):
            if calls[0] >= budget or time.time() > deadline:
                return best_x[0], best_f[0]

            local_calls = [0]
            local_best_x = [x_start.copy()]
            local_best_f = [best_f[0]]

            def wrapper(z):
                if calls[0] >= budget or time.time() > deadline or local_calls[0] >= max_iter * n * 2:
                    return local_best_f[0]

                z_clipped = np.clip(z, -5, 5)
                val = func(z_clipped)
                calls[0] += 1
                local_calls[0] += 1

                if val < local_best_f[0]:
                    local_best_f[0] = val
                    local_best_x[0] = z_clipped.copy()

                if val < best_f[0]:
                    best_f[0] = val
                    best_x[0] = z_clipped.copy()

                return val

            try:
                result = minimize(
                    wrapper,
                    x_start,
                    method='Nelder-Mead',
                    options={'maxiter': max_iter, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
            except Exception:
                pass

            return best_x[0], best_f[0]

        # Main search loop: alternate between local refinement and global jumps
        attempts = 0
        max_attempts = 80
        last_best_f = best_f[0]
        calls_since_improvement = 0
        stagnation_limit = 40  # If no improvement in 40 calls, consider a jump

        while calls[0] < budget and time.time() < deadline and attempts < max_attempts:
            attempts += 1

            # Determine starting point
            if attempts == 1:
                x_start = best_x[0].copy()
            else:
                if calls_since_improvement > stagnation_limit:
                    # Stagnation: try a jump
                    r = rng.random()
                    if r < 0.3:
                        # Local perturbation
                        x_start = best_x[0] + rng.normal(0, 0.3, n)
                    elif r < 0.6:
                        # Medium jump
                        x_start = best_x[0] + rng.normal(0, 1.0, n)
                    else:
                        # Global jump
                        x_start = rng.uniform(-4, 4, n)
                    calls_since_improvement = 0
                else:
                    # Continue refining from current best
                    x_start = best_x[0].copy()

            # Dynamic local budget: more early on, less later
            remaining = budget - calls[0]
            if remaining > budget * 0.5:
                local_max_iter = 300
            elif remaining > budget * 0.2:
                local_max_iter = 150
            else:
                local_max_iter = 50

            calls_before = calls[0]
            x_best_local, f_best_local = local_search(x_start, local_max_iter)
            calls_after = calls[0]

            # Update stagnation counter
            if f_best_local < last_best_f:
                last_best_f = f_best_local
                calls_since_improvement = 0
            else:
                calls_since_improvement += calls_after - calls_before

        return best_x[0], best_f[0]
