"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start optimizer combining (1+1)-ES global search with coordinate descent refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 285  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_made = 0

        def safe_func(x):
            nonlocal calls_made
            if calls_made >= budget:
                return np.inf
            calls_made += 1
            return func(x)

        def check_time():
            return calls_made < budget and time.time() < deadline - 5

        def coordinate_descent(x, fx, max_evals=20):
            """Adaptive coordinate descent for local refinement."""
            nonlocal best_f, best_x
            step = np.ones(dim) * 0.5
            for _ in range(max_evals):
                if not check_time():
                    break
                improved = False
                for d in range(dim):
                    if not check_time():
                        break
                    # Try positive
                    x_trial = x.copy()
                    x_trial[d] = np.clip(x_trial[d] + step[d], -5, 5)
                    if not np.allclose(x_trial, x):
                        f_trial = safe_func(x_trial)
                        if f_trial < fx:
                            x = x_trial
                            fx = f_trial
                            step[d] = min(step[d] * 1.5, 3.0)
                            improved = True
                            if fx < best_f:
                                best_f = fx
                                best_x = x.copy()
                            continue
                    # Try negative
                    x_trial = x.copy()
                    x_trial[d] = np.clip(x_trial[d] - step[d], -5, 5)
                    if not np.allclose(x_trial, x):
                        f_trial = safe_func(x_trial)
                        if f_trial < fx:
                            x = x_trial
                            fx = f_trial
                            step[d] = min(step[d] * 1.5, 3.0)
                            improved = True
                            if fx < best_f:
                                best_f = fx
                                best_x = x.copy()
                if not improved:
                    step *= 0.5
                    if np.max(step) < 1e-5:
                        break
            return x, fx

        def es_step(x, fx, sigma, max_evals=15):
            """(1+1)-ES with adaptive sigma for local exploration."""
            nonlocal best_f, best_x
            for _ in range(max_evals):
                if not check_time():
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = safe_func(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.85
            return x, fx, sigma

        # Phase 1: Generate diverse starting points
        n_starts = 8
        starts = [np.zeros(dim)]
        for i in range(n_starts - 1):
            starts.append(rng.uniform(-4, 4, dim))

        # Phase 2: For each start, do ES exploration then coordinate descent
        for s_idx, x0 in enumerate(starts):
            if not check_time():
                break

            x = np.clip(x0.copy(), -5, 5)
            fx = safe_func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            # ES exploration
            remaining = budget - calls_made
            es_evals = min(12, max(5, remaining // (n_starts - s_idx + 1)))
            x, fx, sigma = es_step(x, fx, sigma=1.5, max_evals=es_evals)

            # Coordinate descent refinement
            if check_time():
                cd_evals = min(15, budget - calls_made)
                x, fx = coordinate_descent(x, fx, max_evals=cd_evals)

        # Phase 3: Final fine-grained polish around the best point
        if best_x is not None and check_time():
            x = best_x.copy()
            fx = best_f
            # Fine coordinate search
            for scale in [0.1, 0.01, 0.001]:
                if not check_time():
                    break
                step = np.ones(dim) * scale
                for _ in range(8):
                    if not check_time():
                        break
                    improved = False
                    for d in range(dim):
                        if not check_time():
                            break
                        for direction in [1, -1]:
                            x_trial = x.copy()
                            x_trial[d] = np.clip(x_trial[d] + direction * step[d], -5, 5)
                            if not np.allclose(x_trial, x):
                                f_trial = safe_func(x_trial)
                                if f_trial < fx:
                                    x = x_trial
                                    fx = f_trial
                                    improved = True
                                    if fx < best_f:
                                        best_f = fx
                                        best_x = x.copy()
                    if not improved:
                        break

        if best_x is None:
            best_x = np.zeros(dim)
        return best_x, best_f
# EVOLVE-BLOCK-END
