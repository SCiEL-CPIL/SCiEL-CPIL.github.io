import numpy as np
import time
import scipy.optimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit

        calls = 0
        best_x = None
        best_f = np.inf

        def safe_func(x):
            nonlocal calls, best_x, best_f
            if time.time() > deadline:
                raise TimeoutError("Deadline exceeded")
            if calls >= self.budget:
                raise StopIteration("Budget exceeded")

            x = np.asarray(x, dtype=np.float64)
            f = func(x)
            calls += 1

            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        rng = np.random.default_rng(42)
        n = self.dim

        # Phase 1: (1+1)-ES with adaptive sigma and restarts
        # Use ~75% of budget for global exploration
        global_budget = int(self.budget * 0.75)

        # Start at origin
        x = np.zeros(n)
        try:
            safe_func(x)
        except (TimeoutError, StopIteration):
            return best_x, best_f

        sigma = 1.0
        stall_count = 0
        stall_limit = 60
        best_local = best_f
        best_local_x = best_x.copy() if best_x is not None else x.copy()

        while calls < global_budget and time.time() < deadline:
            # Generate candidate
            direction = rng.standard_normal(n)
            candidate = x + sigma * direction
            candidate = np.clip(candidate, -5.0, 5.0)

            try:
                f_candidate = safe_func(candidate)
            except (TimeoutError, StopIteration):
                break

            if f_candidate < best_local - 1e-12:
                # Success: move to candidate, grow sigma
                best_local = f_candidate
                best_local_x = candidate.copy()
                x = candidate.copy()
                sigma *= 1.15
                stall_count = 0
            else:
                # Failure: stay, shrink sigma
                sigma *= 0.92
                stall_count += 1

                # If stalled too long, restart from a random point
                if stall_count >= stall_limit:
                    x = rng.uniform(-4.0, 4.0, n)
                    sigma = 1.0
                    stall_count = 0
                    try:
                        fx = safe_func(x)
                        if fx < best_local:
                            best_local = fx
                            best_local_x = x.copy()
                    except (TimeoutError, StopIteration):
                        break

        # Phase 2: Local refinement with Nelder-Mead on the global best
        remaining = self.budget - calls
        if remaining > 10 and best_x is not None and time.time() < deadline:
            try:
                result = scipy.optimize.minimize(
                    safe_func,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': remaining,
                        'xatol': 1e-10,
                        'fatol': 1e-12,
                        'adaptive': True
                    }
                )
            except (TimeoutError, StopIteration, Exception):
                pass

        return best_x, best_f
