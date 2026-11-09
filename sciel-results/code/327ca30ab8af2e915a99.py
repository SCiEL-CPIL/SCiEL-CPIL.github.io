"""Optimizer for BBOB f13 (Sharp ridge) with time guard."""
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.perf_counter()
        deadline = t0 + 270.0  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Best-so-far tracking
        best_x = None
        best_f = np.inf
        call_count = 0

        def try_update(x, f):
            nonlocal best_x, best_f, call_count
            if f < best_f:
                best_x = x.copy()
                best_f = f
            call_count += 1

        # Phase 1: Coarse global search with multiple restarts
        # Use a population-based approach to find a good starting point
        # We'll do a simple CMA-ES-like or random restart with local refinement

        # Strategy: 
        # 1. First, do a coarse grid/random search to find a promising region
        # 2. Then do a local (1+1)-ES or CMA-ES from the best point

        # Given the sharp ridge structure, we need to be careful.
        # Let's use a combination of:
        # - Random restarts with different initial points
        # - For each restart, run a (1+1)-ES with adaptive step size
        # - Keep track of the best overall

        # Phase 1: Quick random sampling to get initial estimates
        n_initial = min(50, budget // 10)
        for i in range(n_initial):
            x = rng.uniform(-5, 5, dim)
            f = func(x)
            try_update(x, f)
            if call_count >= budget:
                break

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = 0.0

        # Phase 2: Local optimization from best point using (1+1)-ES with restarts
        # We'll do multiple restarts, each with a limited budget
        # The key insight for sharp ridge: we need to follow the ridge carefully

        # Let's implement a (1+1)-ES with step size adaptation
        # and restart if no progress for a while

        def run_local_es(start_x, max_calls):
            """Run (1+1)-ES from start_x, using at most max_calls evaluations."""
            nonlocal best_x, best_f, call_count
            x = start_x.copy()
            f = best_f  # We know this value
            sigma = 0.5
            no_improve = 0
            max_no_improve = 30
            local_calls = 0

            for _ in range(max_calls):
                if call_count + 1 > budget:
                    break
                if time.perf_counter() > deadline:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                call_count += 1
                local_calls += 1

                if fy < f:
                    x = y
                    f = fy
                    no_improve = 0
                    sigma *= 1.2
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
                else:
                    no_improve += 1
                    sigma *= 0.9

                if no_improve > max_no_improve:
                    break

            return f

        # Do several local ES runs from different starting points
        # First from the best found point
        remaining = budget - call_count
        if remaining > 0:
            # Split remaining budget among several restarts
            n_restarts = min(5, max(1, remaining // 50))
            calls_per_restart = max(10, remaining // n_restarts)

            for r in range(n_restarts):
                if call_count >= budget:
                    break
                if time.perf_counter() > deadline:
                    break

                if r == 0:
                    start = best_x.copy()
                else:
                    # Perturb the best point
                    start = np.clip(best_x + rng.standard_normal(dim) * 0.5, -5, 5)

                run_local_es(start, calls_per_restart)

        # Phase 3: If we still have budget, do a finer local search
        remaining = budget - call_count
        if remaining > 20:
            # Reduce step size and continue from best
            x = best_x.copy()
            f = best_f
            sigma = 0.1
            no_improve = 0

            for _ in range(remaining):
                if time.perf_counter() > deadline:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                call_count += 1

                if fy < f:
                    x = y
                    f = fy
                    no_improve = 0
                    sigma *= 1.1
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
                else:
                    no_improve += 1
                    sigma *= 0.95

                if no_improve > 50:
                    break

        return best_x, best_f
