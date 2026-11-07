"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start (1+1)-ES with restarts + Rastrigin-aware integer snapping."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget

        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 330s kill

        def time_ok():
            return time.time() < deadline

        calls = 0

        def evaluate(x):
            nonlocal calls
            x = np.clip(x, -5.0, 5.0)
            fx = float(func(x))
            calls += 1
            return x, fx

        # Reserve a slice for the Rastrigin-aware refinement phase.
        refine_budget = min(60, max(10, budget // 10))
        explore_budget = budget - refine_budget

        # ---------------- Phase 1: Multi-start (1+1)-ES with restarts --------
        best_x = None
        best_fx = np.inf

        # A single (1+1)-ES run from a starting point, with adaptive sigma.
        # Returns the best point found within that run.
        def es_run(x0, fx0, max_calls):
            nonlocal calls
            x = x0.copy()
            fx = fx0
            sigma = 1.5
            no_improve = 0
            max_no_improve = 25
            while calls < max_calls and time_ok():
                y = np.clip(x + sigma * rng.standard_normal(dim), -5.0, 5.0)
                fy = float(func(y))
                calls += 1
                if fy < fx:
                    x = y
                    fx = fy
                    sigma *= 1.1
                    no_improve = 0
                else:
                    sigma *= 0.9
                    no_improve += 1

                # Stagnation: reset sigma; if step is tiny, kick out of basin.
                if no_improve >= max_no_improve:
                    if sigma < 1e-3:
                        # Likely trapped in a local minimum; kick toward a
                        # fresh random point to escape the basin.
                        kick = x + rng.uniform(-3.0, 3.0, dim)
                        kick = np.clip(kick, -5.0, 5.0)
                        fk = float(func(kick))
                        calls += 1
                        if fk < fx:
                            x, fx = kick, fk
                        else:
                            x = x0.copy()
                            fx = fx0
                        sigma = 1.0
                        no_improve = 0
                    else:
                        sigma = max(sigma * 2.0, 0.5)
                        no_improve = 0

                if sigma < 1e-8:
                    break
            return x, fx

        # Initial broad exploration: a small population to find promising basins.
        pop_size = 10
        for i in range(pop_size):
            if not time_ok() or calls >= explore_budget:
                break
            x = rng.uniform(-5.0, 5.0, dim)
            fx = float(func(x))
            calls += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

        if best_x is None:
            x = np.zeros(dim)
            fx = float(func(x))
            calls += 1
            best_x, best_fx = x, fx

        # Refine the best point, then do further multi-start restarts from
        # random points while budget allows. Each run is capped so that we
        # always get several restarts rather than one long run.
        # First run from the best point found in exploration.
        per_run = max(20, explore_budget // 6)
        x, fx = es_run(best_x, best_fx, min(calls + per_run, explore_budget))
        if fx < best_fx:
            best_fx, best_x = fx, x.copy()

        # Additional restarts from random points to escape local minima.
        while calls < explore_budget and time_ok():
            x0 = rng.uniform(-5.0, 5.0, dim)
            fx0 = float(func(x0))
            calls += 1
            if fx0 < best_fx:
                best_fx, best_x = fx0, x0.copy()
            x, fx = es_run(x0, fx0, min(calls + per_run, explore_budget))
            if fx < best_fx:
                best_fx, best_x = fx, x.copy()

        # ---------------- Phase 2: Rastrigin-aware refinement ----------------
        # Rastrigin local minima lie at integer coordinates. Snapping each
        # coordinate to its nearest integer (and floor/ceil) can jump directly
        # to the true basin. Apply this to the global best point.
        if time_ok() and calls < budget:
            x = best_x.copy()
            fx = best_fx
            for coord in range(dim):
                if not time_ok() or calls >= budget:
                    break
                for cand in (np.round(x[coord]), np.floor(x[coord]), np.ceil(x[coord])):
                    if not time_ok() or calls >= budget:
                        break
                    xt = x.copy()
                    xt[coord] = cand
                    xt = np.clip(xt, -5.0, 5.0)
                    fxt = float(func(xt))
                    calls += 1
                    if fxt < fx:
                        x = xt
                        fx = fxt
            if fx < best_fx:
                best_fx, best_x = fx, x.copy()

        # ---------------- Phase 3: Fine-grained local search ----------------
        # A short polishing (1+1)-ES with a small sigma to settle into the
        # basin minimum.
        if time_ok() and calls < budget:
            x = best_x.copy()
            fx = best_fx
            sigma = 0.05
            while calls < budget and time_ok():
                y = np.clip(x + sigma * rng.standard_normal(dim), -5.0, 5.0)
                fy = float(func(y))
                calls += 1
                if fy < fx:
                    x = y
                    fx = fy
                    sigma *= 1.1
                else:
                    sigma *= 0.85
                    if sigma < 1e-9:
                        break
            if fx < best_fx:
                best_fx, best_x = fx, x.copy()

        return best_x, best_fx
# EVOLVE-BLOCK-END
