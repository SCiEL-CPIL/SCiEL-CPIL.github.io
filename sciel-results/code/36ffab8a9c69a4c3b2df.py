"""Multi-start CMA-ES-like optimizer for BBOB f21 (Gallagher 101 peaks)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start local-search optimizer with structured initialization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 270.0
        calls_made = 0
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(2024)

        best_x = None
        best_f = np.inf

        def evaluate(x):
            nonlocal best_x, best_f, calls_made
            x = np.clip(x, -5.0, 5.0)
            f = float(func(x))
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Structured initialization
        n_init = 30
        init_points = []
        init_points.append(np.zeros(dim))
        # Grid-like points along axes
        for d in range(dim):
            for val in [-3.0, -1.0, 1.0, 3.0]:
                p = np.zeros(dim)
                p[d] = val
                init_points.append(p)
        # Random points to fill
        while len(init_points) < n_init:
            init_points.append(rng.uniform(-4.0, 4.0, dim))
        init_points = np.array(init_points[:n_init])

        init_results = []
        for i in range(n_init):
            if calls_made >= budget or time.time() > deadline:
                break
            f = evaluate(init_points[i])
            init_results.append((init_points[i].copy(), f))

        init_results.sort(key=lambda t: t[1])

        # Phase 2: Local refinement using adaptive random-walk ES
        K = min(5, len(init_results))
        if K == 0:
            return best_x, best_f

        top_starts = init_results[:K]
        remaining = budget - calls_made
        per_start = max(30, remaining // K)

        for x0, f0 in top_starts:
            if calls_made >= budget or time.time() > deadline:
                break

            x = x0.copy()
            fx = f0
            sigma = 0.4 * np.ones(dim)
            es_calls = 0
            stagnation = 0
            max_stag = 20

            while es_calls < per_start:
                if calls_made >= budget or time.time() > deadline:
                    break

                d = rng.standard_normal(dim)
                y = x + sigma * d
                y = np.clip(y, -5.0, 5.0)
                fy = evaluate(y)
                es_calls += 1

                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.15
                    sigma = np.clip(sigma, 0.01, 4.0)
                    stagnation = 0
                else:
                    sigma *= 0.88
                    sigma = np.clip(sigma, 0.01, 4.0)
                    stagnation += 1

                if stagnation >= max_stag:
                    if best_x is not None and rng.random() < 0.6:
                        x = best_x.copy() + 0.3 * rng.standard_normal(dim)
                        fx = evaluate(x)
                        es_calls += 1
                    else:
                        x = rng.uniform(-4.0, 4.0, dim)
                        fx = evaluate(x)
                        es_calls += 1
                    sigma = 0.4 * np.ones(dim)
                    stagnation = 0

        # Phase 3: Final polish with very small steps around global best
        if calls_made < budget and time.time() < deadline and best_x is not None:
            xp = best_x.copy()
            fp = best_f
            sp = 0.03
            for _ in range(min(60, budget - calls_made)):
                if time.time() > deadline:
                    break
                y = xp + sp * rng.standard_normal(dim)
                y = np.clip(y, -5.0, 5.0)
                fy = evaluate(y)
                if fy < fp:
                    xp, fp = y, fy
                    sp = max(0.005, sp * 0.92)
                else:
                    sp = max(0.003, sp * 0.85)
                if sp < 0.005:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
