"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start: coarse sampling -> coordinate line search -> adaptive ES."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin before 300s/330s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_used = 0

        def safe_call(x):
            nonlocal calls_used
            if calls_used >= budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            try:
                val = float(func(np.asarray(x, dtype=np.float64)))
                calls_used += 1
                return val
            except Exception:
                calls_used += 1
                return np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        def time_ok():
            return time.time() < deadline

        def line_search(x, fx, func, dim, calls_used_ref, deadline):
            """Coordinate-wise line search with golden section refinement."""
            nonlocal calls_used
            x = x.copy()
            fx = fx
            for d in range(dim):
                if calls_used >= budget or not time_ok():
                    break
                # Try a few step sizes along dimension d
                best_step = 0
                best_fx = fx
                for step_size in [0.5, 1.0, 2.0, 4.0]:
                    for sign in [1, -1]:
                        y = x.copy()
                        y[d] += sign * step_size
                        y = np.clip(y, -5, 5)
                        fy = safe_call(y)
                        if fy < best_fx:
                            best_fx = fy
                            best_step = sign * step_size
                if best_step != 0:
                    x[d] += best_step
                    fx = best_fx
                    update_best(x, fx)
            return x, fx

        def es_refine(x, fx, func, dim, calls_used_ref, deadline):
            """Adaptive (1+1)-ES refinement."""
            nonlocal calls_used
            x = x.copy()
            fx = fx
            sigma = 1.0
            for _ in range(1000):
                if calls_used >= budget or not time_ok():
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = safe_call(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.1
                    update_best(x, fx)
                else:
                    sigma *= 0.85
                if sigma < 1e-6:
                    break
            return x, fx

        # Phase 1: Coarse random sampling to find promising regions
        n_coarse = min(30, budget // 10)
        coarse_points = []
        coarse_vals = []
        for _ in range(n_coarse):
            if not time_ok():
                break
            x = rng.uniform(-5, 5, dim)
            fx = safe_call(x)
            if fx < np.inf:
                coarse_points.append(x)
                coarse_vals.append(fx)

        if not coarse_points:
            # Fallback: try origin
            x0 = np.zeros(dim)
            fx0 = safe_call(x0)
            if fx0 < np.inf:
                best_x = x0
                best_f = fx0
            else:
                best_x = np.zeros(dim)
                best_f = 0.0
            return best_x, best_f

        # Sort by fitness
        order = np.argsort(coarse_vals)
        
        # Select top k starting points
        k = min(5, len(coarse_points))
        top_indices = order[:k]

        # Initialize best
        best_x = coarse_points[top_indices[0]].copy()
        best_f = coarse_vals[top_indices[0]]

        # Phase 2: Local refinement from each top starting point
        # Distribute remaining budget among restarts
        remaining_budget = budget - calls_used
        # Give more budget to better starts
        weights = np.zeros(k)
        for i, idx in enumerate(top_indices):
            # Weight inversely proportional to rank (better gets more)
            weights[i] = 1.0 / (i + 1)
        weights /= weights.sum()

        for i, start_idx in enumerate(top_indices):
            if calls_used >= budget or not time_ok():
                break

            x = coarse_points[start_idx].copy()
            fx = coarse_vals[start_idx]

            # Allocate budget for this restart
            alloc = int(remaining_budget * weights[i])
            alloc = max(5, min(alloc, budget - calls_used))

            # Phase 2a: Coordinate line search
            x, fx = line_search(x, fx, func, dim, calls_used, deadline)

            # Phase 2b: ES refinement
            x, fx = es_refine(x, fx, func, dim, calls_used, deadline)

        return best_x, best_f
# EVOLVE-BLOCK-END
