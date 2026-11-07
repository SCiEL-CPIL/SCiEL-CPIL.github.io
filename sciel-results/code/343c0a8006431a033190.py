"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Linear function optimizer: estimate gradient, jump to best vertex."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        dim = self.dim
        budget = self.budget

        # Track best solution
        best_x = np.zeros(dim)
        best_f = np.inf
        calls_made = 0

        def safe_call(x):
            nonlocal calls_made
            if calls_made >= budget:
                return None
            calls_made += 1
            try:
                return func(x)
            except Exception:
                return None

        # Step 1: Evaluate at center (origin) to get baseline
        center = np.zeros(dim)
        f_center = safe_call(center)
        if f_center is None:
            # Fallback: just do random search
            rng = np.random.default_rng(42)
            for _ in range(budget - calls_made):
                if time.time() > deadline:
                    break
                x = rng.uniform(-5, 5, dim)
                fx = safe_call(x)
                if fx is not None and fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            return best_x, best_f

        best_f = min(best_f, f_center)
        best_x = center.copy()

        # Step 2: Estimate gradient by evaluating along each axis
        # Use a small step to stay within bounds and get linear approximation
        step = 0.5
        grad = np.zeros(dim)
        for i in range(dim):
            if time.time() > deadline:
                break
            x_plus = center.copy()
            x_plus[i] += step
            x_minus = center.copy()
            x_minus[i] -= step

            fp = safe_call(np.clip(x_plus, -5, 5))
            fm = safe_call(np.clip(x_minus, -5, 5))

            if fp is not None and fm is not None:
                grad[i] = (fp - fm) / (2 * step)
                # Track best
                if fp < best_f:
                    best_f = fp
                    best_x = np.clip(x_plus, -5, 5).copy()
                if fm < best_f:
                    best_f = fm
                    best_x = np.clip(x_minus, -5, 5).copy()
            elif fp is not None:
                if fp < best_f:
                    best_f = fp
                    best_x = np.clip(x_plus, -5, 5).copy()
            elif fm is not None:
                if fm < best_f:
                    best_f = fm
                    best_x = np.clip(x_minus, -5, 5).copy()

        # Step 3: For a linear function, the minimum is at a vertex.
        # The best vertex has x_i = -5 if grad[i] > 0, else +5
        # (since we want to minimize f(x) = f0 + grad·(x - center))
        if np.any(np.abs(grad) > 1e-10):
            vertex = np.where(grad > 0, -5.0, 5.0)
            vertex = np.clip(vertex, -5, 5)
            f_vertex = safe_call(vertex)
            if f_vertex is not None and f_vertex < best_f:
                best_f = f_vertex
                best_x = vertex.copy()
        else:
            # Gradient is ~0, function is nearly flat; try a few random vertices
            rng = np.random.default_rng(123)
            for _ in range(min(10, budget - calls_made)):
                if time.time() > deadline:
                    break
                v = rng.choice([-5, 5], dim)
                fv = safe_call(v)
                if fv is not None and fv < best_f:
                    best_f = fv
                    best_x = v.copy()

        # Step 4: Use remaining budget for local refinement around best_x
        # Do a simple (1+1)-ES style local search
        if calls_made < budget and time.time() < deadline:
            rng = np.random.default_rng(99)
            sigma = 0.5
            x = best_x.copy()
            fx = best_f
            for _ in range(budget - calls_made):
                if time.time() > deadline:
                    break
                if calls_made >= budget:
                    break
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = safe_call(y)
                if fy is None:
                    break
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                else:
                    sigma *= 0.8
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
