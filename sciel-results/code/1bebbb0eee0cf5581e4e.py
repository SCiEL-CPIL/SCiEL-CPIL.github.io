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

        def time_ok():
            return time.time() < deadline

        # Step 1: Probe the function at all 2^d vertices of an inscribed hypercube
        # centered at origin with half-side s=1. This robustly determines the
        # gradient direction without finite-difference magnitude issues.
        s = 1.0
        # Generate all 2^dim sign combinations
        signs = np.array(list(np.ndindex(*([2] * dim))), dtype=np.int8)
        # Convert binary indices to ±1
        signs = 1 - 2 * signs  # 0->1, 1->-1, so signs[i] in {-1, 1}
        # Actually ndindex gives tuples of 0/1, so 1-2*0=1, 1-2*1=-1
        # We want vertex = center + s * sign_vec
        center = np.zeros(dim)
        
        # Evaluate all 2^dim vertices
        vertex_values = []
        vertex_points = []
        for sign_vec in signs:
            if not time_ok() or calls_made >= budget:
                break
            x = center + s * sign_vec.astype(float)
            x = np.clip(x, -5, 5)
            fx = safe_call(x)
            if fx is not None:
                vertex_values.append(fx)
                vertex_points.append(x.copy())
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

        # Step 2: Determine gradient direction from vertex evaluations
        # For a linear function f(x) = f0 + w·x, the vertex with minimum f
        # has sign_i = -sign(w_i). We can estimate w_i by comparing pairs.
        # Use the best vertex as the candidate for the minimum corner.
        # For a linear function, the global min is at a vertex of [-5,5]^d.
        # The best inscribed vertex tells us the direction; the true min
        # is at the corresponding corner of [-5,5]^d.
        
        if len(vertex_values) >= 2:
            # Find the best vertex among those evaluated
            best_idx = int(np.argmin(vertex_values))
            best_sign = signs[best_idx].astype(float)
            
            # The true minimum vertex of [-5,5]^d in this direction
            # For linear f(x) = f0 + w·x, min at x_i = -5 if w_i > 0, else 5
            # best_sign[i] = -1 means w_i > 0 (since that vertex was best in inscribed cube)
            # So the true min corner: x_i = -5 if best_sign[i] == -1, else 5
            candidate = np.where(best_sign < 0, -5.0, 5.0)
            candidate = np.clip(candidate, -5, 5)
            
            if calls_made < budget and time_ok():
                f_cand = safe_call(candidate)
                if f_cand is not None and f_cand < best_f:
                    best_f = f_cand
                    best_x = candidate.copy()
        else:
            # Not enough data; fall back to random vertex search
            rng = np.random.default_rng(123)
            for _ in range(min(10, budget - calls_made)):
                if not time_ok():
                    break
                v = rng.choice([-5, 5], dim)
                fv = safe_call(v)
                if fv is not None and fv < best_f:
                    best_f = fv
                    best_x = v.copy()

        # Step 3: Use remaining budget for local refinement around best_x
        # Do a simple (1+1)-ES style local search
        if calls_made < budget and time_ok():
            rng = np.random.default_rng(99)
            sigma = 0.5
            x = best_x.copy()
            fx = best_f
            for _ in range(budget - calls_made):
                if not time_ok():
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
