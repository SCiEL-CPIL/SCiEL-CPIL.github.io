"""Black-box continuous optimizer for BBOB-style f14 (Different powers)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-start L-BFGS-B optimizer with Halton sequence initialization.
    
    Strategy:
    1. Use a Halton sequence to generate diverse starting points across the box.
    2. Apply L-BFGS-B (a gradient-based method) from each start point.
    3. L-BFGS-B converges rapidly for smooth functions like BBOB f14.
    4. This approach typically reaches high precision (low error) in fewer
       function evaluations than population-based methods like CMA-ES,
       which is critical for maximizing the AOCC score (early progress).
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 300.0  # Safety margin before 330s kill
        dim = self.dim
        budget = self.budget
        calls = 0

        best_x = np.zeros(dim)
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_x = x.copy()
                best_f = f

        def time_left():
            return deadline - time.time() > 5.0

        # Halton sequence for diverse starting points
        def halton_sequence(index, dim):
            """Generate a point in the Halton sequence for the given index and dimension."""
            result = []
            for i in range(1, dim + 1):
                f = 1.0
                r = 0.0
                idx = index
                while idx > 0:
                    f /= i
                    r += f * (idx % i)
                    idx //= i
                result.append(r)
            return np.array(result)

        def map_to_box(point):
            """Map a point in [0,1]^dim to [-5,5]^dim."""
            return -5.0 + 10.0 * point

        # Try multiple starting points using Halton sequence
        # Start with a few well-distributed points, then refine
        n_starts = 10
        starts = []
        for i in range(1, n_starts + 1):
            h = halton_sequence(i, dim)
            x = map_to_box(h)
            starts.append(x)
        
        # Also add center and corners
        starts.append(np.zeros(dim))
        for d in range(dim):
            for sign in [1, -1]:
                x = np.zeros(dim)
                x[d] = sign * 4.0
                starts.append(x)

        # Shuffle starts with a fixed seed for reproducibility
        rng = np.random.default_rng(42)
        rng.shuffle(starts)

        # Sort by distance to center (prefer center first for early progress)
        starts.sort(key=lambda x: np.linalg.norm(x))

        # Use remaining budget for local polishing
        def polish(x0, remaining_budget):
            """Polish a starting point using L-BFGS-B with limited evaluations."""
            nonlocal calls, best_x, best_f

            # Evaluate the starting point
            f0 = func(x0)
            calls += 1
            update_best(x0, f0)

            if calls >= budget or not time_left():
                return

            # Use scipy's L-BFGS-B with a custom function that tracks calls
            from scipy.optimize import minimize

            def wrapped_func(x):
                nonlocal calls
                if calls >= budget or not time_left():
                    return best_f
                x_clipped = np.clip(x, -5, 5)
                f = func(x_clipped)
                calls += 1
                update_best(x_clipped, f)
                return f

            try:
                result = minimize(
                    wrapped_func,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 50, 'maxfun': max(1, remaining_budget - calls - 5)}
                )
                if result.fun < best_f:
                    best_x = np.clip(result.x, -5, 5)
                    best_f = result.fun
            except Exception:
                pass

        # Try each starting point
        for x0 in starts:
            if calls >= budget or not time_left():
                break

            remaining = budget - calls
            if remaining < 10:
                break

            # Allocate a fraction of remaining budget to each start
            alloc = min(50, remaining // 2)
            if alloc < 5:
                break

            polish(x0, alloc)

            # If we've made good progress, use remaining budget for final polish
            if best_f < 100 and calls < budget and time_left():
                # Final polish from the best point found
                final_alloc = budget - calls
                if final_alloc > 10:
                    polish(best_x, final_alloc)
                break

        # If we still have budget, do a final polish from the best point
        if calls < budget and time_left():
            final_alloc = budget - calls
            if final_alloc > 5:
                polish(best_x, final_alloc)

        return best_x, best_f

# EVOLVE-BLOCK-END
