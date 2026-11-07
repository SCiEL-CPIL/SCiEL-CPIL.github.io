import numpy as np
import time

class Optimizer:
    """
    Optimizer using Nelder-Mead with adaptive restarts for BBOB f13 (Sharp Ridge).
    Nelder-Mead is robust for low-dimensional problems and handles ridges well
    when combined with restarts to escape local minima.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290.0  # Safety margin for 330s limit

        calls = 0
        best_x = None
        best_f = float('inf')
        rng = np.random.default_rng(42)

        def safe_call(x):
            nonlocal calls, best_f, best_x
            if calls >= budget:
                return None
            if time.time() > deadline:
                return None
            calls += 1
            try:
                x_clipped = np.clip(x, -5.0, 5.0)
                f_val = func(x_clipped)
                if f_val < best_f:
                    best_f = f_val
                    best_x = x_clipped.copy()
                return f_val
            except Exception:
                return None

        # Phase 1: Global Exploration
        # Sample points to find the basin of attraction.
        n_random = 15
        for _ in range(n_random):
            if calls >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5.0, 5.0, size=dim)
            f_val = safe_call(x)
            if f_val is None:
                break

        if best_x is None:
            # Fallback
            x0 = np.zeros(dim)
            f0 = safe_call(x0)
            if f0 is None:
                return best_x, best_f
            best_x = x0
            best_f = f0

        # Nelder-Mead with restarts
        def nelder_mead(x_start, f_start):
            nonlocal calls, best_f, best_x

            # Initialize simplex
            simplex = [x_start.copy()]
            f_vals = [f_start]

            # Add initial points
            for i in range(dim):
                x_new = x_start.copy()
                if i == 0:
                    x_new[i] += 0.1
                else:
                    x_new[i] += 0.1
                f_new = safe_call(x_new)
                if f_new is None:
                    return
                simplex.append(x_new)
                f_vals.append(f_new)

            alpha = 1.0   # reflection
            gamma = 2.0   # expansion
            rho = 0.5     # contraction
            sigma = 0.5   # shrink

            max_iterations = 100
            for it in range(max_iterations):
                if calls >= budget or time.time() > deadline:
                    break

                # Sort simplex by function values
                order = np.argsort(f_vals)
                simplex = [simplex[i] for i in order]
                f_vals = [f_vals[i] for i in order]

                # Check convergence
                if abs(f_vals[-1] - f_vals[0]) < 1e-10:
                    break

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                f_r = safe_call(x_r)
                if f_r is None:
                    break

                if f_r < f_vals[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    f_e = safe_call(x_e)
                    if f_e is None:
                        break
                    if f_e < f_r:
                        simplex[-1] = x_e
                        f_vals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        f_vals[-1] = f_r
                elif f_r < f_vals[-2]:
                    simplex[-1] = x_r
                    f_vals[-1] = f_r
                else:
                    # Contraction
                    if f_r < f_vals[-1]:
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        x_c = centroid - rho * (simplex[-1] - centroid)
                    f_c = safe_call(x_c)
                    if f_c is None:
                        break
                    if f_c < f_r:
                        simplex[-1] = x_c
                        f_vals[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            f_new = safe_call(simplex[i])
                            if f_new is None:
                                return
                            f_vals[i] = f_new

                # Update global best
                idx_min = np.argmin(f_vals)
                if f_vals[idx_min] < best_f:
                    best_f = f_vals[idx_min]
                    best_x = simplex[idx_min].copy()

        # Run Nelder-Mead with restarts
        x_start = best_x.copy()
        f_start = best_f

        # Restart strategy: if no improvement, try a new random start
        n_restarts = 5
        for r in range(n_restarts):
            if calls >= budget or time.time() > deadline:
                break

            if r == 0:
                nelder_mead(x_start, f_start)
            else:
                # New random start
                x_new = rng.uniform(-5.0, 5.0, size=dim)
                f_new = safe_call(x_new)
                if f_new is None:
                    break
                nelder_mead(x_new, f_new)

        return best_x, best_f
