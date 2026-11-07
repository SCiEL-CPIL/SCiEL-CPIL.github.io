"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Deterministic grid sampling + multi-start Nelder-Mead for BBOB f21."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 310.0  # Safety margin before 330s kill
        n = self.dim
        budget = self.budget
        calls = 0
        best_x = None
        best_f = float('inf')

        def safe_call(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return float('inf')
            calls += 1
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = np.array(x)
            return fx

        def nelder_mead(func, x0, max_iters=50, tol=1e-10):
            """Nelder-Mead simplex method with budget awareness."""
            nonlocal calls, best_f, best_x
            if calls >= budget or time.time() > deadline - 5:
                return x0, func(x0)

            # Initialize simplex
            simplex = [x0.copy()]
            for i in range(n):
                x_new = x0.copy()
                x_new[i] += 0.5 if x0[i] < 0 else -0.5
                simplex.append(np.clip(x_new, -5, 5))

            # Evaluate initial simplex
            vals = []
            for s in simplex:
                if calls >= budget or time.time() > deadline - 5:
                    return x0, best_f
                v = safe_call(s)
                vals.append(v)

            for it in range(max_iters):
                if calls >= budget or time.time() > deadline - 5:
                    break

                # Sort by function value
                order = np.argsort(vals)
                simplex = [simplex[i] for i in order]
                vals = [vals[i] for i in order]

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                alpha = 1.0
                x_r = np.clip(centroid + alpha * (centroid - simplex[-1]), -5, 5)
                if calls >= budget or time.time() > deadline - 5:
                    break
                f_r = safe_call(x_r)

                if f_r < vals[0]:
                    # Expansion
                    gamma = 2.0
                    x_e = np.clip(centroid + gamma * (centroid - simplex[-1]), -5, 5)
                    if calls >= budget or time.time() > deadline - 5:
                        break
                    f_e = safe_call(x_e)
                    if f_e < f_r:
                        simplex[-1] = x_e
                        vals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        vals[-1] = f_r
                elif f_r < vals[-2]:
                    simplex[-1] = x_r
                    vals[-1] = f_r
                else:
                    # Contraction
                    rho = 0.5
                    x_c = np.clip(centroid + rho * (simplex[-1] - centroid), -5, 5)
                    if calls >= budget or time.time() > deadline - 5:
                        break
                    f_c = safe_call(x_c)
                    if f_c < vals[-1]:
                        simplex[-1] = x_c
                        vals[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = np.clip(simplex[0] + 0.5 * (simplex[i] - simplex[0]), -5, 5)
                            if calls >= budget or time.time() > deadline - 5:
                                return x0, best_f
                            vals[i] = safe_call(simplex[i])

                # Check convergence
                if max(vals) - min(vals) < tol:
                    break

            # Return best from simplex
            idx = np.argmin(vals)
            return simplex[idx], vals[idx]

        # Phase 1: Deterministic global sampling
        points_to_eval = []

        # Center
        points_to_eval.append(np.zeros(n))

        # Vertices: all combinations of -5 and 5
        for mask in range(2 ** n):
            v = np.array([5.0 if (mask >> i) & 1 else -5.0 for i in range(n)])
            points_to_eval.append(v)

        # Axis midpoints
        for i in range(n):
            for sign in [-1, 1]:
                mid = np.zeros(n)
                mid[i] = sign * 2.5
                points_to_eval.append(mid)

        # Additional structured points: face centers and edge midpoints
        # Face centers: one axis at 0, others at ±5
        for i in range(n):
            for mask in range(2 ** (n - 1)):
                v = np.zeros(n)
                j = 0
                for k in range(n):
                    if k != i:
                        v[k] = 5.0 if (mask >> j) & 1 else -5.0
                        j += 1
                points_to_eval.append(v)

        # Evaluate these points deterministically
        for p in points_to_eval:
            if calls >= budget or time.time() > deadline - 60:
                break
            safe_call(p)

        # Fallback
        if best_x is None:
            safe_call(np.zeros(n))

        # Phase 2: Multi-start Nelder-Mead from top diverse points
        if best_x is not None and calls < budget and time.time() < deadline - 10:
            # Collect all evaluated points with their values
            # We need to track them - let's re-evaluate a subset for diversity
            # Use the initial sampling points as starting candidates
            start_points = [best_x.copy()]

            # Add diverse starting points from the grid
            for p in points_to_eval[:20]:  # First 20 grid points as diverse starts
                if p is not None and np.linalg.norm(p - best_x) > 1.0:
                    start_points.append(p)

            # Sort start points by distance to best (prefer diverse but not too far)
            start_points.sort(key=lambda x: abs(np.linalg.norm(x - best_x) - 2.0))

            # Run Nelder-Mead from top diverse starts
            for sp in start_points:
                if calls >= budget or time.time() > deadline - 5:
                    break
                x_res, f_res = nelder_mead(func, sp, max_iters=30)
                if f_res < best_f:
                    best_f = f_res
                    best_x = x_res

            # If we have budget left, do a final refinement from best
            if calls < budget - 50 and time.time() < deadline - 10:
                x_res, f_res = nelder_mead(func, best_x, max_iters=80)
                if f_res < best_f:
                    best_f = f_res
                    best_x = x_res

        return best_x, best_f
# EVOLVE-BLOCK-END
