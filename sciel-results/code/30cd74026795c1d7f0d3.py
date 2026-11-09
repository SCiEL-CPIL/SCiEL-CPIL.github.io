"""Optimizer for BBOB f17 (Schaffer's F7) using Powell's method with restarts."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Powell's method based optimizer for BBOB f17 (Schaffer's F7)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 330s limit, leave 40s margin

        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_fx = np.inf
        calls_made = 0

        def call_func(x):
            nonlocal calls_made
            if calls_made >= budget:
                return np.inf
            fx = float(func(x))
            calls_made += 1
            return fx

        def time_ok():
            return time.time() < deadline and calls_made < budget

        # Phase 1: Initial sampling for early progress
        points = [np.zeros(d)]
        # Add some structured points
        for i in range(d):
            for val in [-3.0, -1.5, 1.5, 3.0]:
                x = np.zeros(d)
                x[i] = val
                points.append(x.copy())
        # Add random points
        for _ in range(10):
            points.append(rng.uniform(-4, 4, d))

        for x in points:
            if not time_ok():
                break
            x = np.clip(x, -5, 5)
            fx = call_func(x)
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(d)
            best_fx = call_func(best_x)

        # Phase 2: Powell's method with line search
        def powell_method(x0, max_calls):
            nonlocal best_x, best_fx, calls_made

            n = d
            # Initial simplex: x0 and x0 + delta * e_i
            delta = 0.5
            simplex = [x0.copy()]
            for i in range(n):
                pt = x0.copy()
                pt[i] += delta
                simplex.append(np.clip(pt, -5, 5))

            # Evaluate initial simplex
            fvals = [call_func(pt) for pt in simplex]

            local_best_idx = np.argmin(fvals)
            local_best_fx = fvals[local_best_idx]
            local_best_x = simplex[local_best_idx].copy()

            if local_best_fx < best_fx:
                best_fx = local_best_fx
                best_x = local_best_x.copy()

            max_iter = 30  # Maximum Powell iterations
            tol = 1e-8

            for iteration in range(max_iter):
                if not time_ok() or calls_made >= max_calls:
                    break

                # Sort simplex by function value
                order = np.argsort(fvals)
                simplex = [simplex[i].copy() for i in order]
                fvals = [fvals[i] for i in order]

                # Check convergence
                if abs(fvals[-1] - fvals[0]) < tol:
                    break

                # Find worst point
                worst_idx = n  # Last element after sorting
                worst_pt = simplex[worst_idx]
                worst_fx = fvals[worst_idx]

                # Centroid of all points except worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection of worst point through centroid
                reflected = 2.0 * centroid - worst_pt
                reflected = np.clip(reflected, -5, 5)
                ref_fx = call_func(reflected)

                if not time_ok() or calls_made >= max_calls:
                    break

                if ref_fx < fvals[0]:
                    # Expansion
                    expanded = centroid + 2.0 * (reflected - centroid)
                    expanded = np.clip(expanded, -5, 5)
                    exp_fx = call_func(expanded)

                    if not time_ok() or calls_made >= max_calls:
                        break

                    if exp_fx < ref_fx:
                        new_pt = expanded
                        new_fx = exp_fx
                    else:
                        new_pt = reflected
                        new_fx = ref_fx
                else:
                    # Contraction
                    contracted = centroid + 0.5 * (worst_pt - centroid)
                    contracted = np.clip(contracted, -5, 5)
                    con_fx = call_func(contracted)

                    if not time_ok() or calls_made >= max_calls:
                        break

                    if con_fx < worst_fx:
                        new_pt = contracted
                        new_fx = con_fx
                    else:
                        # Shrink
                        new_simplex = []
                        new_fvals = []
                        for i in range(n):
                            shrunken = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            shrunken = np.clip(shrunken, -5, 5)
                            sh_fx = call_func(shrunken)
                            new_simplex.append(shrunken)
                            new_fvals.append(sh_fx)
                        simplex = new_simplex
                        fvals = new_fvals
                        local_best_idx = np.argmin(fvals)
                        local_best_fx = fvals[local_best_idx]
                        local_best_x = simplex[local_best_idx].copy()
                        if local_best_fx < best_fx:
                            best_fx = local_best_fx
                            best_x = local_best_x.copy()
                        continue

                # Replace worst point
                simplex[worst_idx] = new_pt
                fvals[worst_idx] = new_fx

                local_best_idx = np.argmin(fvals)
                local_best_fx = fvals[local_best_idx]
                local_best_x = simplex[local_best_idx].copy()
                if local_best_fx < best_fx:
                    best_fx = local_best_fx
                    best_x = local_best_x.copy()

            return local_best_x, local_best_fx

        # Main Powell run
        if calls_made < budget and time.time() < deadline:
            remaining = budget - calls_made
            first_run_calls = min(remaining, 350)
            x0 = best_x.copy()
            _, _ = powell_method(x0, calls_made + first_run_calls)

        # Second run if budget allows
        if calls_made < budget and time.time() < deadline:
            remaining = budget - calls_made
            if remaining > 20:
                x0_2 = best_x + rng.normal(0, 0.3, d)
                x0_2 = np.clip(x0_2, -5, 5)
                fx2 = call_func(x0_2)
                if fx2 < best_fx:
                    best_fx = fx2
                    best_x = x0_2.copy()

                if calls_made < budget and time.time() < deadline:
                    second_run_calls = min(budget - calls_made, 40)
                    _, _ = powell_method(best_x.copy(), calls_made + second_run_calls)

        # Phase 3: Final polish
        if calls_made < budget and time.time() < deadline:
            polish_calls = min(15, budget - calls_made)
            for _ in range(polish_calls):
                if not time_ok():
                    break
                y = best_x + 0.01 * rng.standard_normal(d)
                y = np.clip(y, -5, 5)
                fy = call_func(y)
                if fy < best_fx:
                    best_fx = fy
                    best_x = y.copy()

        return best_x, best_fx
# EVOLVE-BLOCK-END
