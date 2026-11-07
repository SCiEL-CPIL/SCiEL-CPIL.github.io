"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead optimizer for BBOB-style Katsuura function."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 275.0  # Safety margin

        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_made = 0

        def counted_func(x):
            nonlocal calls_made, best_f, best_x
            if calls_made >= self.budget:
                raise RuntimeError("Budget exceeded")
            if time.time() > deadline - 3:
                raise RuntimeError("Time limit")
            calls_made += 1
            val = float(func(x))
            if val < best_f:
                best_f = val
                best_x = np.array(x, copy=True)
            return val

        dim = self.dim
        budget = self.budget

        def nelder_mead(start_point, max_iter=2000):
            """Nelder-Mead simplex optimization."""
            nonlocal best_f, best_x, calls_made

            # Build initial simplex
            simplex = [start_point.copy()]
            for i in range(dim):
                new_point = start_point.copy()
                step = 0.05 if abs(start_point[i]) > 0.05 else 0.05
                new_point[i] += step
                simplex.append(new_point)

            # Evaluate initial simplex
            values = []
            for x in simplex:
                try:
                    val = counted_func(x)
                    values.append(val)
                except RuntimeError:
                    return

            # Nelder-Mead parameters
            alpha = 1.0   # reflection
            gamma = 2.0   # expansion
            rho = 0.5     # contraction
            sigma_nm = 0.5  # shrink

            for _ in range(max_iter):
                if calls_made >= budget or time.time() > deadline - 3:
                    return

                # Sort by function value
                order = np.argsort(values)
                simplex = [simplex[i] for i in order]
                values = [values[i] for i in order]

                # Check convergence
                if abs(values[-1] - values[0]) < 1e-12:
                    break

                # Centroid of all points except the worst
                centroid = np.mean(simplex[:-1], axis=0)
                worst = simplex[-1]
                worst_val = values[-1]

                # Reflection
                x_r = centroid + alpha * (centroid - worst)
                try:
                    f_r = counted_func(x_r)
                except RuntimeError:
                    return

                if f_r < values[0]:
                    # Expansion
                    x_e = centroid + gamma * (centroid - worst)
                    try:
                        f_e = counted_func(x_e)
                    except RuntimeError:
                        return

                    if f_e < f_r:
                        simplex[-1] = x_e
                        values[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        values[-1] = f_r
                elif f_r < values[-2]:
                    # Simple reflection
                    simplex[-1] = x_r
                    values[-1] = f_r
                else:
                    # Contraction
                    if f_r < worst_val:
                        # Outside contraction
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        # Inside contraction
                        x_c = centroid - rho * (centroid - worst)

                    try:
                        f_c = counted_func(x_c)
                    except RuntimeError:
                        return

                    if f_c < min(f_r, worst_val):
                        simplex[-1] = x_c
                        values[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            try:
                                values[i] = counted_func(simplex[i])
                            except RuntimeError:
                                return

                # Update best
                min_idx = np.argmin(values)
                if values[min_idx] < best_f:
                    best_f = values[min_idx]
                    best_x = np.array(simplex[min_idx], copy=True)

        # Phase 1: Initial exploration with diverse starting points
        # Use a grid-like plus random approach for better coverage
        n_initial = 15
        start_points = []

        # Center point
        start_points.append(np.zeros(dim))

        # Axis-aligned points
        for i in range(dim):
            for sign in [1, -1]:
                p = np.zeros(dim)
                p[i] = sign * 2.5
                start_points.append(p)

        # Random points
        while len(start_points) < n_initial:
            p = rng.uniform(-4, 4, dim)
            start_points.append(p)

        # Evaluate all initial points first
        for p in start_points:
            try:
                counted_func(p)
            except RuntimeError:
                break

        # Phase 2: Run Nelder-Mead from best initial points
        # Sort initial points by their function values
        evaluated_points = []
        for p in start_points:
            try:
                val = counted_func(p)
                evaluated_points.append((val, p))
            except RuntimeError:
                break

        evaluated_points.sort(key=lambda x: x[0])

        # Run NM from top diverse points
        max_restarts = 8
        for i in range(min(max_restarts, len(evaluated_points))):
            if calls_made >= budget or time.time() > deadline - 10:
                break

            # Skip if too similar to previous starts
            start_p = evaluated_points[i][1].copy()
            nelder_mead(start_p, max_iter=1500)

        # Phase 3: Final polish from the best point found
        if best_x is not None and calls_made < budget and time.time() < deadline - 10:
            # Create a tighter simplex around the best point
            polish_point = best_x.copy()
            nelder_mead(polish_point, max_iter=1000)

        # Phase 4: If budget remains, do a few more random restarts
        while calls_made < budget - 50 and time.time() < deadline - 15:
            p = rng.uniform(-4, 4, dim)
            # Add some jitter to best point
            p = best_x + rng.normal(0, 0.1, dim)
            p = np.clip(p, -5, 5)
            try:
                counted_func(p)
            except RuntimeError:
                break

            if time.time() > deadline - 10:
                break
            nelder_mead(p, max_iter=500)

        return best_x, best_f
# EVOLVE-BLOCK-END
