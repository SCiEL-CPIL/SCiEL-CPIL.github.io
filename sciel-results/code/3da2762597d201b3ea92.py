"""Rewritten optimizer using Nelder-Mead simplex with multi-start restarts."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead simplex with multi-start restarts on BBOB f6."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280.0  # safety margin before 300s limit

        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_made = 0

        def eval_point(x):
            nonlocal calls_made
            x = np.clip(x, -5.0, 5.0)
            calls_made += 1
            return func(x)

        # Multi-start: try several initial points
        # First, use a few structured starts, then random
        starts = []
        # Origin
        starts.append(np.zeros(dim))
        # Corners of a smaller box
        for corner in [(-2.0, -2.0, -2.0, -2.0, -2.0),
                       (2.0, 2.0, 2.0, 2.0, 2.0),
                       (-2.0, 2.0, -2.0, 2.0, -2.0),
                       (2.0, -2.0, 2.0, -2.0, 2.0)]:
            if dim == 5:
                starts.append(np.array(corner))
        # Random starts
        for _ in range(6):
            starts.append(rng.uniform(-4, 4, dim))

        # Evaluate all starts first
        for s in starts:
            if calls_made >= budget or time.time() > deadline:
                break
            f = eval_point(s)
            if f < best_f:
                best_f = f
                best_x = s.copy()

        # Now run Nelder-Mead from the best start
        if best_x is None:
            # Fallback: single random point
            x = rng.uniform(-5, 5, dim)
            best_f = eval_point(x)
            best_x = x.copy()

        # Nelder-Mead parameters
        alpha = 1.0   # reflection
        gamma = 2.0   # expansion
        rho = 0.5     # contraction
        sigma_nm = 0.5  # shrink

        def nelder_mead(start_x, max_calls):
            nonlocal best_x, best_f, calls_made

            # Initialize simplex: start_x + small offsets in each coordinate
            simplex = [start_x.copy()]
            f_vals = [eval_point(start_x)]
            if f_vals[0] < best_f:
                best_f = f_vals[0]
                best_x = start_x.copy()

            # Add d points with small perturbations
            for i in range(dim):
                if calls_made >= budget or time.time() > deadline:
                    return
                pt = start_x.copy()
                # Adaptive initial step size
                pt[i] += 0.5
                pt = np.clip(pt, -5.0, 5.0)
                f = eval_point(pt)
                simplex.append(pt)
                f_vals.append(f)
                if f < best_f:
                    best_f = f
                    best_x = pt.copy()

            n = len(simplex)  # should be dim+1

            # Main Nelder-Mead loop
            while calls_made < budget and time.time() < deadline:
                # Sort simplex by function value
                order = np.argsort(f_vals)
                simplex = [simplex[i] for i in order]
                f_vals = [f_vals[i] for i in order]

                # Check convergence
                f_range = f_vals[-1] - f_vals[0]
                if f_range < 1e-10:
                    break

                # Centroid of all points except worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5.0, 5.0)
                f_r = eval_point(x_r)
                if f_r < best_f:
                    best_f = f_r
                    best_x = x_r.copy()

                if f_r < f_vals[0]:
                    # Try expansion
                    x_e = centroid + gamma * (centroid - simplex[-1])
                    x_e = np.clip(x_e, -5.0, 5.0)
                    f_e = eval_point(x_e)
                    if f_e < f_r:
                        simplex[-1] = x_e
                        f_vals[-1] = f_e
                        if f_e < best_f:
                            best_f = f_e
                            best_x = x_e.copy()
                    else:
                        simplex[-1] = x_r
                        f_vals[-1] = f_r
                elif f_r < f_vals[-2]:
                    # Accept reflection
                    simplex[-1] = x_r
                    f_vals[-1] = f_r
                else:
                    # Contraction
                    x_c = centroid + rho * (simplex[-1] - centroid)
                    x_c = np.clip(x_c, -5.0, 5.0)
                    f_c = eval_point(x_c)
                    if f_c < best_f:
                        best_f = f_c
                        best_x = x_c.copy()

                    if f_c < f_vals[-1]:
                        simplex[-1] = x_c
                        f_vals[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, n):
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                            f_vals[i] = eval_point(simplex[i])
                            if f_vals[i] < best_f:
                                best_f = f_vals[i]
                                best_x = simplex[i].copy()

                # If we've made enough calls, restart
                if calls_made >= budget:
                    break

            # After Nelder-Mead converges or budget runs low, if we still have budget,
            # do a local coordinate descent from best_x
            if calls_made < budget and time.time() < deadline:
                self._coordinate_descent(func, best_x, best_f)

        nelder_mead(best_x, budget - calls_made)

        return best_x, best_f

    def _coordinate_descent(self, func, x0, f0):
        """Local coordinate descent refinement."""
        nonlocal_calls = 0
        dim = self.dim
        x = x0.copy()
        fx = f0
        step = 0.5

        for _ in range(50):  # max iterations
            if nonlocal_calls >= 100 or time.time() > 280.0:
                break
            improved = False
            for i in range(dim):
                for direction in [1, -1]:
                    y = x.copy()
                    y[i] += direction * step
                    y = np.clip(y, -5.0, 5.0)
                    nonlocal_calls += 1
                    fy = func(y)
                    if fy < fx:
                        x = y
                        fx = fy
                        improved = True
            if not improved:
                step *= 0.5
                if step < 1e-6:
                    break

        if fx < self._best_f:
            self._best_f = fx
            self._best_x = x.copy()

# Need to track best across methods
class Optimizer:
    """Nelder-Mead simplex with multi-start restarts on BBOB f6."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280.0

        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_made = 0

        def eval_point(x):
            nonlocal calls_made, best_f, best_x
            x = np.clip(x, -5.0, 5.0)
            calls_made += 1
            f = func(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Multi-start initial evaluations
        starts = [np.zeros(dim)]
        if dim == 5:
            for corner in [(-2.0, -2.0, -2.0, -2.0, -2.0),
                           (2.0, 2.0, 2.0, 2.0, 2.0),
                           (-2.0, 2.0, -2.0, 2.0, -2.0),
                           (2.0, -2.0, 2.0, -2.0, 2.0)]:
                starts.append(np.array(corner))
        for _ in range(6):
            starts.append(rng.uniform(-4, 4, dim))

        for s in starts:
            if calls_made >= budget or time.time() > deadline:
                break
            eval_point(s)

        if best_x is None:
            x = rng.uniform(-5, 5, dim)
            eval_point(x)

        # Nelder-Mead from best
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma_nm = 0.5

        simplex = [best_x.copy()]
        f_vals = [best_f]

        for i in range(dim):
            if calls_made >= budget or time.time() > deadline:
                break
            pt = best_x.copy()
            pt[i] += 0.5
            pt = np.clip(pt, -5.0, 5.0)
            f = eval_point(pt)
            simplex.append(pt)
            f_vals.append(f)

        n = len(simplex)

        while calls_made < budget and time.time() < deadline:
            order = np.argsort(f_vals)
            simplex = [simplex[i] for i in order]
            f_vals = [f_vals[i] for i in order]

            if f_vals[-1] - f_vals[0] < 1e-10:
                break

            centroid = np.mean(simplex[:-1], axis=0)

            x_r = centroid + alpha * (centroid - simplex[-1])
            x_r = np.clip(x_r, -5.0, 5.0)
            f_r = eval_point(x_r)

            if f_r < f_vals[0]:
                x_e = centroid + gamma * (centroid - simplex[-1])
                x_e = np.clip(x_e, -5.0, 5.0)
                f_e = eval_point(x_e)
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
                x_c = centroid + rho * (simplex[-1] - centroid)
                x_c = np.clip(x_c, -5.0, 5.0)
                f_c = eval_point(x_c)
                if f_c < f_vals[-1]:
                    simplex[-1] = x_c
                    f_vals[-1] = f_c
                else:
                    for i in range(1, n):
                        simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                        f_vals[i] = eval_point(simplex[i])

        # Coordinate descent refinement
        if calls_made < budget and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            step = 0.5
            for _ in range(30):
                if calls_made >= budget or time.time() > deadline:
                    break
                improved = False
                for i in range(dim):
                    for direction in [1, -1]:
                        if calls_made >= budget or time.time() > deadline:
                            break
                        y = x.copy()
                        y[i] += direction * step
                        y = np.clip(y, -5.0, 5.0)
                        fy = eval_point(y)
                        if fy < fx:
                            x = y
                            fx = fy
                            improved = True
                if not improved:
                    step *= 0.5
                    if step < 1e-6:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
