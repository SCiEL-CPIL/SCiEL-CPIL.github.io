"""Direct-structured optimizer for BBOB f20 (Schwefel) with deadline guard."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """
    Multi-start Nelder-Mead with coordinate descent refinement for BBOB f20 (Schwefel).
    Uses Nelder-Mead's geometric operations to navigate the rugged Schwefel landscape,
    with diverse starts to handle hidden shifts/rotations.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 275  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_made = 0

        def try_func(x):
            nonlocal calls_made
            if calls_made >= budget:
                return None
            fx = func(x)
            calls_made += 1
            return fx

        def nelder_mead(x0, max_calls, rng, init_step=0.5):
            """Nelder-Mead simplex method from x0 with adaptive initialization."""
            nonlocal best_f, best_x, calls_made
            n = dim
            x = x0.copy().astype(float)
            x = np.clip(x, -5, 5)
            fx = try_func(x)
            if fx is None:
                return
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            # Initialize simplex with diverse vertices
            simplex = [x.copy()]
            simplex_f = [fx]
            for i in range(n):
                xi = x.copy()
                # Adaptive step: larger if far from center, smaller if near
                if abs(xi[i]) > 4.0:
                    step_i = -init_step * 0.5
                else:
                    step_i = init_step
                xi[i] += step_i
                xi = np.clip(xi, -5, 5)
                fxi = try_func(xi)
                if fxi is None:
                    return
                simplex.append(xi)
                simplex_f.append(fxi)
                if fxi < best_f:
                    best_f = fxi
                    best_x = xi.copy()

            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            min_step = 1e-10
            max_iter = max_calls
            stagnant = 0

            for _ in range(max_iter):
                if calls_made >= budget:
                    return
                if time.time() > deadline:
                    return

                # Sort simplex
                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]

                # Check convergence
                if abs(simplex_f[0] - simplex_f[-1]) < 1e-12:
                    break

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fxr = try_func(xr)
                if fxr is None:
                    return

                if fxr < simplex_f[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fxe = try_func(xe)
                    if fxe is None:
                        return
                    if fxe < fxr:
                        simplex[-1] = xe
                        simplex_f[-1] = fxe
                    else:
                        simplex[-1] = xr
                        simplex_f[-1] = fxr
                    stagnant = 0
                elif fxr < simplex_f[-2]:
                    simplex[-1] = xr
                    simplex_f[-1] = fxr
                    stagnant = 0
                else:
                    # Contraction
                    if fxr < simplex_f[-1]:
                        # Outside contraction
                        xc = centroid + rho * (xr - centroid)
                    else:
                        # Inside contraction
                        xc = centroid - rho * (centroid - xr)
                    xc = np.clip(xc, -5, 5)
                    fxc = try_func(xc)
                    if fxc is None:
                        return
                    if fxc < min(fxr, simplex_f[-1]):
                        simplex[-1] = xc
                        simplex_f[-1] = fxc
                        stagnant = 0
                    else:
                        # Shrink
                        stagnant += 1
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fsi = try_func(simplex[i])
                            if fsi is None:
                                return
                            simplex_f[i] = fsi
                        # If stuck, try a small random perturbation
                        if stagnant > 5:
                            for i in range(1, n + 1):
                                simplex[i] = simplex[0] + rng.normal(0, 0.1, n)
                                simplex[i] = np.clip(simplex[i], -5, 5)
                                fsi = try_func(simplex[i])
                                if fsi is None:
                                    return
                                simplex_f[i] = fsi
                            stagnant = 0

                # Update best
                for i in range(n + 1):
                    if simplex_f[i] < best_f:
                        best_f = simplex_f[i]
                        best_x = simplex[i].copy()

                # Check simplex size
                spread = max(np.linalg.norm(simplex[0] - simplex[i]) for i in range(1, n + 1))
                if spread < min_step:
                    break

        def coordinate_polish(x0, max_calls, rng):
            """High-precision coordinate descent with adaptive step for final refinement."""
            nonlocal best_f, best_x, calls_made
            x = x0.copy().astype(float)
            fx = try_func(x)
            if fx is None:
                return
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            step = 0.05
            for _ in range(max_calls):
                if calls_made >= budget:
                    return
                if time.time() > deadline:
                    return

                improved = False
                for i in range(dim):
                    if calls_made >= budget:
                        return
                    if time.time() > deadline:
                        return

                    for sign in [1.0, -1.0]:
                        y = x.copy()
                        y[i] += sign * step
                        y = np.clip(y, -5, 5)
                        fy = try_func(y)
                        if fy is None:
                            return
                        if fy < fx:
                            x, fx = y, fy
                            improved = True
                            if fx < best_f:
                                best_f = fx
                                best_x = x.copy()
                            break

                if not improved:
                    step *= 0.5
                    if step < 1e-9:
                        break

        # Diverse starting points
        rng = np.random.default_rng(42)

        starts = [
            np.zeros(dim),
            np.full(dim, 4.20968),  # Schwefel global minimum
            np.full(dim, -4.20968),
            np.full(dim, 2.0),
            np.full(dim, -2.0),
        ]
        # Add some random starts
        for _ in range(3):
            starts.append(rng.uniform(-4.5, 4.5, dim))

        # Phase 1: Evaluate all starts
        for x0 in starts:
            if calls_made >= budget:
                break
            if time.time() > deadline:
                break
            fx = try_func(x0)
            if fx is None:
                break
            if fx < best_f:
                best_f = fx
                best_x = x0.copy()

        if best_x is None:
            return None

        # Phase 2: Main Nelder-Mead from best start
        if calls_made < budget and best_x is not None:
            remaining = budget - calls_made
            nm_budget = min(remaining, 350)
            nelder_mead(best_x, nm_budget, rng, init_step=0.8)

        # Phase 3: Second Nelder-Mead with smaller step for refinement
        if calls_made < budget and best_x is not None:
            remaining = budget - calls_made
            if remaining > 50:
                nm2_budget = min(remaining, 100)
                nelder_mead(best_x, nm2_budget, rng, init_step=0.1)

        # Phase 4: Final coordinate polish
        if calls_made < budget and best_x is not None:
            remaining = budget - calls_made
            polish_budget = min(remaining, 40)
            coordinate_polish(best_x, polish_budget, rng)

        return best_x, best_f
# EVOLVE-BLOCK-END
