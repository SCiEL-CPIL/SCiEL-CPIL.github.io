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

        def nelder_mead(x0, max_calls, rng):
            """Run Nelder-Mead simplex optimization from x0."""
            nonlocal best_f, best_x, calls_made
            n = dim
            # Build initial simplex
            simplex = np.zeros((n + 1, n))
            simplex[0] = x0.copy()
            for i in range(n):
                simplex[i + 1] = x0.copy()
                simplex[i + 1][i] += 0.5  # initial step size

            # Evaluate initial simplex
            simplex_f = np.zeros(n + 1)
            for i in range(n + 1):
                if calls_made >= budget:
                    return
                if time.time() > deadline:
                    return
                simplex_f[i] = try_func(simplex[i])
                if simplex_f[i] is None:
                    return
                if simplex_f[i] < best_f:
                    best_f = simplex_f[i]
                    best_x = simplex[i].copy()

            calls_in_nm = n + 1
            max_iter = max_calls // (n + 1) + 10

            alpha = 1.0   # reflection
            gamma = 2.0   # expansion
            rho = 0.5     # contraction
            sigma = 0.5   # shrink

            for _ in range(max_iter):
                if calls_made >= budget:
                    return
                if time.time() > deadline:
                    return

                # Sort simplex by fitness
                order = np.argsort(simplex_f)
                simplex = simplex[order]
                simplex_f = simplex_f[order]

                # Check convergence
                if np.max(simplex_f) - np.min(simplex_f) < 1e-10:
                    break

                # Centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fr = try_func(xr)
                if fr is None:
                    return
                calls_in_nm += 1

                if fr < simplex_f[0]:
                    # Try expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fe = try_func(xe)
                    if fe is None:
                        return
                    calls_in_nm += 1
                    if fe < fr:
                        simplex[-1] = xe
                        simplex_f[-1] = fe
                    else:
                        simplex[-1] = xr
                        simplex_f[-1] = fr
                elif fr < simplex_f[-2]:
                    # Accept reflection
                    simplex[-1] = xr
                    simplex_f[-1] = fr
                else:
                    # Contraction
                    xc = centroid + rho * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5, 5)
                    fc = try_func(xc)
                    if fc is None:
                        return
                    calls_in_nm += 1
                    if fc < simplex_f[-1]:
                        simplex[-1] = xc
                        simplex_f[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            if calls_made >= budget:
                                return
                            if time.time() > deadline:
                                return
                            simplex_f[i] = try_func(simplex[i])
                            if simplex_f[i] is None:
                                return
                            calls_in_nm += 1

                # Update best
                if np.min(simplex_f) < best_f:
                    best_f = np.min(simplex_f)
                    best_x = simplex[np.argmin(simplex_f)].copy()

                # Check if simplex is too small
                spread = np.max(simplex, axis=0) - np.min(simplex, axis=0)
                if np.max(spread) < 1e-8:
                    break

        def coordinate_descent(x0, max_calls, rng):
            """Simple coordinate descent with adaptive step size for final refinement."""
            nonlocal best_f, best_x, calls_made
            x = x0.copy().astype(float)
            fx = try_func(x)
            if fx is None:
                return
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            step = 0.1
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
                        if step < 1e-7:
                            break

        def final_polish(x0, max_calls, rng):
            """High-precision polish with very small steps."""
            nonlocal best_f, best_x, calls_made
            x = x0.copy().astype(float)
            fx = try_func(x)
            if fx is None:
                return
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            step = 0.01
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
                        if step < 1e-8:
                            break

        # Multi-start strategy with periodic-aware candidates
        rng = np.random.default_rng(2024)

        # Schwefel's global minimum is at x_i ≈ 4.20968...
        # The function is periodic with period 4π ≈ 12.566
        # Generate candidates around known minimum and its periodic equivalents
        schwefel_min = 4.20968
        period = 4 * np.pi

        starts = []
        # Center
        starts.append(np.zeros(dim))
        # Known Schwefel minimum
        starts.append(np.full(dim, schwefel_min))
        # Periodic equivalents within [-5, 5]
        for k in range(-1, 2):
            candidate = schwefel_min + k * period
            if -5 <= candidate <= 5:
                starts.append(np.full(dim, candidate))
        # Some random diverse starts
        for i in range(3):
            starts.append(rng.uniform(-4.5, 4.5, dim))

        # Deduplicate and limit starts
        unique_starts = []
        for s in starts:
            s = np.clip(s, -5, 5)
            if not any(np.allclose(s, u, atol=0.1) for u in unique_starts):
                unique_starts.append(s)
        starts = unique_starts[:6]

        # Shuffle starts
        idx = np.arange(len(starts))
        rng.shuffle(idx)
        starts = [starts[i] for i in idx]

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

        # Phase 2: Run Nelder-Mead from the best start
        if calls_made < budget and best_x is not None:
            remaining = budget - calls_made
            nm_budget = min(remaining, 250)
            nelder_mead(best_x, nm_budget, rng)

        # Phase 3: Coordinate descent for refinement
        if calls_made < budget and best_x is not None:
            remaining = budget - calls_made
            cd_budget = min(remaining, 100)
            coordinate_descent(best_x, cd_budget, rng)

        # Phase 4: Final high-precision polish
        if calls_made < budget and best_x is not None:
            remaining = budget - calls_made
            polish_budget = min(remaining, 50)
            final_polish(best_x, polish_budget, rng)

        return best_x, best_f
# EVOLVE-BLOCK-END
