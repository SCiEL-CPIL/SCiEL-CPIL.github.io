"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead for f22 (Gallagher 21 peaks)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        rng = np.random.default_rng(42)

        best_x = None
        best_f = float('inf')
        calls_made = 0
        budget = self.budget
        dim = self.dim

        def time_ok():
            return time.time() < deadline

        def nelder_mead(x0, max_iter=300):
            nonlocal best_x, best_f, calls_made
            n = len(x0)
            # Build simplex
            simplex = [x0.copy()]
            for i in range(n):
                xi = x0.copy()
                if x0[i] + 0.5 <= 5.0:
                    xi[i] += 0.5
                else:
                    xi[i] -= 0.5
                simplex.append(xi)

            # Evaluate initial simplex
            for i in range(n + 1):
                if not time_ok() or calls_made >= budget:
                    return
                simplex[i] = np.clip(simplex[i], -5, 5)
                simplex[i] = func(simplex[i])
                calls_made += 1
                if simplex[i] < best_f:
                    best_f = simplex[i]
                    best_x = simplex[i].copy()

            alpha = 1.0  # reflection
            gamma = 2.0  # expansion
            rho = 0.5    # contraction
            sigma = 0.5  # shrink

            for _ in range(max_iter):
                if not time_ok() or calls_made >= budget:
                    break

                # Sort simplex by function value
                order = np.argsort(simplex, axis=0)[:, 0]
                simplex = simplex[order]

                # Check convergence
                if simplex[-1] - simplex[0] < 1e-10:
                    break

                # Centroid of all but worst
                centroid = np.mean(simplex[:n], axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[n])
                xr = np.clip(xr, -5, 5)
                fr = func(xr)
                calls_made += 1
                if fr < best_f:
                    best_f = fr
                    best_x = xr.copy()

                if fr < simplex[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[n])
                    xe = np.clip(xe, -5, 5)
                    fe = func(xe)
                    calls_made += 1
                    if fe < best_f:
                        best_f = fe
                        best_x = xe.copy()
                    if fe < fr:
                        simplex[n] = xe
                        simplex[n] = fe
                    else:
                        simplex[n] = xr
                        simplex[n] = fr
                elif fr < simplex[n - 1]:
                    simplex[n] = xr
                    simplex[n] = fr
                else:
                    # Contraction
                    if fr < simplex[n]:
                        xc = centroid + rho * (simplex[n] - centroid)
                        xc = np.clip(xc, -5, 5)
                        fc = func(xc)
                        calls_made += 1
                        if fc < best_f:
                            best_f = fc
                            best_x = xc.copy()
                        if fc < fr:
                            simplex[n] = xc
                            simplex[n] = fc
                        else:
                            # Shrink
                            for i in range(1, n + 1):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], -5, 5)
                                simplex[i] = func(simplex[i])
                                calls_made += 1
                                if simplex[i] < best_f:
                                    best_f = simplex[i]
                                    best_x = simplex[i].copy()
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            simplex[i] = func(simplex[i])
                            calls_made += 1
                            if simplex[i] < best_f:
                                best_f = simplex[i]
                                best_x = simplex[i].copy()

        # Phase 1: Random sampling to find diverse starting points
        n_random = 15
        random_points = []
        for i in range(n_random):
            if not time_ok() or calls_made >= budget:
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            random_points.append((fx, x))

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls_made += 1

        # Phase 2: Run Nelder-Mead from top diverse starting points
        random_points.sort(key=lambda t: t[0])
        # Use top 5 diverse points
        used = []
        for i, (fx, x) in enumerate(random_points):
            if not time_ok() or calls_made >= budget:
                break
            # Check diversity: at least 1.0 distance from all used points
            diverse = True
            for ux in used:
                if np.linalg.norm(x - ux) < 1.0:
                    diverse = False
                    break
            if diverse:
                used.append(x.copy())
                nelder_mead(x.copy(), max_iter=200)
                if len(used) >= 5:
                    break

        # If we still have budget, do a final random search to find better basins
        while calls_made < budget and time_ok():
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
                # Try Nelder-Mead from this new best
                if time_ok() and calls_made < budget - 10:
                    nelder_mead(x.copy(), max_iter=100)

        return best_x, best_f
# EVOLVE-BLOCK-END
