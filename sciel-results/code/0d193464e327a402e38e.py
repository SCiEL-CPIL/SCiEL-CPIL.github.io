"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Deterministic Nelder-Mead with multi-scale restarts for Büche-Rastrigin."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # 10s safety margin before 330s kill
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_made = 0

        def call_func(x):
            nonlocal calls_made
            calls_made += 1
            return func(x)

        def nelder_mead(x0, remaining_budget, sigma=1.0):
            nonlocal best_x, best_f, calls_made
            nonlocal deadline

            # Initialize simplex
            n = dim
            simplex = np.zeros((n + 1, n))
            simplex[0] = x0
            for i in range(n):
                simplex[i + 1] = x0.copy()
                simplex[i + 1][i] += sigma

            # Evaluate initial simplex
            fvals = np.zeros(n + 1)
            for i in range(n + 1):
                if calls_made >= budget or time.time() > deadline:
                    break
                fvals[i] = call_func(simplex[i])
                if fvals[i] < best_f:
                    best_f = fvals[i]
                    best_x = simplex[i].copy()

            if calls_made >= budget or time.time() > deadline:
                return

            # Nelder-Mead iterations
            alpha = 1.0
            gamma = 2.0
            rho = 0.5

            while calls_made < budget and time.time() < deadline:
                # Sort simplex by function value
                order = np.argsort(fvals)
                simplex = simplex[order]
                fvals = fvals[order]

                # Centroid of all but worst
                centroid = simplex[:-1].mean(axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fr = call_func(xr)
                if fr < best_f:
                    best_f = fr
                    best_x = xr.copy()

                if fr < fvals[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fe = call_func(xe)
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                        if fe < best_f:
                            best_f = fe
                            best_x = xe.copy()
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                elif fr < fvals[-2]:
                    simplex[-1] = xr
                    fvals[-1] = fr
                else:
                    # Contraction
                    xc = centroid + rho * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5, 5)
                    fc = call_func(xc)
                    if fc < fvals[-1]:
                        simplex[-1] = xc
                        fvals[-1] = fc
                        if fc < best_f:
                            best_f = fc
                            best_x = xc.copy()
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            if calls_made >= budget or time.time() > deadline:
                                break
                            fvals[i] = call_func(simplex[i])
                            if fvals[i] < best_f:
                                best_f = fvals[i]
                                best_x = simplex[i].copy()

                        if calls_made >= budget or time.time() > deadline:
                            break

                # Check for convergence
                if np.max(fvals) - np.min(fvals) < 1e-10:
                    break

            return simplex[0].copy()

        # Phase 1: Deterministic multi-scale search
        # Start with a broad simplex spanning the box
        x0 = np.zeros(dim)
        sigma_large = 4.0
        nelder_mead(x0, budget, sigma_large)

        # Phase 2: Local refinement from best found
        if best_x is not None and calls_made < budget and time.time() < deadline:
            nelder_mead(best_x, budget, sigma=0.5)

        # Phase 3: Additional restarts from different points
        restart_points = [
            np.array([4.0] * dim),
            np.array([-4.0] * dim),
            np.array([4.0, -4.0, 4.0, -4.0, 4.0]) if dim >= 5 else np.array([4.0, -4.0, 4.0, -4.0, 4.0][:dim]),
            np.array([-4.0, 4.0, -4.0, 4.0, -4.0]) if dim >= 5 else np.array([-4.0, 4.0, -4.0, 4.0, -4.0][:dim]),
        ]

        for rp in restart_points:
            if calls_made >= budget or time.time() > deadline:
                break
            if best_x is not None:
                # Only restart if we haven't found a very good solution yet
                nelder_mead(rp, budget, sigma=2.0)
            else:
                nelder_mead(rp, budget, sigma=2.0)

        # Phase 4: Fine-grained local search around best
        if best_x is not None and calls_made < budget and time.time() < deadline:
            nelder_mead(best_x, budget, sigma=0.1)

        if best_x is None:
            best_x = np.zeros(dim)

        return best_x, best_f
# EVOLVE-BLOCK-END
