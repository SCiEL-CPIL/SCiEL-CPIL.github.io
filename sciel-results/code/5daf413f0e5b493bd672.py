"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead simplex optimizer with adaptive restarts and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280.0  # Safety margin before 300s limit

        rng = np.random.default_rng(42)
        calls = [0]

        def safe_call(x):
            if calls[0] >= budget:
                return np.inf
            x = np.clip(x, -5, 5)
            try:
                val = float(func(x))
            except Exception:
                val = np.inf
            calls[0] += 1
            return val

        def nelder_mead(func_safe, x0, budget_left, deadline):
            dim = len(x0)
            # Initialize simplex
            simplex = [x0.copy() + 0.01 * np.ones(dim)]
            for i in range(dim):
                v = x0.copy()
                v[i] += 0.05
                simplex.append(v)
            simplex = np.array(simplex)
            fvals = np.array([func_safe(v) for v in simplex])

            # Nelder-Mead parameters
            alpha = 1.0   # reflection
            gamma = 2.0   # expansion
            rho = 0.5     # contraction
            sigma = 0.5   # shrink
            min_fval = np.min(fvals)

            while calls[0] < budget and time.time() < deadline:
                # Sort simplex by function value
                order = np.argsort(fvals)
                simplex = simplex[order]
                fvals = fvals[order]
                min_fval = fvals[0]

                # Check for convergence
                if np.ptp(fvals) < 1e-10:
                    break

                # Centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                fr = func_safe(xr)

                if fr < fvals[0]:
                    # Try expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    fe = func_safe(xe)
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                elif fr < fvals[-2]:
                    # Accept reflection
                    simplex[-1] = xr
                    fvals[-1] = fr
                else:
                    # Contraction
                    if fr < fvals[-1]:
                        # Outside contraction
                        xc = centroid + rho * (simplex[-1] - centroid)
                    else:
                        # Inside contraction
                        xc = centroid - rho * (centroid - simplex[-1])
                    fc = func_safe(xc)
                    if fc < min(fvals[-1], fr):
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            fvals[i] = func_safe(simplex[i])
                        break

                if calls[0] >= budget:
                    break

            order = np.argsort(fvals)
            return simplex[order[0]], fvals[order[0]]

        # Multi-start strategy
        best_x = None
        best_f = np.inf
        num_starts = 5

        for s in range(num_starts):
            if calls[0] >= budget or time.time() >= deadline:
                break

            if s == 0:
                x0 = rng.uniform(-5, 5, dim)
            else:
                # Random restarts with some spread
                x0 = rng.uniform(-4, 4, dim)

            x_best, f_best = nelder_mead(safe_call, x0, budget - calls[0], deadline)

            if f_best < best_f:
                best_f = f_best
                best_x = x_best

            # If we've used more than half the budget, stop restarting
            if calls[0] > budget // 2:
                break

        # Final local refinement around best point if budget remains
        if best_x is not None and calls[0] < budget and time.time() < deadline:
            x_ref, f_ref = nelder_mead(safe_call, best_x, budget - calls[0], deadline)
            if f_ref < best_f:
                best_f = f_ref
                best_x = x_ref

        if best_x is None:
            # Fallback: return the first point if no evaluation was made
            x0 = rng.uniform(-5, 5, dim)
            best_x = x0
            best_f = safe_call(x0)

        return best_x, best_f
# EVOLVE-BLOCK-END
