"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart Nelder-Mead for Discus (f11)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290.0

        rng = np.random.default_rng(99)
        best_x = None
        best_fx = np.inf
        calls = 0

        def time_ok():
            return time.time() < deadline

        def calls_ok():
            return calls < budget

        # Nelder-Mead implementation
        def nelder_mead(x0, max_iter, alpha=1.0, gamma=2.0, rho=0.5, sigma=2.0):
            nonlocal calls, best_x, best_fx
            n = len(x0)
            # Create initial simplex
            simplex = [x0.copy()]
            for i in range(n):
                xi = x0.copy()
                xi[i] += 0.25 if x0[i] < 4.75 else -0.25
                simplex.append(np.clip(xi, -5, 5))

            fvals = []
            for v in simplex:
                if not time_ok() or not calls_ok():
                    return
                fv = func(v)
                calls += 1
                fvals.append(fv)
                if fv < best_fx:
                    best_fx = fv
                    best_x = v.copy()

            for _ in range(max_iter):
                if not time_ok() or not calls_ok():
                    break

                # Sort simplex
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fr = func(xr)
                calls += 1
                if fr < best_fx:
                    best_fx = fr
                    best_x = xr.copy()

                if fvals[0] <= fr < fvals[-2]:
                    simplex[-1] = xr
                    fvals[-1] = fr
                elif fr < fvals[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fe = func(xe)
                    calls += 1
                    if fe < best_fx:
                        best_fx = fe
                        best_x = xe.copy()
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                else:
                    # Contraction
                    if fr < fvals[-1]:
                        # Outside contraction
                        xc = centroid + rho * (xr - centroid)
                    else:
                        # Inside contraction
                        xc = centroid - rho * (centroid - simplex[-1])
                    xc = np.clip(xc, -5, 5)
                    fc = func(xc)
                    calls += 1
                    if fc < best_fx:
                        best_fx = fc
                        best_x = xc.copy()

                    if fc < min(fr, fvals[-1]):
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            if not time_ok() or not calls_ok():
                                return
                            fi = func(simplex[i])
                            calls += 1
                            fvals[i] = fi
                            if fi < best_fx:
                                best_fx = fi
                                best_x = simplex[i].copy()

                # Check convergence
                if max(fvals) - min(fvals) < 1e-10:
                    break

        # Phase 1: Initial sampling
        n_init = min(15, budget)
        for i in range(n_init):
            if not time_ok() or not calls_ok():
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            calls += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

        if best_x is None:
            return np.zeros(dim), 0.0

        # Phase 2: Nelder-Mead from best point
        # Use remaining budget
        remaining = budget - calls
        # Nelder-Mead uses (dim+1) evals per iteration, so max_iter ~ remaining/(dim+1)
        max_iter = max(1, remaining // (dim + 2))
        nelder_mead(best_x.copy(), max_iter)

        # Phase 3: If we have budget left, do another NM from a perturbed best
        if calls < budget and time_ok():
            x2 = best_x + 0.5 * rng.standard_normal(dim)
            x2 = np.clip(x2, -5, 5)
            remaining2 = budget - calls
            max_iter2 = max(1, remaining2 // (dim + 2))
            nelder_mead(x2, max_iter2)

        return best_x, best_fx
# EVOLVE-BLOCK-END
