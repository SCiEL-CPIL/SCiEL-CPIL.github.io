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

        def nelder_mead(x0, max_evals=40, tol=1e-8):
            """Nelder-Mead simplex with budget awareness."""
            nonlocal calls, best_x, best_f
            if calls >= budget or time.time() > deadline - 5:
                return

            # Initialize simplex
            simplex = [x0.copy()]
            for i in range(n):
                pt = x0.copy()
                step = 0.5 if x0[i] >= 0 else -0.5
                pt[i] += step
                simplex.append(pt)

            # Evaluate initial simplex
            fvals = []
            for pt in simplex:
                if calls >= budget or time.time() > deadline - 5:
                    return
                pt_clipped = np.clip(pt, -5, 5)
                fvals.append(safe_call(pt_clipped))

            evals_used = n + 1
            max_iter = max(1, (max_evals - evals_used) // (n + 1))

            for _ in range(max_iter):
                if calls >= budget or time.time() > deadline - 5:
                    return

                # Sort simplex by function value
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                if calls >= budget or time.time() > deadline - 5:
                    return
                fr = safe_call(xr)
                evals_used += 1

                if fr < fvals[0]:
                    # Expansion
                    xe = centroid + 2 * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    if calls >= budget or time.time() > deadline - 5:
                        return
                    fe = safe_call(xe)
                    evals_used += 1
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                elif fr < fvals[-2]:
                    simplex[-1] = xr
                    fvals[-1] = fr
                else:
                    # Contraction
                    if fr < fvals[-1]:
                        # Outside contraction
                        xc = centroid + 0.5 * (xr - centroid)
                    else:
                        # Inside contraction
                        xc = centroid - 0.5 * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5, 5)
                    if calls >= budget or time.time() > deadline - 5:
                        return
                    fc = safe_call(xc)
                    evals_used += 1
                    if fc <= min(fr, fvals[-1]):
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            if calls >= budget or time.time() > deadline - 5:
                                return
                            fvals[i] = safe_call(simplex[i])
                        evals_used += n

                # Check convergence
                if max(fvals) - min(fvals) < tol:
                    break

        # Phase 1: Deterministic global sampling for early progress
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

        # Evaluate these points deterministically
        for p in points_to_eval:
            if calls >= budget or time.time() > deadline - 30:
                break
            safe_call(p)

        # Fallback
        if best_x is None:
            safe_call(np.zeros(n))

        # Phase 2: Multi-start Nelder-Mead from diverse starting points
        if best_x is not None and calls < budget and time.time() < deadline - 10:
            # Generate diverse starting points around the best point
            np.random.seed(42)
            start_points = [best_x.copy()]

            # Add points from the initial sampling that were relatively good
            for p in points_to_eval:
                if not np.allclose(p, best_x):
                    start_points.append(p.copy())

            # Add random perturbations around the best point
            for _ in range(5):
                perturbation = np.random.randn(n) * 1.0
                pt = best_x + perturbation
                pt = np.clip(pt, -5, 5)
                start_points.append(pt)

            # Try each starting point with Nelder-Mead
            for sp in start_points:
                if calls >= budget or time.time() > deadline - 5:
                    break
                nelder_mead(sp, max_evals=35)

        return best_x, best_f
# EVOLVE-BLOCK-END
