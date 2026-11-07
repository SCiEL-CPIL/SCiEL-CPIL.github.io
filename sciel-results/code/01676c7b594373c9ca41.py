"""Optimizer for BBOB-style f11 (Discus) using Nelder-Mead simplex with restarts."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 330s kill

        d = self.dim
        budget = self.budget
        calls = [0]

        def wrapped(x):
            calls[0] += 1
            return float(func(x))

        def alive():
            return calls[0] < budget and time.time() < deadline

        # ---- Phase 0: cheap random starts to get a good initial point ----
        rng = np.random.default_rng(42)
        candidates = [np.zeros(d)]
        for _ in range(8):
            candidates.append(rng.uniform(-4, 4, d))
        best_x = None
        best_f = np.inf
        for c in candidates:
            if not alive():
                break
            c = np.clip(c, -5, 5)
            fc = wrapped(c)
            if fc < best_f:
                best_f = fc
                best_x = c.copy()

        x = best_x.copy()
        fx = best_f

        # ---- Nelder-Mead simplex method (clean implementation) ----
        def nelder_mead(x0, fx0, max_iters):
            """Run Nelder-Mead from x0, returning best point and value found."""
            nonlocal best_f, best_x
            n = d
            npts = n + 1  # number of simplex vertices

            # Initialize simplex: x0 plus n perturbations
            simplex = np.zeros((npts, n))
            fvals = np.zeros(npts)
            simplex[0] = x0
            fvals[0] = fx0
            for i in range(n):
                simplex[i + 1] = x0.copy()
                delta = 0.5 if abs(x0[i]) < 4.5 else 0.25
                simplex[i + 1, i] += delta
                simplex[i + 1] = np.clip(simplex[i + 1], -5, 5)
                fvals[i + 1] = wrapped(simplex[i + 1])

            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5

            for iteration in range(max_iters):
                if not alive():
                    break

                # Sort vertices by function value (ascending)
                idx = np.argsort(fvals)
                simplex = simplex[idx]
                fvals = fvals[idx]

                # Update global best
                if fvals[0] < best_f:
                    best_f = fvals[0]
                    best_x = simplex[0].copy()

                # Convergence check
                if abs(fvals[0] - fvals[-1]) < 1e-12 * max(1.0, abs(fvals[0])):
                    break

                # Centroid of all points except the worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fr = wrapped(xr)

                if fr < fvals[0]:
                    # Try expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fe = wrapped(xe)
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
                    xc = np.clip(xc, -5, 5)
                    fc = wrapped(xc)
                    if fc <= fr:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink all points except the best toward the best
                        for i in range(1, npts):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fvals[i] = wrapped(simplex[i])

            # Final update
            idx = np.argsort(fvals)
            if fvals[idx[0]] < best_f:
                best_f = fvals[idx[0]]
                best_x = simplex[idx[0]].copy()

            return simplex[idx[0]].copy(), fvals[idx[0]]

        # Run Nelder-Mead with multiple restarts
        # First run from the best initial point
        x, fx = nelder_mead(x, fx, 200)

        # If we still have budget, do a few more runs from perturbed points
        while alive() and calls[0] < budget - 20:
            # Create a perturbation around the current best
            pert = best_x + rng.normal(0, 0.3, d)
            pert = np.clip(pert, -5, 5)
            fp = wrapped(pert)
            if fp < best_f:
                best_f = fp
                best_x = pert.copy()
                x, fx = nelder_mead(x, fx, 100)
            else:
                # Try from a random point if we're stuck
                if fx > best_f + 1e-8:
                    x2 = best_x + rng.normal(0, 1.0, d)
                    x2 = np.clip(x2, -5, 5)
                    fx2 = wrapped(x2)
                    if fx2 < best_f:
                        best_f = fx2
                        best_x = x2.copy()
                        x, fx = nelder_mead(x2, fx2, 100)
                    else:
                        # No improvement; break to save budget
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
