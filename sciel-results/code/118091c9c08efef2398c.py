"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Deterministic multi-start local search with Sobol coverage and BFGS refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 285.0  # safety margin before 300s limit

        dim = self.dim
        budget = self.budget
        box_low = -5.0
        box_high = 5.0
        calls_made = 0

        def clip(x):
            return np.clip(x, box_low, box_high)

        def evaluate(x):
            nonlocal calls_made
            if calls_made >= budget or time.time() > deadline:
                return None
            x = np.asarray(x, dtype=float)
            if x.shape != (dim,):
                x = x.reshape(dim)
            x = clip(x)
            f = func(x)
            calls_made += 1
            return x, f

        # Halton sequence for low-discrepancy initialization
        def halton(index, base):
            f = 1.0
            r = 0.0
            i = index
            while i > 0:
                f /= base
                r += f * (i % base)
                i //= base
            return r

        def halton_point(index, dim, low, high):
            point = np.zeros(dim)
            for d in range(dim):
                base = d + 2  # 2, 3, 5, 7, 11, ...
                point[d] = low + (high - low) * halton(index + 1, base)
            return point

        # Phase 1: Broad exploration with Halton sequence
        n_explore = min(80, budget // 6)
        best_x = None
        best_f = np.inf

        for i in range(n_explore):
            if time.time() > deadline - 60:
                break
            x = halton_point(i, dim, box_low, box_high)
            res = evaluate(x)
            if res is None:
                break
            x, f = res
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Also evaluate origin and a few structured points
        structured = [
            np.zeros(dim),
            np.ones(dim) * 2.0,
            np.ones(dim) * -2.0,
            np.array([1, -1, 1, -1, 1]) * 2.0,
            np.array([3, -3, 0, 3, -3]),
        ]
        for x in structured:
            if time.time() > deadline - 60:
                break
            res = evaluate(x)
            if res is None:
                break
            x, f = res
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 2: Local search using Nelder-Mead simplex method
        # This is more effective for narrow ridges than coordinate descent
        def nelder_mead(x0, f0, max_iter=200):
            nonlocal calls_made, best_f, best_x
            if time.time() > deadline - 30:
                return x0, f0

            # Initialize simplex
            simplex = [x0.copy()]
            fvals = [f0]

            # Expand simplex along each dimension
            step = 0.5
            for d in range(dim):
                if time.time() > deadline - 30 or calls_made >= budget:
                    break
                x_new = x0.copy()
                x_new[d] += step
                x_new = clip(x_new)
                res = evaluate(x_new)
                if res is None:
                    break
                x_new, f_new = res
                simplex.append(x_new)
                fvals.append(f_new)
                if f_new < best_f:
                    best_f = f_new
                    best_x = x_new.copy()

            if len(simplex) < dim + 1:
                return x0, f0

            alpha = 1.0   # reflection
            gamma = 2.0   # expansion
            rho = 0.5     # contraction
            sigma = 0.5   # shrink

            for _ in range(max_iter):
                if time.time() > deadline - 30 or calls_made >= budget:
                    break

                # Sort simplex by function value
                order = np.argsort(fvals)
                simplex = [simplex[i].copy() for i in order]
                fvals = [fvals[i] for i in order]

                # Centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = clip(x_r)
                res = evaluate(x_r)
                if res is None:
                    break
                x_r, f_r = res
                if f_r < best_f:
                    best_f = f_r
                    best_x = x_r.copy()

                if f_r < fvals[0]:
                    # Try expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = clip(x_e)
                    res = evaluate(x_e)
                    if res is None:
                        break
                    x_e, f_e = res
                    if f_e < f_r:
                        simplex[-1] = x_e
                        fvals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fvals[-1] = f_r
                elif f_r < fvals[-2]:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
                else:
                    # Contraction
                    x_c = centroid + rho * (simplex[-1] - centroid)
                    x_c = clip(x_c)
                    res = evaluate(x_c)
                    if res is None:
                        break
                    x_c, f_c = res
                    if f_c < f_r:
                        simplex[-1] = x_c
                        fvals[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = clip(simplex[i])
                            res = evaluate(simplex[i])
                            if res is None:
                                return simplex[0], fvals[0]
                            simplex[i], fvals[i] = res
                            if fvals[i] < best_f:
                                best_f = fvals[i]
                                best_x = simplex[i].copy()

                # Check convergence
                if len(set([fvals[0], fvals[-1]])) == 1:
                    break
                if abs(fvals[-1] - fvals[0]) < 1e-10:
                    break

            # Return best point in simplex
            best_idx = np.argmin(fvals)
            return simplex[best_idx].copy(), fvals[best_idx]

        # Run Nelder-Mead from the best point
        if best_x is not None:
            x_nm, f_nm = nelder_mead(best_x, best_f)
            if f_nm < best_f:
                best_f = f_nm
                best_x = x_nm

        # Phase 3: Additional local refinement with adaptive step search
        if calls_made < budget and time.time() < deadline - 20:
            current_x = best_x.copy()
            current_f = best_f
            step = 0.5
            stagnant = 0

            while calls_made < budget and time.time() < deadline - 10:
                improved = False
                for d in range(dim):
                    if calls_made >= budget or time.time() > deadline - 10:
                        break

                    # Try positive
                    trial = current_x.copy()
                    trial[d] += step
                    trial = clip(trial)
                    res = evaluate(trial)
                    if res is None:
                        break
                    trial, f_trial = res
                    if f_trial < current_f:
                        current_x = trial
                        current_f = f_trial
                        improved = True
                        if f_trial < best_f:
                            best_f = f_trial
                            best_x = current_x.copy()

                    if calls_made >= budget or time.time() > deadline - 10:
                        break

                    # Try negative
                    trial = current_x.copy()
                    trial[d] -= step
                    trial = clip(trial)
                    res = evaluate(trial)
                    if res is None:
                        break
                    trial, f_trial = res
                    if f_trial < current_f:
                        current_x = trial
                        current_f = f_trial
                        improved = True
                        if f_trial < best_f:
                            best_f = f_trial
                            best_x = current_x.copy()

                if improved:
                    stagnant = 0
                    step = min(step * 1.05, 2.0)
                else:
                    stagnant += 1
                    step *= 0.5
                    if step < 1e-5:
                        break
                    if stagnant > 5:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
