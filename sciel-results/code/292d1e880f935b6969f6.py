"""Optimizer for BBOB f8 (Rosenbrock) on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin before 300s

        dim = self.dim
        budget = self.budget

        # Wrap func with clipping and a call counter
        call_count = [0]

        def safe_func(x):
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            call_count[0] += 1
            return float(func(x))

        # --- Phase 1: Multi-start with diverse initial points ---
        # Generate a set of diverse starting points
        start_points = []

        # Center
        start_points.append(np.zeros(dim))

        # Structured grid points at ±2, ±3, ±4
        for v in [-4, -3, -2, 2, 3, 4]:
            start_points.append(np.full(dim, v))

        # Random points
        rng = np.random.default_rng(42)
        for _ in range(20):
            pt = rng.uniform(-4.5, 4.5, size=dim)
            start_points.append(pt)

        # Add some points along axes
        for i in range(dim):
            for v in [-3, -1, 1, 3]:
                pt = np.zeros(dim)
                pt[i] = v
                start_points.append(pt)

        # Remove duplicates
        seen = set()
        unique_points = []
        for p in start_points:
            key = tuple(np.round(p, 6))
            if key not in seen:
                seen.add(key)
                unique_points.append(p)

        # Evaluate all start points
        best_x = None
        best_f = np.inf
        for p in unique_points:
            if call_count[0] >= 80:
                break
            if time.time() > deadline:
                break
            f = safe_func(p)
            if f < best_f:
                best_f = f
                best_x = p.copy()

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = safe_func(best_x)

        # --- Phase 2: L-BFGS-B from best point ---
        try:
            res = minimize(
                safe_func,
                best_x,
                method='L-BFGS-B',
                bounds=[(-5.0, 5.0)] * dim,
                options={'maxiter': 200, 'maxfun': 200}
            )
            if res.success or res.fun < best_f:
                best_x = res.x
                best_f = res.fun
        except Exception:
            pass

        # --- Phase 3: Powell's method for polishing ---
        try:
            res2 = minimize(
                safe_func,
                best_x,
                method='Powell',
                bounds=[(-5.0, 5.0)] * dim,
                options={'maxiter': 100, 'maxfev': 150}
            )
            if res2.fun < best_f:
                best_x = res2.x
                best_f = res2.fun
        except Exception:
            pass

        # --- Phase 4: Nelder-Mead for final polish ---
        def nelder_mead(x0, max_iters=100):
            n = dim
            # Initialize simplex
            simplex = [x0.copy()]
            fvals = [safe_func(x0)]
            for i in range(n):
                xi = x0.copy()
                xi[i] += 0.1 if x0[i] < 4.9 else -0.1
                simplex.append(xi)
                fvals.append(safe_func(xi))

            for it in range(max_iters):
                if call_count[0] >= budget - 10:
                    break
                if time.time() > deadline:
                    break
                # Sort simplex
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

                centroid = np.mean(simplex[:n], axis=0)
                # Reflection
                xr = centroid + 1.0 * (centroid - simplex[n])
                fr = safe_func(xr)
                if fvals[0] <= fr < fvals[n - 1]:
                    simplex[n] = xr
                    fvals[n] = fr
                elif fr < fvals[0]:
                    # Expansion
                    xe = centroid + 2.0 * (centroid - simplex[n])
                    fe = safe_func(xe)
                    if fe < fr:
                        simplex[n] = xe
                        fvals[n] = fe
                    else:
                        simplex[n] = xr
                        fvals[n] = fr
                else:
                    # Contraction
                    xc = centroid + 0.5 * (simplex[n] - centroid)
                    fc = safe_func(xc)
                    if fc <= fr:
                        simplex[n] = xc
                        fvals[n] = fc
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            fvals[i] = safe_func(simplex[i])
                        break

            best_idx = int(np.argmin(fvals))
            return simplex[best_idx], fvals[best_idx]

        x_nm, f_nm = nelder_mead(best_x, 100)
        if f_nm < best_f:
            best_x = x_nm
            best_f = f_nm

        # --- Phase 5: Try a few more L-BFGS-B restarts from nearby points ---
        if call_count[0] < budget - 50 and time.time() < deadline:
            for _ in range(3):
                if call_count[0] >= budget - 50:
                    break
                if time.time() > deadline:
                    break
                # Perturb the current best
                perturbation = rng.normal(0, 0.1, size=dim)
                x_try = np.clip(best_x + perturbation, -5.0, 5.0)
                try:
                    res3 = minimize(
                        safe_func,
                        x_try,
                        method='L-BFGS-B',
                        bounds=[(-5.0, 5.0)] * dim,
                        options={'maxiter': 100, 'maxfun': 100}
                    )
                    if res3.fun < best_f:
                        best_x = res3.x
                        best_f = res3.fun
                except Exception:
                    pass

        return best_x, best_f
# EVOLVE-BLOCK-END
