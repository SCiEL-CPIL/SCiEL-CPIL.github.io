"""Optimizer for BBOB f8 (Rosenbrock) on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s

        # Wrap func with clipping and a call counter
        call_count = [0]
        def safe_func(x):
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            call_count[0] += 1
            return float(func(x))

        # --- Phase 1: Global scan (100 evals) ---
        # Grid points: 3^5 = 243 is too many, use 2^5=32 + structured points
        # Use a mix: corners of sub-boxes + center + random
        scan_points = []
        # 3-level grid on a subset of dimensions, or structured points
        # Let's use: for each dim, try -4, 0, 4 -> 3^5 = 243, too many
        # Instead: 2^5 = 32 points at {-3, 3}^5 plus center, plus some random
        for vals in np.ndindex(2, 2, 2, 2, 2):
            pt = np.array([3.0 if v else -3.0 for v in vals], dtype=float)
            scan_points.append(pt)
        scan_points.append(np.zeros(5))
        # Add some structured points
        for v in [-4, -2, 0, 2, 4]:
            scan_points.append(np.full(5, v))
        # Remove duplicates
        seen = set()
        unique_points = []
        for p in scan_points:
            key = tuple(np.round(p, 6))
            if key not in seen:
                seen.add(key)
                unique_points.append(p)

        # Evaluate scan points (up to 100 calls)
        best_x = None
        best_f = np.inf
        for p in unique_points:
            if call_count[0] >= 100:
                break
            if time.time() > deadline:
                break
            f = safe_func(p)
            if f < best_f:
                best_f = f
                best_x = p.copy()

        # --- Phase 2: Gradient descent from best_x ---
        # Use numerical gradient (central differences)
        def gradient(x):
            g = np.zeros(self.dim)
            h = 1e-5
            for i in range(self.dim):
                xp = x.copy(); xp[i] += h
                xm = x.copy(); xm[i] -= h
                g[i] = (safe_func(xp) - safe_func(xm)) / (2 * h)
            return g

        x = best_x.copy()
        fx = best_f
        step = 1.0
        # Adaptive gradient descent
        for it in range(300):
            if call_count[0] >= self.budget - 100:  # leave budget for polish
                break
            if time.time() > deadline:
                break
            g = gradient(x)
            gn = np.linalg.norm(g)
            if gn < 1e-10:
                break
            # Adaptive step: try to decrease
            step = 0.5 / (gn + 1e-8)
            x_new = np.clip(x - step * g, -5.0, 5.0)
            fx_new = safe_func(x_new)
            if fx_new < fx:
                x = x_new
                fx = fx_new
                step *= 1.5
            else:
                step *= 0.5
                if step < 1e-12:
                    break

        # --- Phase 3: Nelder-Mead polish ---
        def nelder_mead(x0, max_iters):
            n = self.dim
            # Initialize simplex
            simplex = [x0.copy()]
            fvals = [safe_func(x0)]
            for i in range(n):
                xi = x0.copy()
                xi[i] += 0.1 if x0[i] < 4.9 else -0.1
                simplex.append(xi)
                fvals.append(safe_func(xi))

            best_idx = int(np.argmin(fvals))
            for it in range(max_iters):
                if call_count[0] >= self.budget - 10:
                    break
                if time.time() > deadline:
                    break
                # Sort simplex
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]
                best_idx = 0
                worst_idx = n

                centroid = np.mean(simplex[:n], axis=0)
                # Reflection
                xr = centroid + 1.0 * (centroid - simplex[worst_idx])
                fr = safe_func(xr)
                if fvals[0] <= fr < fvals[worst_idx - 1]:
                    simplex[worst_idx] = xr
                    fvals[worst_idx] = fr
                elif fr < fvals[0]:
                    # Expansion
                    xe = centroid + 2.0 * (centroid - simplex[worst_idx])
                    fe = safe_func(xe)
                    if fe < fr:
                        simplex[worst_idx] = xe
                        fvals[worst_idx] = fe
                    else:
                        simplex[worst_idx] = xr
                        fvals[worst_idx] = fr
                else:
                    # Contraction
                    if fr < fvals[worst_idx]:
                        # Outside contraction
                        xc = centroid + 0.5 * (simplex[worst_idx] - centroid)
                        fc = safe_func(xc)
                        if fc <= fr:
                            simplex[worst_idx] = xc
                            fvals[worst_idx] = fc
                        else:
                            # Shrink
                            for i in range(1, n + 1):
                                simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                                fvals[i] = safe_func(simplex[i])
                            break
                    else:
                        # Inside contraction
                        xc = centroid - 0.5 * (simplex[worst_idx] - centroid)
                        fc = safe_func(xc)
                        if fc < fvals[worst_idx]:
                            simplex[worst_idx] = xc
                            fvals[worst_idx] = fc
                        else:
                            # Shrink
                            for i in range(1, n + 1):
                                simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                                fvals[i] = safe_func(simplex[i])
                            break

                best_idx = int(np.argmin(fvals))

            return simplex[best_idx], fvals[best_idx]

        x_polish, f_polish = nelder_mead(x, 200)
        if f_polish < fx:
            x = x_polish
            fx = f_polish

        # Final check: make sure we have a valid result
        if best_x is None:
            best_x = np.zeros(self.dim)

        return x, fx
# EVOLVE-BLOCK-END
