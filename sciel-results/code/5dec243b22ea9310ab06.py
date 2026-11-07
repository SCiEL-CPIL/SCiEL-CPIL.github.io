import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0  # Safety margin before 300s limit

        rng = np.random.default_rng(12345)
        call_count = 0
        best_x = None
        best_f = np.inf
        self._best_x = None
        self._best_f = np.inf
        dim = self.dim

        def safe_func(x):
            nonlocal call_count, best_x, best_f
            if call_count >= self.budget or time.time() > deadline - 2:
                raise ValueError("Budget or time exceeded")
            call_count += 1
            x_arr = np.asarray(x, dtype=float)
            val = float(func(x_arr))
            if np.isfinite(val) and val < best_f:
                best_f = val
                best_x = np.array(x_arr, dtype=float)
                self._best_x = best_x.copy()
                self._best_f = best_f
            return val

        # --- Phase 1: Structured coarse exploration (grid + random + axes) ---
        # Rosenbrock in 5D has a long narrow valley; we need to get into it.
        # Use a sparse 3^5 grid, random points, and axis-aligned points.
        grid_vals = np.linspace(-4, 4, 3)
        grid_pts = np.array(np.meshgrid(*([grid_vals] * dim))).T.reshape(-1, dim)

        n_random = 50
        random_pts = rng.uniform(-4, 4, (n_random, dim))

        axis_pts = []
        for i in range(dim):
            for v in [-3, -2, -1, 0, 1, 2, 3]:
                p = np.zeros(dim)
                p[i] = v
                axis_pts.append(p)
        axis_pts = np.array(axis_pts)

        all_pts = np.vstack([grid_pts, random_pts, axis_pts])

        for p in all_pts:
            if call_count >= self.budget or time.time() > deadline - 10:
                break
            try:
                safe_func(p)
            except ValueError:
                break
            except Exception:
                continue

        if best_x is None:
            return np.zeros(dim), 0.0

        # --- Phase 2: Multi-start Nelder-Mead for basin refinement ---
        # Nelder-Mead is derivative-free and handles the non-quadratic landscape
        # of Rosenbrock's valley better than pure gradient methods from a single start.
        # We run several short NM runs from diverse starting points near the best.

        def nelder_mead_local(x0, max_calls=60):
            """Local Nelder-Mead using safe_func, returns (best_x, best_f)."""
            nonlocal call_count, best_x, best_f

            simplex = [x0.copy()]
            base_step = 0.5
            for i in range(dim):
                xi = x0.copy()
                xi[i] += base_step
                simplex.append(xi)

            fvals = []
            for v in simplex:
                try:
                    fv = safe_func(v)
                except ValueError:
                    return x0, best_f
                fvals.append(fv)

            alpha, gamma, rho, sigma_nm = 1.0, 2.0, 0.5, 0.5
            max_iter = max(5, max_calls // (dim + 2))

            for it in range(max_iter):
                if time.time() > deadline - 5 or call_count >= self.budget - 20:
                    break

                idx = np.argsort(fvals)
                simplex = [simplex[i] for i in idx]
                fvals = [fvals[i] for i in idx]

                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                try:
                    fr = safe_func(xr)
                except ValueError:
                    break

                if fr < fvals[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    try:
                        fe = safe_func(xe)
                    except ValueError:
                        break
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
                    xc = centroid + rho * (simplex[-1] - centroid)
                    try:
                        fc = safe_func(xc)
                    except ValueError:
                        break
                    if fc < fvals[-1]:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            try:
                                fvals[i] = safe_func(simplex[i])
                            except ValueError:
                                break
                        break

            best_idx = np.argmin(fvals)
            return simplex[best_idx], fvals[best_idx]

        # Generate diverse starting points for NM
        nm_starts = [best_x.copy()]
        for scale in [0.3, 0.6, 1.0, 1.5]:
            pert = rng.normal(0, scale, dim)
            nm_starts.append(np.clip(best_x + pert, -5, 5))
        # Add a few random points in the basin region
        for _ in range(3):
            nm_starts.append(rng.uniform(-3, 3, dim))

        for x0 in nm_starts:
            if call_count >= self.budget - 30 or time.time() > deadline - 10:
                break
            try:
                x_nm, f_nm = nelder_mead_local(x0, max_calls=50)
                if f_nm < best_f:
                    best_f = f_nm
                    best_x = x_nm.copy()
                    self._best_x = best_x.copy()
                    self._best_f = best_f
            except Exception:
                continue

        # --- Phase 3: L-BFGS-B polish from the best point found ---
        remaining = self.budget - call_count
        if remaining > 10 and time.time() < deadline - 10:
            try:
                res = minimize(
                    safe_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 300, 'maxfun': remaining, 'maxls': 50,
                             'gtol': 1e-10, 'ftol': 1e-14}
                )
                if np.isfinite(res.fun) and res.fun < best_f:
                    best_f = float(res.fun)
                    best_x = np.array(res.x, dtype=float)
                    self._best_x = best_x.copy()
                    self._best_f = best_f
            except Exception:
                pass

        # --- Phase 4: Final Nelder-Mead polish for any remaining budget ---
        remaining = self.budget - call_count
        if remaining > 20 and time.time() < deadline - 5:
            try:
                x_nm2, f_nm2 = nelder_mead_local(best_x, max_calls=min(40, remaining - 5))
                if f_nm2 < best_f:
                    best_f = f_nm2
                    best_x = x_nm2.copy()
                    self._best_x = best_x.copy()
                    self._best_f = best_f
            except Exception:
                pass

        return best_x, best_f
