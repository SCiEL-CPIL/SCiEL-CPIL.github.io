"""Direct solver for BBOB f9 (rotated Rosenbrock) using origin-first multi-start with adaptive restarts."""
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Origin-first L-BFGS-B with adaptive diverse restarts and high-precision polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 270.0
        dim = self.dim
        budget = self.budget
        bounds = [(-5.0, 5.0)] * dim

        call_count = [0]
        best_x = [None]
        best_f = [np.inf]

        def counted_func(x):
            call_count[0] += 1
            if call_count[0] > budget:
                return best_f[0]
            x_clipped = np.clip(x, -5.0, 5.0)
            f = func(x_clipped)
            if f < best_f[0]:
                best_f[0] = f
                best_x[0] = x_clipped.copy()
            return f

        def time_ok(margin=5.0):
            return time.time() < deadline - margin

        rng = np.random.default_rng(42)

        # Phase 1: Evaluate origin first (Rosenbrock global min in native coords)
        origin = np.zeros(dim)
        counted_func(origin)
        origin_f = best_f[0]

        # Phase 2: Immediate L-BFGS-B from origin — this should find the valley quickly
        if call_count[0] < budget - 30 and time_ok(15.0):
            try:
                res = minimize(
                    counted_func,
                    origin,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                if res.fun < best_f[0]:
                    best_f[0] = res.fun
                    best_x[0] = np.clip(res.x, -5.0, 5.0)
            except Exception:
                pass

        # Phase 3: Diverse restarts from points that may be near the rotated valley
        # The valley is a 1D curve; diagonal and axis-aligned points are good probes
        rng2 = np.random.default_rng(7)
        restart_candidates = []

        # Diagonal points (varying magnitude)
        for s in [3.0, 2.5, 2.0, 1.5, 1.0, -1.0, -2.0, -3.0]:
            restart_candidates.append(np.full(dim, s))

        # Axis-aligned perturbations around origin
        for i in range(dim):
            for s in [2.0, -2.0, 1.0, -1.0]:
                p = np.zeros(dim)
                p[i] = s
                restart_candidates.append(p)

        # Random points in the box
        for _ in range(15):
            restart_candidates.append(rng2.uniform(-4.0, 4.0, dim))

        # Shuffle for diversity
        perm = rng2.permutation(len(restart_candidates))
        restart_candidates = [restart_candidates[i] for i in perm]

        # Track used points to avoid redundant starts
        used_points = [best_x[0].copy()] if best_x[0] is not None else []
        restarts_done = 0
        max_restarts = 8

        for x0 in restart_candidates:
            if call_count[0] >= budget - 10:
                break
            if not time_ok(10.0):
                break
            if restarts_done >= max_restarts:
                break

            x0_clipped = np.clip(x0, -5.0, 5.0)

            # Check distance to already-used points
            too_close = False
            for up in used_points:
                if np.linalg.norm(x0_clipped - up) < 0.5:
                    too_close = True
                    break
            if too_close:
                continue

            # Evaluate this restart point
            f0 = counted_func(x0_clipped)
            used_points.append(x0_clipped.copy())
            restarts_done += 1

            # If this point is promising, run L-BFGS-B from it
            if f0 < best_f[0] + 50.0 and call_count[0] < budget - 15 and time_ok(10.0):
                try:
                    res = minimize(
                        counted_func,
                        x0_clipped,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 100, 'ftol': 1e-13, 'gtol': 1e-11}
                    )
                    if res.fun < best_f[0]:
                        best_f[0] = res.fun
                        best_x[0] = np.clip(res.x, -5.0, 5.0)
                except Exception:
                    pass

        # Phase 4: Nelder-Mead polish if we're stuck (gradient methods may stall on flat valley)
        if best_x[0] is not None and call_count[0] < budget - 5 and time_ok(5.0):
            current_f = best_f[0]
            # Only try NM if L-BFGS-B didn't get us very close (heuristic: > 1.0 above opt)
            # We don't know f_opt, so use relative check: if improvement in last few calls was small
            try:
                res = minimize(
                    counted_func,
                    best_x[0],
                    method='Nelder-Mead',
                    bounds=bounds,
                    options={'maxiter': 50, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
                if res.fun < best_f[0]:
                    best_f[0] = res.fun
                    best_x[0] = np.clip(res.x, -5.0, 5.0)
            except Exception:
                pass

        # Phase 5: Final high-precision L-BFGS-B polish
        if best_x[0] is not None and call_count[0] < budget - 3 and time_ok(3.0):
            try:
                res = minimize(
                    counted_func,
                    best_x[0],
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 30, 'ftol': 1e-16, 'gtol': 1e-14}
                )
                if res.fun < best_f[0]:
                    best_f[0] = res.fun
                    best_x[0] = np.clip(res.x, -5.0, 5.0)
            except Exception:
                pass

        # Ensure valid result
        if best_x[0] is None:
            best_x[0] = np.zeros(dim)
            if call_count[0] < budget:
                best_f[0] = counted_func(best_x[0])

        return best_x[0], best_f[0]
