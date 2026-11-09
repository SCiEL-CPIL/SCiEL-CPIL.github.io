"""Rewritten optimizer: deterministic multi-scale coordinate descent with time guard."""
import time
import numpy as np


class Optimizer:
    """Deterministic multi-scale coordinate descent for Büche-Rastrigin (f4)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 280.0  # safety margin before 300s kill

        dim = self.dim
        budget = self.budget

        # Phase 1: Coarse grid scan to find a good starting region
        # Use structured scan points with good coverage
        scan_points = []
        # Corners
        for i in range(2 ** dim):
            pt = np.array([[-5.0, 5.0][(i >> d) & 1] for d in range(dim)])
            scan_points.append(pt)
        # Center
        scan_points.append(np.zeros(dim))
        # Mid-points along each axis at multiple radii
        for d in range(dim):
            for val in [-4.0, -3.0, -2.0, 2.0, 3.0, 4.0]:
                pt = np.zeros(dim)
                pt[d] = val
                scan_points.append(pt)
        # Diagonal points
        for r in [2.0, 3.0, 4.0]:
            pt = np.full(dim, r)
            scan_points.append(pt)
            pt = np.full(dim, -r)
            scan_points.append(pt)
        # Mixed-sign points
        for r in [2.0, 3.0]:
            for d in range(dim):
                pt = np.full(dim, r)
                pt[d] = -r
                scan_points.append(pt)

        best_x = None
        best_f = np.inf
        calls_used = 0

        for sp in scan_points:
            if calls_used >= budget:
                break
            if time.time() > deadline - 5.0:
                break
            f_val = func(sp)
            calls_used += 1
            if f_val < best_f:
                best_f = f_val
                best_x = sp.copy()

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls_used += 1

        # Phase 2: Adaptive coordinate descent with momentum
        # Use geometrically shrinking base steps, but adapt locally
        base_steps = [6.0, 3.0, 1.5, 0.75, 0.375, 0.1875, 0.09, 0.045, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]

        x = best_x.copy()
        fx = best_f

        for step in base_steps:
            if calls_used >= budget or time.time() > deadline - 5.0:
                break
            # Within each step size, do coordinate descent passes with adaptive refinement
            improved = True
            max_passes = 3
            for _pass in range(max_passes):
                if not improved:
                    break
                if calls_used >= budget or time.time() > deadline - 5.0:
                    break
                improved = False
                for d in range(dim):
                    if calls_used >= budget or time.time() > deadline - 5.0:
                        break
                    # Try both directions with the current step
                    for direction in [1.0, -1.0]:
                        if calls_used >= budget or time.time() > deadline - 5.0:
                            break
                        x_new = x.copy()
                        x_new[d] += direction * step
                        x_new = np.clip(x_new, -5.0, 5.0)
                        f_new = func(x_new)
                        calls_used += 1
                        if f_new < fx:
                            fx = f_new
                            x = x_new
                            improved = True
                            # If successful, try to continue in same direction (momentum)
                            momentum_steps = 3
                            for _m in range(momentum_steps):
                                if calls_used >= budget or time.time() > deadline - 5.0:
                                    break
                                x_mom = x.copy()
                                x_mom[d] += direction * step * 0.5
                                x_mom = np.clip(x_mom, -5.0, 5.0)
                                f_mom = func(x_mom)
                                calls_used += 1
                                if f_mom < fx:
                                    fx = f_mom
                                    x = x_mom
                                else:
                                    break
                    # If no improvement in either direction, try a half-step
                    if not improved:
                        for direction in [1.0, -1.0]:
                            if calls_used >= budget or time.time() > deadline - 5.0:
                                break
                            x_half = x.copy()
                            x_half[d] += direction * step * 0.5
                            x_half = np.clip(x_half, -5.0, 5.0)
                            f_half = func(x_half)
                            calls_used += 1
                            if f_half < fx:
                                fx = f_half
                                x = x_half
                                improved = True

        # Phase 3: Local restarts to escape local minima
        # Use remaining budget for targeted restarts near current best
        if calls_used < budget and time.time() < deadline - 5.0:
            rng = np.random.default_rng(42)
            remaining = budget - calls_used
            n_restarts = min(remaining // 20, 8)  # up to 8 restarts, each using ~20 calls

            for _r in range(n_restarts):
                if calls_used >= budget or time.time() > deadline - 5.0:
                    break
                # Start near current best with some perturbation
                sigma_restart = 0.5
                x_r = x + sigma_restart * rng.standard_normal(dim)
                x_r = np.clip(x_r, -5.0, 5.0)
                f_r = func(x_r)
                calls_used += 1

                if f_r < fx:
                    fx = f_r
                    x = x_r

                # Do a few rounds of coordinate descent from this new point
                local_steps = [0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
                x_local = x.copy()
                fx_local = fx
                for ls in local_steps:
                    if calls_used >= budget or time.time() > deadline - 5.0:
                        break
                    for d in range(dim):
                        if calls_used >= budget or time.time() > deadline - 5.0:
                            break
                        for direction in [1.0, -1.0]:
                            if calls_used >= budget or time.time() > deadline - 5.0:
                                break
                            x_new = x_local.copy()
                            x_new[d] += direction * ls
                            x_new = np.clip(x_new, -5.0, 5.0)
                            f_new = func(x_new)
                            calls_used += 1
                            if f_new < fx_local:
                                fx_local = f_new
                                x_local = x_new

                if fx_local < fx:
                    fx = fx_local
                    x = x_local

        # Phase 4: Fine polish with adaptive step
        if calls_used < budget and time.time() < deadline - 5.0:
            fine_steps = [0.0005, 0.0002, 0.0001, 0.00005, 0.00002, 0.00001, 0.000005]
            for step in fine_steps:
                if calls_used >= budget or time.time() > deadline - 5.0:
                    break
                for d in range(dim):
                    if calls_used >= budget or time.time() > deadline - 5.0:
                        break
                    for direction in [1.0, -1.0]:
                        if calls_used >= budget or time.time() > deadline - 5.0:
                            break
                        x_new = x.copy()
                        x_new[d] += direction * step
                        x_new = np.clip(x_new, -5.0, 5.0)
                        f_new = func(x_new)
                        calls_used += 1
                        if f_new < fx:
                            fx = f_new
                            x = x_new

        return x, fx
