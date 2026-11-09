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
        # Büche-Rastrigin has many local minima; a coarse grid helps find a decent basin.
        # Use a small number of candidate points spread across the domain.
        n_scan = 12
        # Generate deterministic scan points: corners, center, and a few structured points
        scan_points = []
        # Corners
        for i in range(2 ** dim):
            pt = np.array([[-5.0, 5.0][(i >> d) & 1] for d in range(dim)])
            scan_points.append(pt)
        # Center
        scan_points.append(np.zeros(dim))
        # Mid-points along each axis
        for d in range(dim):
            for val in [-3.0, 3.0]:
                pt = np.zeros(dim)
                pt[d] = val
                scan_points.append(pt)
        # Trim to n_scan
        scan_points = scan_points[:n_scan]

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

        # Phase 2: Multi-scale coordinate descent
        # Geometrically shrinking step sizes, cycling through dimensions.
        # Start with large steps to traverse the landscape, shrink to fine-tune.
        base_steps = [8.0, 4.0, 2.0, 1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]

        x = best_x.copy()
        fx = best_f

        for step in base_steps:
            # Within each step size, do coordinate descent passes
            # Try positive and negative steps along each dimension
            improved = True
            while improved and calls_used < budget:
                if time.time() > deadline - 5.0:
                    break
                improved = False
                for d in range(dim):
                    if calls_used >= budget:
                        break
                    if time.time() > deadline - 5.0:
                        break
                    for direction in [1.0, -1.0]:
                        if calls_used >= budget:
                            break
                        if time.time() > deadline - 5.0:
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

        # Phase 3: If budget remains, do a finer local search with smaller steps
        if calls_used < budget:
            fine_steps = [0.0005, 0.0002, 0.0001, 0.00005, 0.00002, 0.00001]
            for step in fine_steps:
                if calls_used >= budget:
                    break
                if time.time() > deadline - 5.0:
                    break
                for d in range(dim):
                    if calls_used >= budget:
                        break
                    if time.time() > deadline - 5.0:
                        break
                    for direction in [1.0, -1.0]:
                        if calls_used >= budget:
                            break
                        if time.time() > deadline - 5.0:
                            break
                        x_new = x.copy()
                        x_new[d] += direction * step
                        x_new = np.clip(x_new, -5.0, 5.0)
                        f_new = func(x_new)
                        calls_used += 1
                        if f_new < fx:
                            fx = f_new
                            x = x_new

        # Phase 4: Use remaining budget for a simple random perturbation refinement
        if calls_used < budget:
            rng = np.random.default_rng(42)
            sigma = 0.01
            while calls_used < budget:
                if time.time() > deadline - 2.0:
                    break
                x_new = x + sigma * rng.standard_normal(dim)
                x_new = np.clip(x_new, -5.0, 5.0)
                f_new = func(x_new)
                calls_used += 1
                if f_new < fx:
                    fx = f_new
                    x = x_new
                    sigma *= 1.2
                else:
                    sigma *= 0.85
                    if sigma < 1e-8:
                        break

        return x, fx
