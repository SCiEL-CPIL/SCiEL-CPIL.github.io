"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Combined multi-scale coordinate descent with structured scan and momentum."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 330s limit

        best_x = None
        best_fx = float('inf')
        calls_made = 0

        def update_best(x, fx):
            nonlocal best_x, best_fx
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

        # Phase 1: Structured Initial Scan
        # Büche-Rastrigin has a global minimum that is shifted and rotated.
        # We need to find a basin of attraction. A coarse grid scan is effective.
        scan_points = []

        # 1. Corners of the box [-5, 5]^dim
        for i in range(2 ** self.dim):
            pt = np.array([[-5.0, 5.0][(i >> d) & 1] for d in range(self.dim)])
            scan_points.append(pt)

        # 2. Center
        scan_points.append(np.zeros(self.dim))

        # 3. Face centers (one axis non-zero, others 0)
        for d in range(self.dim):
            for val in [-4.0, -2.0, 2.0, 4.0]:
                pt = np.zeros(self.dim)
                pt[d] = val
                scan_points.append(pt)

        # 4. Some intermediate points to catch rotated minima
        for i in range(self.dim):
            for j in range(self.dim):
                if i == j:
                    continue
                for v1 in [-3.0, 3.0]:
                    for v2 in [-3.0, 3.0]:
                        pt = np.zeros(self.dim)
                        pt[i] = v1
                        pt[j] = v2
                        scan_points.append(pt)

        # Limit scan to ensure we have budget left for optimization
        max_scan = 40
        scan_points = scan_points[:max_scan]

        for sp in scan_points:
            if calls_made >= self.budget or time.time() >= deadline:
                break
            f_val = func(sp)
            calls_made += 1
            update_best(sp, f_val)

        if best_x is None:
            best_x = np.zeros(self.dim)
            best_fx = func(best_x)
            calls_made += 1

        current_x = best_x.copy()
        current_fx = best_fx

        # Phase 2: Multi-Scale Coordinate Descent with Momentum
        # We shrink the step size geometrically. This allows us to "hop" over
        # local minima barriers with large steps and then refine with small steps.
        base_steps = [5.0, 2.5, 1.25, 0.625, 0.3125, 0.156, 0.078, 0.039, 0.02, 0.01, 0.005, 0.002, 0.001]

        for step in base_steps:
            if time.time() >= deadline or calls_made >= self.budget:
                break

            # For each step size, we iterate until no improvement is found in a full pass
            # or we run out of budget/time.
            max_iter_per_step = 100

            for _iter in range(max_iter_per_step):
                if time.time() >= deadline or calls_made >= self.budget:
                    break

                improved = False
                for d in range(self.dim):
                    if time.time() >= deadline or calls_made >= self.budget:
                        break

                    # Try positive direction
                    x_pos = current_x.copy()
                    x_pos[d] += step
                    x_pos = np.clip(x_pos, -5.0, 5.0)
                    f_pos = func(x_pos)
                    calls_made += 1

                    if f_pos < current_fx:
                        current_x = x_pos
                        current_fx = f_pos
                        improved = True
                        update_best(current_x, current_fx)
                        # Momentum: try to continue in the same direction
                        # We do this by re-checking the same dimension immediately
                        # This is a simple form of momentum that helps traverse valleys
                        for _mom in range(5):  # Max 5 momentum steps
                            if time.time() >= deadline or calls_made >= self.budget:
                                break
                            x_mom = current_x.copy()
                            x_mom[d] += step
                            x_mom = np.clip(x_mom, -5.0, 5.0)
                            f_mom = func(x_mom)
                            calls_made += 1
                            if f_mom < current_fx:
                                current_x = x_mom
                                current_fx = f_mom
                                update_best(current_x, current_fx)
                            else:
                                break

                    # Try negative direction from original current_x
                    x_neg = current_x.copy()
                    x_neg[d] -= step
                    x_neg = np.clip(x_neg, -5.0, 5.0)
                    f_neg = func(x_neg)
                    calls_made += 1

                    if f_neg < current_fx:
                        current_x = x_neg
                        current_fx = f_neg
                        improved = True
                        update_best(current_x, current_fx)
                        # Momentum: try to continue in the same direction
                        for _mom in range(5):
                            if time.time() >= deadline or calls_made >= self.budget:
                                break
                            x_mom = current_x.copy()
                            x_mom[d] -= step
                            x_mom = np.clip(x_mom, -5.0, 5.0)
                            f_mom = func(x_mom)
                            calls_made += 1
                            if f_mom < current_fx:
                                current_x = x_mom
                                current_fx = f_mom
                                update_best(current_x, current_fx)
                            else:
                                break

                if not improved:
                    break

        # Phase 3: Very fine refinement
        very_fine_steps = [0.0005, 0.0002, 0.0001, 0.00005, 0.00002, 0.00001]

        for step in very_fine_steps:
            if time.time() >= deadline or calls_made >= self.budget:
                break

            for _iter in range(50):
                if time.time() >= deadline or calls_made >= self.budget:
                    break

                improved = False
                for d in range(self.dim):
                    if time.time() >= deadline or calls_made >= self.budget:
                        break

                    # Try positive direction
                    x_pos = current_x.copy()
                    x_pos[d] += step
                    x_pos = np.clip(x_pos, -5.0, 5.0)
                    f_pos = func(x_pos)
                    calls_made += 1

                    if f_pos < current_fx:
                        current_x = x_pos
                        current_fx = f_pos
                        improved = True
                        update_best(current_x, current_fx)

                    # Try negative direction from original current_x
                    x_neg = current_x.copy()
                    x_neg[d] -= step
                    x_neg = np.clip(x_neg, -5.0, 5.0)
                    f_neg = func(x_neg)
                    calls_made += 1

                    if f_neg < current_fx:
                        current_x = x_neg
                        current_fx = f_neg
                        improved = True
                        update_best(current_x, current_fx)

                if not improved:
                    break

        if best_x is None:
            best_x = np.zeros(self.dim)
            best_fx = func(best_x)
            calls_made += 1

        return best_x, best_fx
# EVOLVE-BLOCK-END
