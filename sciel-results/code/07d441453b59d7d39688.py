"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Deterministic multi-scale coordinate descent for Lunacek bi-Rastrigin."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _coordinate_descent(self, func, x, fx, step_sizes, calls_made, deadline, budget):
        """Run multi-scale coordinate descent from x with given step sizes."""
        dim = self.dim
        for sigma in step_sizes:
            improved = False
            for coord in range(dim):
                if time.time() > deadline or calls_made >= budget:
                    return x, fx, calls_made
                # Try positive and negative steps along this coordinate
                for sign in [1.0, -1.0]:
                    y = x.copy()
                    y[coord] = np.clip(y[coord] + sign * sigma, -5.0, 5.0)
                    fy = func(y)
                    calls_made += 1
                    if fy < fx:
                        x, fx = y, fy
                        improved = True
            if not improved:
                break  # Plateau at this scale, move to next
        return x, fx, calls_made

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 310.0  # 20s safety margin before 330s kill
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_made = 0

        # Fixed geometric step size sequence: 4.0 down to 1e-4
        step_sizes = np.geomspace(4.0, 1e-4, 12)

        # Strategic initial points: corners, center, and shifted positions
        # Lunacek's global min tends to be away from origin; try diverse starts
        init_points = []
        
        # Corners of the box (sampled subset for 5D)
        corners = []
        for i in range(16):
            pt = np.array([5.0 if (i >> j) & 1 else -5.0 for j in range(dim)])
            corners.append(pt)
        init_points.extend(corners)
        
        # Center
        init_points.append(np.zeros(dim))
        
        # Shifted points (Lunacek often has global min offset from origin)
        for offset in [2.0, -2.0, 3.0, -3.0, 1.5, -1.5]:
            init_points.append(np.full(dim, offset))
        
        # Mixed points
        init_points.append(np.array([2.0, -2.0, 3.0, -1.0, 1.5]))
        init_points.append(np.array([-3.0, 2.0, -1.5, 3.0, -2.0]))
        init_points.append(np.array([1.0, 1.0, -2.0, -1.0, 2.0]))

        # Deduplicate
        unique_inits = []
        seen = set()
        for p in init_points:
            key = tuple(np.round(p, 4))
            if key not in seen:
                seen.add(key)
                unique_inits.append(p)

        # Phase 1: Quick screen of initial points (1 call each)
        for p in unique_inits:
            if time.time() > deadline or calls_made >= budget:
                break
            p = np.clip(p, -5.0, 5.0)
            fp = func(p)
            calls_made += 1
            if fp < best_f:
                best_f = fp
                best_x = p.copy()

        # Phase 2: Multi-scale coordinate descent from best point
        if best_x is not None and calls_made < budget:
            x, fx, calls_made = self._coordinate_descent(
                func, best_x.copy(), best_f, step_sizes, calls_made, deadline, budget
            )
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 3: If budget remains, do second pass from best with finer steps
        if calls_made < budget and time.time() < deadline:
            fine_steps = np.geomspace(1.0, 1e-5, 10)
            x, fx, calls_made = self._coordinate_descent(
                func, best_x.copy(), best_f, fine_steps, calls_made, deadline, budget
            )
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 4: If budget remains, try a few more strategic points with descent
        if calls_made < budget and time.time() < deadline:
            additional_starts = [
                np.array([0.5, -0.5, 0.5, -0.5, 0.5]),
                np.array([-0.5, 0.5, -0.5, 0.5, -0.5]),
                np.array([1.0, -1.0, 1.0, -1.0, 1.0]),
                np.array([-1.0, 1.0, -1.0, 1.0, -1.0]),
            ]
            for p in additional_starts:
                if time.time() > deadline or calls_made >= budget:
                    break
                p = np.clip(p, -5.0, 5.0)
                fp = func(p)
                calls_made += 1
                if fp < best_f:
                    best_f = fp
                    best_x = p.copy()
                    # Do a quick descent from this new best
                    if calls_made < budget:
                        x, fx, calls_made = self._coordinate_descent(
                            func, best_x.copy(), best_f, step_sizes[:6], calls_made, deadline, budget
                        )
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()

        # Fallback
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls_made += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
