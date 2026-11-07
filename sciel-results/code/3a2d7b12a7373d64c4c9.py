"""Rewritten optimizer: batch axis-probing + direct jump + (1+1)-ES polish with deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Axis-probing initializer followed by (1+1)-ES with adaptive step size."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        d = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s wall-clock

        # --- Phase 1: Batch axis probing to locate the optimum quickly ---
        # Evaluate center, then probe along each axis in both directions.
        # This gives us directional gradient info to jump toward the optimum.
        x_center = np.zeros(d)
        f_center = func(x_center)
        best_x = x_center.copy()
        best_f = f_center
        calls = 1

        # Probe each axis at a moderate step
        probe_step = 4.0
        axis_vals = {}
        for i in range(d):
            x_plus = x_center.copy()
            x_plus[i] = probe_step
            f_plus = func(x_plus)
            calls += 1
            if f_plus < best_f:
                best_f = f_plus
                best_x = x_plus.copy()

            x_minus = x_center.copy()
            x_minus[i] = -probe_step
            f_minus = func(x_minus)
            calls += 1
            if f_minus < best_f:
                best_f = f_minus
                best_x = x_minus.copy()

            axis_vals[i] = (f_plus, f_minus)

        # Check deadline
        if time.time() > deadline:
            return best_x, best_f

        # --- Phase 2: Direct jump using linear interpolation along best axis ---
        # For a linear function, the optimum is at the boundary in the direction
        # of steepest descent. Use the axis probe values to estimate direction.
        # Compute gradient estimate from axis probes
        grad_est = np.zeros(d)
        for i in range(d):
            fp, fm = axis_vals[i]
            grad_est[i] = (fp - fm) / (2.0 * probe_step)

        # Direction of steepest descent
        norm_g = np.linalg.norm(grad_est)
        if norm_g > 1e-12:
            direction = -grad_est / norm_g
            # Jump to boundary in that direction
            # Find max step to stay in [-5,5]
            steps = []
            for i in range(d):
                if direction[i] > 1e-12:
                    steps.append((5.0 - best_x[i]) / direction[i])
                elif direction[i] < -1e-12:
                    steps.append((-5.0 - best_x[i]) / direction[i])
            max_step = min(steps) if steps else 0.0
            max_step = max(0.0, max_step)

            # Try a few points along the direction
            for frac in [1.0, 0.8, 0.6, 0.4, 0.2]:
                candidate = best_x + frac * max_step * direction
                candidate = np.clip(candidate, -5, 5)
                fc = func(candidate)
                calls += 1
                if fc < best_f:
                    best_f = fc
                    best_x = candidate
                if time.time() > deadline:
                    return best_x, best_f

        # --- Phase 3: (1+1)-ES polish from best point found ---
        sigma = 1.0
        remaining = budget - calls
        for _ in range(remaining):
            if time.time() > deadline:
                break
            y = np.clip(best_x + sigma * rng.standard_normal(d), -5, 5)
            fy = func(y)
            calls += 1
            if fy < best_f:
                best_x = y
                best_f = fy
                sigma *= 1.3
            else:
                sigma *= 0.85
            if sigma < 1e-8:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
