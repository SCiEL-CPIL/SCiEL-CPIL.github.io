"""Rewritten optimizer: coordinate-descent / Powell-style for Bent Cigar."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Coordinate descent with adaptive step sizes, tailored for Bent Cigar (f12)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin under 300s
        rng = np.random.default_rng(12345)

        # Initial point: center is a good guess for shifted/rotated instances
        # but let's try a few random starts and pick the best
        x = np.zeros(self.dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        calls = 1

        # Try a few random starts to get a better initial point
        for _ in range(5):
            if time.time() > deadline:
                break
            cand = rng.uniform(-4, 4, self.dim)
            cf = func(cand)
            calls += 1
            if cf < best_fx:
                best_fx = cf
                best_x = cand.copy()
                x = cand.copy()
                fx = cf

        # Adaptive step sizes per dimension
        # Bent cigar: f(x) = (x_1 - x*)^2 + 100 * sum((x_i - x*)^2 for i>1)
        # The first dimension (after rotation) has much smaller curvature.
        # We don't know the rotation, so use moderate initial steps.
        steps = np.ones(self.dim) * 1.0
        min_step = 1e-8
        max_step = 10.0

        # Coordinate descent with line search
        while calls < self.budget:
            if time.time() > deadline:
                break

            improved = False
            for d in range(self.dim):
                if time.time() > deadline or calls >= self.budget:
                    break

                # Try positive and negative directions
                best_local = fx
                best_local_d = 0.0

                # Adaptive step: try current step, then shrink/grow
                step = steps[d]
                for _ in range(6):
                    if calls >= self.budget or time.time() > deadline:
                        break

                    # Try +step
                    x[d] = np.clip(x[d] + step, -5, 5)
                    f_plus = func(x)
                    calls += 1
                    if f_plus < best_local:
                        best_local = f_plus
                        best_local_d = step
                    # Restore
                    x[d] -= step

                    # Try -step
                    x[d] = np.clip(x[d] - step, -5, 5)
                    f_minus = func(x)
                    calls += 1
                    if f_minus < best_local:
                        best_local = f_minus
                        best_local_d = -step
                    # Restore
                    x[d] += step

                    # If we found improvement, accept and shrink step
                    if best_local_d != 0.0:
                        x[d] = np.clip(x[d] + best_local_d, -5, 5)
                        fx = best_local
                        steps[d] *= 0.5
                        improved = True
                        break
                    else:
                        # No improvement at this step, shrink step
                        step *= 0.5
                        if step < min_step:
                            break

                if best_local < fx:
                    x[d] = np.clip(x[d] + best_local_d, -5, 5)
                    fx = best_local
                    improved = True

            if not improved:
                # No dimension improved; try a random perturbation to escape
                if calls < self.budget and time.time() < deadline:
                    perturb = rng.standard_normal(self.dim) * 0.5
                    x_try = np.clip(x + perturb, -5, 5)
                    f_try = func(x_try)
                    calls += 1
                    if f_try < fx:
                        x = x_try
                        fx = f_try
                        improved = True
                    else:
                        # Shrink all steps
                        steps *= 0.5
                        if np.all(steps < min_step):
                            break
            else:
                # Update best
                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()

        return best_x, best_fx
# EVOLVE-BLOCK-END
