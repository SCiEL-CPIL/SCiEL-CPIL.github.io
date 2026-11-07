"""Bootstrap spawn: Lattice search + basin-jump ES for 5D Rastrigin."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Lattice-based search exploiting Rastrigin periodicity, then basin-jump ES."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        calls = [0]

        def call_func(x):
            calls[0] += 1
            return float(func(x))

        def time_ok():
            return calls[0] < budget and time.time() < deadline

        best_x = None
        best_f = np.inf

        # Phase 1: Structured lattice search
        grid_vals = np.array([-5.0, -3.0, -1.0, 1.0, 3.0, 5.0])
        grids = np.meshgrid(*([grid_vals] * dim), indexing='ij')
        grid_points = np.stack([g.ravel() for g in grids], axis=1)

        n_lattice = 120
        dists = np.linalg.norm(grid_points, axis=1)
        order = np.argsort(dists)
        selected = list(order[:40])
        remaining = list(order[40:])
        rng.shuffle(remaining)
        selected.extend(remaining[:80])
        selected = np.array(selected)

        for idx in selected:
            if not time_ok():
                break
            x = grid_points[idx]
            fx = call_func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: Basin-jump (1+1)-ES
        x = best_x.copy()
        fx = best_f
        sigma = 0.5
        no_improve = 0
        max_no_improve = 15
        jumps = 0
        max_jumps = 50

        while time_ok() and jumps < max_jumps:
            for _ in range(30):
                if not time_ok():
                    break
                step = sigma * rng.standard_normal(dim)
                x_trial = np.clip(x + step, -5, 5)
                fx_trial = call_func(x_trial)
                if fx_trial < fx:
                    x = x_trial
                    fx = fx_trial
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma *= 1.05
                    no_improve = 0
                else:
                    sigma *= 0.9
                    no_improve += 1

                if no_improve >= max_no_improve:
                    break

            if not time_ok():
                break

            if no_improve < max_no_improve:
                sigma = max(0.01, sigma * 0.5)
                no_improve = 0
                continue

            jumps += 1
            x_jump = rng.uniform(-5, 5, dim)
            fx_jump = call_func(x_jump)
            if fx_jump < best_f:
                best_f = fx_jump
                best_x = x_jump.copy()
            x = x_jump
            fx = fx_jump
            sigma = 0.5
            no_improve = 0

        # Phase 3: Final polish
        sigma = 0.05
        x = best_x.copy()
        fx = best_f
        for _ in range(min(50, budget - calls[0])):
            if not time_ok():
                break
            step = sigma * rng.standard_normal(dim)
            x_trial = np.clip(x + step, -5, 5)
            fx_trial = call_func(x_trial)
            if fx_trial < fx:
                x = x_trial
                fx = fx_trial
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                sigma *= 1.02
            else:
                sigma *= 0.9

        return best_x, best_f
# EVOLVE-BLOCK-END
