"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Deterministic grid search + coordinate descent for f21 (multi-modal)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start = time.time()
        deadline = start + 320.0
        dim = self.dim
        budget = self.budget

        def time_ok():
            return time.time() < deadline and self._calls_made < budget

        self._calls_made = 0

        def eval_func(x):
            self._calls_made += 1
            return func(x)

        # Phase 1: Evaluate all 2^5 = 32 vertices of [-5,5]^5
        vertices = []
        for mask in range(2 ** dim):
            v = np.array([(5.0 if (mask >> i) & 1 else -5.0) for i in range(dim)])
            vertices.append(v)

        best_x = vertices[0].copy()
        best_f = eval_func(vertices[0])

        for v in vertices[1:]:
            if not time_ok():
                break
            f = eval_func(v)
            if f < best_f:
                best_f = f
                best_x = v.copy()

        # Phase 2: Coordinate descent with adaptive step sizes
        # Start with large steps, shrink as we converge
        step_sizes = [4.0] * dim
        x = best_x.copy()
        f_cur = best_f

        # Try to improve with coordinate descent
        max_iter = 200
        for it in range(max_iter):
            if not time_ok():
                break

            improved = False
            for d in range(dim):
                # Try positive and negative steps
                for sign in [1, -1]:
                    trial = x.copy()
                    trial[d] += sign * step_sizes[d]
                    # Clip to box
                    trial = np.clip(trial, -5.0, 5.0)
                    if not time_ok():
                        break
                    f_trial = eval_func(trial)
                    if f_trial < f_cur:
                        x = trial
                        f_cur = f_trial
                        best_f = f_cur
                        best_x = x.copy()
                        improved = True
                        break  # Don't try opposite direction if this one worked
                if not time_ok():
                    break

            if not improved:
                # No improvement, shrink steps
                step_sizes = [s * 0.5 for s in step_sizes]
                if min(step_sizes) < 0.01:
                    break
            else:
                # Keep step sizes if we improved, or slightly reduce for finer search
                step_sizes = [s * 0.8 for s in step_sizes]
                if min(step_sizes) < 0.01:
                    # Switch to very fine local search
                    step_sizes = [0.01] * dim

        return best_x, best_f
# EVOLVE-BLOCK-END
