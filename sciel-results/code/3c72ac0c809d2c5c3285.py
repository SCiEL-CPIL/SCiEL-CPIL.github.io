"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start coordinate descent with adaptive step sizes for BBOB f20."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _coord_descent(self, func, x, steps, budget, dim, rng, deadline, step_sizes):
        """Adaptive coordinate descent. Returns (best_x, best_f, evals_used)."""
        best_x = x.copy()
        best_f = func(best_x)
        evals = 1
        step_sizes = step_sizes.copy()

        # Adaptive step sizes per dimension
        # Start with moderate steps
        if step_sizes is None or len(step_sizes) != dim:
            step_sizes = np.full(dim, 1.0)

        stall_counter = 0
        max_stall = 30

        while evals < budget and time.time() < deadline:
            improved = False
            for d in range(dim):
                if evals >= budget or time.time() >= deadline:
                    break

                # Try positive and negative steps
                for direction in [1, -1]:
                    if evals >= budget or time.time() >= deadline:
                        break

                    trial = best_x.copy()
                    trial[d] += direction * step_sizes[d]
                    # Clip to box
                    trial = np.clip(trial, -5, 5)

                    f_trial = func(trial)
                    evals += 1

                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial
                        improved = True
                        # Increase step size slightly if we improved
                        step_sizes[d] *= 1.1
                    else:
                        # Decrease step size if no improvement
                        step_sizes[d] *= 0.95

            if not improved:
                stall_counter += 1
                # Random perturbation to escape local minima
                if stall_counter >= max_stall:
                    perturbation = rng.uniform(-0.5, 0.5, dim)
                    trial = np.clip(best_x + perturbation, -5, 5)
                    f_trial = func(trial)
                    evals += 1
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial
                    stall_counter = 0
            else:
                stall_counter = 0

            # Ensure step sizes don't get too small or too large
            step_sizes = np.clip(step_sizes, 1e-6, 10.0)

        return best_x, best_f, evals

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin: 330s limit, stop at 280s

        dim = self.dim
        budget = self.budget

        best_x_global = None
        best_f_global = np.inf

        rng = np.random.default_rng(42)

        # Phase 1: Multi-start coordinate descent
        # Use multiple starting points to explore the landscape
        n_starts = 8
        evals_per_start = max(10, budget // (n_starts + 2))

        # Starting points: center, corners, and random
        start_points = [np.zeros(dim)]  # Center
        # Add some structured starts
        for i in range(1, 4):
            pt = np.zeros(dim)
            for d in range(dim):
                pt[d] = rng.uniform(-4, 4)
            start_points.append(pt)

        # Add random starts
        for i in range(4):
            start_points.append(rng.uniform(-5, 5, dim))

        evals_used_total = 0

        for start_idx, start_point in enumerate(start_points):
            if evals_used_total >= budget or time.time() >= deadline:
                break

            # Determine budget for this start
            remaining_budget = budget - evals_used_total
            if remaining_budget <= 0:
                break

            # Allocate budget: earlier starts get more, later ones get less
            if start_idx < 3:
                alloc = min(evals_per_start * 2, remaining_budget)
            else:
                alloc = min(evals_per_start, remaining_budget)

            alloc = max(10, alloc)

            # Adaptive step sizes: start larger for exploration, smaller for exploitation
            if start_idx < 3:
                step_sizes = np.full(dim, 2.0)
            else:
                step_sizes = np.full(dim, 0.5)

            try:
                x, f, evals_used = self._coord_descent(
                    func, start_point, None, alloc, dim, rng, deadline, step_sizes
                )
            except Exception:
                break

            evals_used_total += evals_used

            if f < best_f_global:
                best_f_global = f
                best_x_global = x

        # Phase 2: Fine-tuning with smaller steps around the best point
        if best_x_global is not None and evals_used_total < budget and time.time() < deadline - 5:
            remaining = budget - evals_used_total
            if remaining > 5:
                fine_step_sizes = np.full(dim, 0.05)
                try:
                    x, f, evals_used = self._coord_descent(
                        func, best_x_global, None, remaining, dim, rng, deadline, fine_step_sizes
                    )
                except Exception:
                    pass

                if f < best_f_global:
                    best_f_global = f
                    best_x_global = x

        # Phase 3: Very fine polishing if budget remains
        if best_x_global is not None and time.time() < deadline - 2:
            # Use remaining budget for ultra-fine steps
            remaining = budget - evals_used_total
            if remaining > 2:
                ultra_step_sizes = np.full(dim, 0.001)
                try:
                    x, f, evals_used = self._coord_descent(
                        func, best_x_global, None, remaining, dim, rng, deadline, ultra_step_sizes
                    )
                except Exception:
                    pass

                if f < best_f_global:
                    best_f_global = f
                    best_x_global = x

        if best_x_global is None:
            best_x_global = np.zeros(dim)
            try:
                best_f_global = func(best_x_global)
            except Exception:
                best_f_global = np.inf

        return best_x_global, best_f_global
# EVOLVE-BLOCK-END
