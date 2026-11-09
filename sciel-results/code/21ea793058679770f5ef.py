"""Deterministic multi-scale coordinate descent + local refinement for Lunacek bi-Rastrigin."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Deterministic multi-scale search with coordinate descent and local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        calls = [0]

        def f(x):
            calls[0] += 1
            return func(x)

        # Pre-allocate best tracking
        best_x = None
        best_f = float('inf')

        def update_best(x, fx):
            nonlocal best_x, best_f
            if fx < best_f:
                best_x = x.copy()
                best_f = fx

        # Phase 1: Coarse grid to find promising regions
        # Lunacek has two basins: one near the shifted origin, one near a shifted secondary minimum
        # We'll do a coarse search first, then refine

        dim = self.dim
        budget = self.budget

        # Start with a small number of strategic points
        # The Lunacek function has a global minimum and a local minimum
        # Global min is typically at the "shifted" location, local min is offset

        # Coarse sampling: evaluate at center and a few offsets
        # Since we don't know the shift, use a small set of diverse starting points

        # Strategy: 
        # 1. Evaluate at origin (clipped)
        # 2. Do coordinate descent from multiple starts
        # 3. Use the best result to guide further refinement

        # Initial evaluations
        starts = [
            np.zeros(dim),
            np.full(dim, 1.0),
            np.full(dim, -1.0),
            np.array([1.0, -1.0, 1.0, -1.0, 1.0][:dim]),
            np.array([-1.0, 1.0, -1.0, 1.0, -1.0][:dim]),
            np.array([2.0, 0.0, -2.0, 0.0, 2.0][:dim]),
            np.array([0.0, 2.0, 0.0, -2.0, 0.0][:dim]),
        ]

        # Deduplicate and clip
        unique_starts = []
        seen = set()
        for s in starts:
            s_clipped = np.clip(s, -5, 5)
            key = tuple(np.round(s_clipped, 6))
            if key not in seen:
                seen.add(key)
                unique_starts.append(s_clipped)

        # Evaluate all starts
        for x0 in unique_starts:
            if calls[0] >= budget:
                break
            if time.time() > deadline:
                break
            fx = f(x0)
            update_best(x0, fx)

        # Phase 2: Coordinate descent from the best point
        # Use a fixed sequence of step sizes for determinism

        # Step sizes: geometric decay
        step_sizes = [4.0, 2.0, 1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 5e-4, 2e-4, 1e-4, 5e-5, 2e-5, 1e-5]

        # For each step size, do coordinate descent
        current_x = best_x.copy() if best_x is not None else np.zeros(dim)
        current_f = best_f

        for sigma in step_sizes:
            if calls[0] >= budget:
                break
            if time.time() > deadline:
                break

            improved = True
            while improved:
                improved = False
                for d in range(dim):
                    if calls[0] >= budget:
                        break
                    if time.time() > deadline:
                        break

                    # Try positive and negative steps along dimension d
                    for sign in [1, -1]:
                        if calls[0] >= budget:
                            break
                        if time.time() > deadline:
                            break

                        new_x = current_x.copy()
                        new_x[d] = np.clip(new_x[d] + sign * sigma, -5, 5)

                        # Skip if no change
                        if np.allclose(new_x, current_x, atol=1e-12):
                            continue

                        fx = f(new_x)
                        if fx < current_f:
                            current_x = new_x
                            current_f = fx
                            update_best(current_x, current_f)
                            improved = True

                if calls[0] >= budget:
                    break
                if time.time() > deadline:
                    break

        # Phase 3: Local refinement with smaller steps if budget remains
        # Try a few more fine-grained steps
        fine_steps = [5e-5, 2e-5, 1e-5, 5e-6, 2e-6, 1e-6]
        for sigma in fine_steps:
            if calls[0] >= budget:
                break
            if time.time() > deadline:
                break

            for _ in range(3):  # up to 3 passes at this scale
                if calls[0] >= budget:
                    break
                if time.time() > deadline:
                    break

                improved = False
                for d in range(dim):
                    if calls[0] >= budget:
                        break
                    if time.time() > deadline:
                        break

                    for sign in [1, -1]:
                        if calls[0] >= budget:
                            break
                        if time.time() > deadline:
                            break

                        new_x = current_x.copy()
                        new_x[d] = np.clip(new_x[d] + sign * sigma, -5, 5)

                        if np.allclose(new_x, current_x, atol=1e-12):
                            continue

                        fx = f(new_x)
                        if fx < current_f:
                            current_x = new_x
                            current_f = fx
                            update_best(current_x, current_f)
                            improved = True

                if not improved:
                    break

        # Phase 4: If we have budget left, try a few more diverse starts and refine
        if calls[0] < budget and time.time() < deadline:
            # Try a few more random-ish deterministic starts
            extra_starts = [
                np.array([0.5, 0.5, 0.5, 0.5, 0.5][:dim]),
                np.array([-0.5, -0.5, -0.5, -0.5, -0.5][:dim]),
                np.array([3.0, -3.0, 3.0, -3.0, 3.0][:dim]),
                np.array([-3.0, 3.0, -3.0, 3.0, -3.0][:dim]),
                np.array([1.5, 1.5, -1.5, -1.5, 1.5][:dim]),
            ]

            for x0 in extra_starts:
                if calls[0] >= budget:
                    break
                if time.time() > deadline:
                    break

                x0 = np.clip(x0, -5, 5)
                fx = f(x0)
                if fx < current_f:
                    current_x = x0.copy()
                    current_f = fx
                    update_best(current_x, current_f)

                    # Quick coordinate descent from this new best
                    for sigma in [1.0, 0.5, 0.1, 0.05, 0.01]:
                        if calls[0] >= budget:
                            break
                        if time.time() > deadline:
                            break
                        for d in range(dim):
                            if calls[0] >= budget:
                                break
                            for sign in [1, -1]:
                                if calls[0] >= budget:
                                    break
                                new_x = current_x.copy()
                                new_x[d] = np.clip(new_x[d] + sign * sigma, -5, 5)
                                if np.allclose(new_x, current_x, atol=1e-12):
                                    continue
                                fx2 = f(new_x)
                                if fx2 < current_f:
                                    current_x = new_x
                                    current_f = fx2
                                    update_best(current_x, current_f)

        return best_x, best_f
# EVOLVE-BLOCK-END
