"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Coordinate-descent hybrid with adaptive per-dim steps and random perturbations."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 325  # safety margin before 330s kill

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_fx = np.inf
        calls = 0

        def update_best(x, fx):
            nonlocal best_x, best_fx
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

        def coordinate_descent(x, fx, n_steps, max_calls):
            """Adaptive per-dimension coordinate descent with golden-section-like refinement."""
            nonlocal calls
            for step in range(n_steps):
                if calls >= max_calls:
                    break
                if time.time() >= deadline - 2:
                    break
                for d in range(dim):
                    if calls >= max_calls:
                        break
                    if time.time() >= deadline - 2:
                        break
                    # Try positive and negative steps along axis d
                    for sign in (1.0, -1.0):
                        if calls >= max_calls:
                            break
                        if time.time() >= deadline - 2:
                            break
                        step_size = 1.0
                        improved = False
                        for _ in range(4):
                            y = x.copy()
                            y[d] += sign * step_size
                            y = np.clip(y, -5, 5)
                            fy = func(y)
                            calls += 1
                            if fy < fx:
                                x, fx = y, fy
                                improved = True
                                step_size *= 0.5
                            else:
                                break
                        if not improved:
                            continue
                    # If we improved, keep going; the next iteration will refine further
            update_best(x, fx)
            return x, fx

        # Start at center of the box
        x = np.zeros(dim)
        fx = func(x)
        calls += 1
        update_best(x, fx)

        # Phase 1: Global exploration with random perturbations + periodic CD polish
        # Use larger random steps to explore, then polish with coordinate descent
        explore_steps = int(budget * 0.6)
        polish_every = 50  # do CD polish every 50 exploration steps

        for i in range(explore_steps):
            if calls >= budget:
                break
            if time.time() >= deadline - 2:
                break

            # Random perturbation with adaptive scale
            scale = max(0.5, 2.0 * np.exp(-0.01 * i))
            y = x + scale * rng.standard_normal(dim)
            y = np.clip(y, -5, 5)
            fy = func(y)
            calls += 1

            if fy < fx:
                x, fx = y, fy
                update_best(x, fx)
            # else: keep current x, but still try a small step in random direction

            # Periodic coordinate descent polish
            if (i + 1) % polish_every == 0:
                remaining = budget - calls
                if remaining > 20:
                    x, fx = coordinate_descent(x, fx, n_steps=2, max_calls=budget)

        # Phase 2: Final intensive coordinate descent polish
        if calls < budget:
            remaining = budget - calls
            if remaining > 10:
                # Multiple rounds of CD with decreasing step sizes
                for round_num in range(3):
                    if calls >= budget:
                        break
                    if time.time() >= deadline - 2:
                        break
                    x, fx = coordinate_descent(x, fx, n_steps=5, max_calls=budget)

        # Phase 3: Very fine local search
        if calls < budget:
            # Try tiny perturbations around best
            for _ in range(min(50, budget - calls)):
                if calls >= budget:
                    break
                if time.time() >= deadline - 2:
                    break
                y = best_x + 0.1 * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls += 1
                if fy < best_fx:
                    best_fx = fy
                    best_x = y.copy()

        return best_x, best_fx
# EVOLVE-BLOCK-END
