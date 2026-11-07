"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with aggressive sigma, lattice-aware restarts, and polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280
        rng = np.random.default_rng(0)
        d = self.dim
        budget = self.budget

        best_x = None
        best_f = float('inf')
        total_calls = 0

        def check_deadline():
            return time.time() > deadline or total_calls >= budget

        # Phase 1: Main (1+1)-ES with aggressive sigma and lattice-aware restarts
        polish_budget = max(20, budget // 25)
        search_budget = budget - polish_budget

        # Start at origin
        x = np.zeros(d)
        fx = func(x)
        total_calls += 1
        if fx < best_f:
            best_f, best_x = fx, x.copy()

        sigma = 2.0
        no_improve = 0
        max_no_improve = 60
        restart_count = 0
        max_restarts = 6
        lattice_prob = 0.3  # Probability of using lattice-aware move

        # Precompute lattice points: multiples of 2 in [-4, 4]
        lattice_vals = np.array([-4.0, -2.0, 0.0, 2.0, 4.0])

        while total_calls < search_budget and not check_deadline():
            if rng.random() < lattice_prob:
                # Lattice-aware move: snap to nearest grid point + small offset
                # Choose a random grid point in each dimension
                grid_x = np.array([lattice_vals[rng.integers(0, len(lattice_vals))] for _ in range(d)])
                # Add small continuous perturbation
                y = np.clip(grid_x + sigma * 0.3 * rng.standard_normal(d), -5, 5)
            else:
                # Standard Gaussian step
                y = np.clip(x + sigma * rng.standard_normal(d), -5, 5)

            fy = func(y)
            total_calls += 1

            if fy < fx:
                x, fx = y, fy
                no_improve = 0
                sigma *= 1.5
                if fx < best_f:
                    best_f, best_x = fx, x.copy()
            else:
                sigma *= 0.87
                no_improve += 1

            # Lightweight restart: if stalled, jump to a lattice point
            if no_improve >= max_no_improve and restart_count < max_restarts:
                restart_count += 1
                # Start from a random lattice point
                x = np.array([lattice_vals[rng.integers(0, len(lattice_vals))] for _ in range(d)])
                fx = func(x)
                total_calls += 1
                if fx < best_f:
                    best_f, best_x = fx, x.copy()
                sigma = 1.5
                no_improve = 0

        # Phase 2: Final polish around the best point found
        if best_x is not None and not check_deadline():
            x = best_x.copy()
            fx = best_f
            sigma = 0.3

            remaining_calls = budget - total_calls
            polish_steps = min(polish_budget, remaining_calls)

            for _ in range(polish_steps):
                if check_deadline():
                    break

                y = np.clip(x + sigma * rng.standard_normal(d), -5, 5)
                fy = func(y)
                total_calls += 1

                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f, best_x = fx, x.copy()
                    sigma *= 1.1
                else:
                    sigma *= 0.9

                if sigma < 1e-8:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
