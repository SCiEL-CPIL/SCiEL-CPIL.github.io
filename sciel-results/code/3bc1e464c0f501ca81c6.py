"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES style optimizer with adaptive step size and restarts."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        best_x = None
        best_f = np.inf
        calls_used = 0
        remaining = self.budget

        # Use a population-based approach with adaptive step size
        # We'll use a simplified CMA-ES-like strategy with multiple restarts
        pop_size = 10
        max_restarts = 3

        for restart in range(max_restarts):
            if remaining <= pop_size:
                break

            # Initialize population around current best or random if first restart
            if best_x is not None:
                center = best_x.copy()
                sigma = 0.5
            else:
                center = rng.uniform(-5, 5, self.dim)
                sigma = 2.0

            # Generate initial population
            pop = []
            for i in range(pop_size):
                if i == 0:
                    x = center.copy()
                else:
                    x = np.clip(center + sigma * rng.standard_normal(self.dim), -5, 5)
                fx = func(x)
                calls_used += 1
                remaining -= 1
                pop.append((fx, x))
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

            if remaining <= 0:
                break

            # Main loop: evaluate one new point per iteration
            # Use a simple adaptive strategy based on recent improvements
            sigma = 2.0 if restart == 0 else 1.0
            success_count = 0
            total_trials = 0
            window = 5

            while remaining > 0:
                # Generate new candidate
                if best_x is not None:
                    y = np.clip(best_x + sigma * rng.standard_normal(self.dim), -5, 5)
                else:
                    y = rng.uniform(-5, 5, self.dim)

                fy = func(y)
                calls_used += 1
                remaining -= 1

                total_trials += 1

                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                    success_count += 1
                    sigma *= 1.1  # Increase step size on success
                else:
                    sigma *= 0.95  # Decrease step size on failure

                # Keep sigma in reasonable bounds
                sigma = np.clip(sigma, 0.01, 5.0)

                # If we've made good progress, we might want to restart with smaller sigma
                # Check if we're stuck (no improvement in last several trials)
                if total_trials > 50 and success_count == 0:
                    # Try a bigger jump
                    if best_x is not None:
                        y = np.clip(best_x + 2.0 * rng.standard_normal(self.dim), -5, 5)
                        fy = func(y)
                        calls_used += 1
                        remaining -= 1
                        if remaining > 0:
                            if fy < best_f:
                                best_f = fy
                                best_x = y.copy()
                                success_count = 1
                            else:
                                sigma = max(sigma * 0.5, 0.01)
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
