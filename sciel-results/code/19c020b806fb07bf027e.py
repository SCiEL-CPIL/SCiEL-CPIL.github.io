"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Random Coordinate Descent with adaptive step sizes and restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit
        n = self.dim
        budget = self.budget

        rng = np.random.default_rng(42)

        # Initial point: start at the known Rosenbrock minimum location
        x = np.ones(n)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        calls_made = 1

        # Per-coordinate step sizes
        sigmas = np.ones(n) * 1.0
        min_sigma = 1e-4
        max_sigma = 10.0

        # Stagnation tracking
        no_improve = 0
        max_no_improve = 40
        restart_count = 0

        while calls_made < budget:
            if time.time() > deadline:
                break

            # Randomly select a coordinate to perturb
            coord = rng.integers(n)

            # Generate candidate with perturbation in one coordinate
            step = sigmas[coord] * rng.standard_normal()
            y = x.copy()
            y[coord] = np.clip(y[coord] + step, -5, 5)

            fy = func(y)
            calls_made += 1

            if fy < fx:
                # Accept the move
                x = y
                fx = fy
                no_improve = 0

                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()

                # Increase step size on success (geometric growth)
                sigmas[coord] = min(sigmas[coord] * 1.2, max_sigma)
            else:
                # Reject the move
                no_improve += 1

                # Decrease step size on failure (geometric shrink)
                sigmas[coord] = max(sigmas[coord] * 0.8, min_sigma)

                # Restart logic when stuck
                if no_improve >= max_no_improve:
                    restart_count += 1

                    # Try a few local perturbations around best point
                    local_tries = min(3, budget - calls_made)
                    found_better = False

                    for _ in range(local_tries):
                        if calls_made >= budget or time.time() > deadline:
                            break
                        # Perturb a random coordinate around best
                        c = rng.integers(n)
                        candidate = best_x.copy()
                        candidate[c] = np.clip(candidate[c] + 1.5 * rng.standard_normal(), -5, 5)
                        cf = func(candidate)
                        calls_made += 1
                        if cf < best_fx:
                            best_fx = cf
                            best_x = candidate.copy()
                            x = best_x.copy()
                            fx = best_fx
                            no_improve = 0
                            sigmas[c] = 1.0
                            found_better = True
                            break

                    if not found_better:
                        # Global random restart, biased toward center for Rosenbrock
                        if restart_count % 2 == 0:
                            x_new = rng.uniform(-3, 3, n)
                        else:
                            x_new = rng.uniform(-5, 5, n)
                        fx_new = func(x_new)
                        calls_made += 1
                        if fx_new < best_fx:
                            best_fx = fx_new
                            best_x = x_new.copy()
                            x = x_new.copy()
                            fx = best_fx
                        else:
                            x = x_new.copy()
                            fx = fx_new
                        no_improve = 0
                        sigmas = np.ones(n) * 1.5

        # Final refinement: do a few more coordinate passes with smaller steps
        remaining = budget - calls_made
        if remaining > 5 and time.time() < deadline - 10:
            # Shrink all sigmas for fine-tuning
            sigmas = np.clip(sigmas * 0.3, min_sigma, max_sigma)
            for _ in range(remaining):
                if calls_made >= budget or time.time() > deadline:
                    break
                coord = rng.integers(n)
                step = sigmas[coord] * rng.standard_normal()
                y = best_x.copy()
                y[coord] = np.clip(y[coord] + step, -5, 5)
                fy = func(y)
                calls_made += 1
                if fy < best_fx:
                    best_fx = fy
                    best_x = y.copy()
                    sigmas[coord] = min(sigmas[coord] * 1.1, max_sigma)
                else:
                    sigmas[coord] = max(sigmas[coord] * 0.85, min_sigma)

        return best_x, best_fx
# EVOLVE-BLOCK-END
