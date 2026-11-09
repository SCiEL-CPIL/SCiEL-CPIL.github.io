"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """(1+1)-ES with origin init, multiplicative sigma adaptation, restarts, and coordinate polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget

        # Time management
        start_time = time.time()
        deadline = start_time + 325.0  # Safety margin before 330s kill

        rng = np.random.default_rng(42)

        # Best solution found overall
        best_x = None
        best_f = np.inf
        eval_count = 0

        def try_update(x, fx):
            nonlocal best_x, best_f, eval_count
            eval_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 1: Main (1+1)-ES from origin
        x = np.zeros(n)
        fx = func(x)
        try_update(x, fx)

        sigma_min = 1e-6
        sigma_max = 50.0

        # Multiple restarts with adaptive sigma
        max_restarts = 4
        for restart in range(max_restarts):
            # Run (1+1)-ES
            sigma = 2.5 if restart == 0 else max(1.5, 2.0 - 0.25 * restart)
            no_improve = 0
            max_no_improve = 60 + restart * 20

            for i in range(budget - eval_count):
                if eval_count >= budget:
                    break
                if time.time() > deadline - 5.0:
                    break

                z = rng.standard_normal(n)
                y = np.clip(x + sigma * z, -5, 5)
                fy = func(y)
                eval_count += 1

                if fy < fx:
                    x = y
                    fx = fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma = min(sigma * 1.25, sigma_max)
                    no_improve = 0
                else:
                    sigma *= 0.9
                    no_improve += 1

                if sigma < sigma_min or no_improve >= max_no_improve:
                    break

            # If we used all budget or time is up, stop
            if eval_count >= budget or time.time() > deadline - 5.0:
                break

            # Restart: diverse starting points using structured patterns
            remaining = budget - eval_count
            if remaining < 10:
                break

            if restart == 0:
                # First restart: moderate perturbation of best
                if best_x is not None:
                    x = np.clip(best_x + rng.standard_normal(n) * 1.0, -5, 5)
                else:
                    x = rng.uniform(-4, 4, n)
            elif restart == 1:
                # Second restart: opposite quadrant bias
                signs = rng.choice([-1, 1], size=n)
                x = signs * rng.uniform(2, 4, n)
            elif restart == 2:
                # Third restart: near origin with small noise
                x = rng.standard_normal(n) * 0.5
            else:
                # Fourth restart: random in full box
                x = rng.uniform(-5, 5, n)

            x = np.clip(x, -5, 5)
            fx = func(x)
            eval_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: Coordinate descent polish on best solution
        if best_x is not None and eval_count < budget and time.time() < deadline - 3.0:
            x = best_x.copy()
            fx = best_f
            # Try decreasing step sizes
            for step in [1.0, 0.5, 0.25, 0.1, 0.05, 0.01]:
                if eval_count >= budget or time.time() > deadline - 3.0:
                    break
                for coord in range(n):
                    if eval_count >= budget or time.time() > deadline - 3.0:
                        break
                    # Try + direction
                    y = x.copy()
                    y[coord] = min(5.0, x[coord] + step)
                    fy = func(y)
                    eval_count += 1
                    if fy < fx:
                        x = y
                        fx = fy
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                    # Try - direction
                    if eval_count >= budget or time.time() > deadline - 3.0:
                        break
                    y = x.copy()
                    y[coord] = max(-5.0, x[coord] - step)
                    fy = func(y)
                    eval_count += 1
                    if fy < fx:
                        x = y
                        fx = fy
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
