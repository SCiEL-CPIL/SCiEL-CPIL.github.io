"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with adaptive sigma, multi-start, and time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 300s limit, keep 10s margin
        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(0)

        best_x = None
        best_f = float('inf')
        calls = 0

        def evaluate(x):
            nonlocal best_x, best_f, calls
            x = np.clip(x, -5.0, 5.0)
            f = float(func(x))
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Multi-start to find a good basin
        # Use a small number of diverse starting points
        starts = []
        starts.append(np.zeros(d))
        starts.append(rng.uniform(-4, 4, d))
        starts.append(rng.uniform(-4, 4, d))
        starts.append(rng.uniform(-4, 4, d))
        starts.append(np.array([2.0] * d))
        starts.append(np.array([-2.0] * d))

        for s in starts:
            if calls >= budget or time.time() > deadline:
                break
            evaluate(s)

        # Phase 2: (1+1)-ES from the best point found
        if best_x is not None and calls < budget and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            sigma = 1.0  # Start with a moderate step size
            # Adaptive sigma: increase on success, decrease on failure
            # Use a slower adaptation rate for stability
            while calls < budget and time.time() < deadline:
                y = np.clip(x + sigma * rng.standard_normal(d), -5, 5)
                fy = evaluate(y)
                if fy < fx:
                    x, fx = y, fy
                    # Increase step size, but cap it to avoid huge jumps
                    sigma = min(sigma * 1.2, 4.0)
                else:
                    # Decrease step size
                    sigma *= 0.9
                    # If sigma becomes too small, try a random restart
                    if sigma < 0.01:
                        # Random restart: pick a new random point
                        if calls < budget - 10 and time.time() < deadline:
                            new_x = rng.uniform(-4, 4, d)
                            new_f = evaluate(new_x)
                            if new_f < fx:
                                x, fx = new_x, new_f
                                sigma = 1.0
                            else:
                                # Keep current best, but reset sigma
                                sigma = 0.5
                        else:
                            break

        return best_x, best_f
# EVOLVE-BLOCK-END
