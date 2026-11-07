import numpy as np
import time


class Optimizer:
    """Multi-restart (1+1)-ES with adaptive step size and final fine-tuning for Katsuura."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0
        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        used_calls = 0

        def check_time():
            return time.time() > deadline

        def try_point(x):
            nonlocal best_x, best_f, used_calls
            if used_calls >= budget or check_time():
                return
            fx = func(x)
            used_calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 1: Initial exploration - evaluate origin and a few structured points
        try_point(np.zeros(d))

        # Phase 2: Multi-restart (1+1)-ES with adaptive step size
        # Use several restarts to explore different regions of the multimodal landscape
        num_restarts = 5
        calls_per_restart = max(10, (budget - used_calls) // num_restarts)
        sigma = 2.0  # Initial step size
        sigma_min = 1e-6
        sigma_max = 5.0
        alpha = 0.85  # Step size decay when no improvement
        beta = 1.5    # Step size growth when improvement (conservative)

        for restart in range(num_restarts):
            if used_calls >= budget or check_time():
                break

            # Initialize current point: for first restart use best so far, else random
            if restart == 0 and best_x is not None:
                x = best_x.copy()
            else:
                x = rng.uniform(-4, 4, d)
                try_point(x)

            # (1+1)-ES loop for this restart
            for _ in range(calls_per_restart):
                if used_calls >= budget or check_time():
                    break

                # Generate candidate
                y = x + sigma * rng.standard_normal(d)
                y = np.clip(y, -5, 5)

                # Evaluate candidate
                fy = func(y)
                used_calls += 1

                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()

                if fy < best_f + 1e-12:  # Accept if not worse than current best (including global)
                    x = y.copy()
                    # Increase step size slightly on success
                    sigma = min(sigma * 1.1, sigma_max)
                else:
                    # Decrease step size on failure
                    sigma = max(sigma * alpha, sigma_min)

        # Phase 3: Final fine-tuning with very small steps around global best
        if best_x is not None and used_calls < budget and not check_time():
            x = best_x.copy()
            fx = best_f
            remaining = budget - used_calls

            # Try small random steps with decreasing magnitude
            for scale in [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]:
                if used_calls >= budget or check_time():
                    break
                for _ in range(min(30, remaining)):
                    if used_calls >= budget or check_time():
                        break
                    z = rng.standard_normal(d)
                    y = np.clip(x + scale * z, -5, 5)
                    try_point(y)
                    if best_x is not None and best_f < fx:
                        x = best_x.copy()
                        fx = best_f

        return best_x, best_f
