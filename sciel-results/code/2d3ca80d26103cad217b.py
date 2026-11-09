"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Portfolio optimizer for BBOB f18 (Schaffers F7 ill-conditioned).
    - Random sampling phase to find a promising region.
    - Adaptive (1+1)-ES with periodic restarts to escape local minima.
    - Time guard to stay well under the 330s wall-clock limit.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # 10s safety margin

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Phase 1: Random sampling to find a good initial point
        n_samples = min(50, budget // 10)
        best_x = None
        best_fx = float('inf')

        for i in range(n_samples):
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

        # Phase 2: Adaptive (1+1)-ES with restarts
        calls_used = n_samples
        best_x = best_x.copy()
        best_fx = best_fx

        # ES parameters
        sigma = 1.5
        stall_count = 0
        max_stall = 20
        restart_sigma = 2.0

        while calls_used < budget:
            # Time check
            if time.time() > deadline - 5.0:
                break

            # Generate candidate
            y = best_x + sigma * rng.standard_normal(dim)
            y = np.clip(y, -5, 5)
            fy = func(y)
            calls_used += 1

            if fy < best_fx:
                best_x = y.copy()
                best_fx = fy
                sigma *= 1.2
                stall_count = 0
            else:
                sigma *= 0.85
                stall_count += 1

            # Restart if stalled
            if stall_count >= max_stall:
                # Reset to best point with larger step size
                sigma = max(restart_sigma, sigma * 2.0)
                restart_sigma *= 1.5
                stall_count = 0
                max_stall = min(max_stall + 10, 80)

        return best_x, best_fx
# EVOLVE-BLOCK-END
