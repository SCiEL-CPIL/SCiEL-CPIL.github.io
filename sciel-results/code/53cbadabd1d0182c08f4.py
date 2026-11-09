"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-phase optimizer for BBOB f17 (Schaffers F7)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 300s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = float('inf')
        calls = 0

        def try_update(x, fx):
            nonlocal best_x, best_f
            if fx < best_f:
                best_x, best_f = x.copy(), fx

        # Phase 1: Initial random sampling (10% of budget)
        n_sample = max(10, budget // 10)
        for _ in range(n_sample):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            calls += 1
            try_update(x, fx)

        if best_x is None:
            best_x = rng.uniform(-5, 5, dim)
            best_f = func(best_x)
            calls += 1

        # Phase 2: (1+1)-ES with tuned step size adaptation
        x = best_x.copy()
        fx = best_f
        sigma = 1.0
        sigma_min = 1e-10
        sigma_max = 10.0

        remaining = budget - calls
        for i in range(remaining):
            if time.time() > deadline:
                break

            # Generate candidate
            step = sigma * rng.standard_normal(dim)
            y = np.clip(x + step, -5, 5)
            fy = func(y)
            calls += 1

            if fy < fx:
                x, fx = y, fy
                try_update(x, fx)
                sigma = min(sigma * 1.2, sigma_max)
            else:
                sigma = max(sigma * 0.9, sigma_min)

            # If we've converged very well, do a local refinement
            if fx < 1e-4 and sigma < 0.01:
                # Try a few very small steps
                for _ in range(min(5, remaining - i)):
                    if time.time() > deadline:
                        break
                    tiny_step = 0.001 * rng.standard_normal(dim)
                    z = np.clip(x + tiny_step, -5, 5)
                    fz = func(z)
                    calls += 1
                    if fz < fx:
                        x, fx = z, fz
                        try_update(x, fx)

        # Phase 3: If we have budget left, do a few random restarts from near best
        if calls < budget and best_f > 1e-6:
            n_restarts = min(5, budget - calls)
            for _ in range(n_restarts):
                if time.time() > deadline:
                    break
                # Perturb the best solution
                x = np.clip(best_x + 0.5 * rng.standard_normal(dim), -5, 5)
                fx = func(x)
                calls += 1
                try_update(x, fx)

                # Short (1+1)-ES from here
                sigma = 0.5
                for _ in range(min(20, budget - calls)):
                    if time.time() > deadline:
                        break
                    y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                    fy = func(y)
                    calls += 1
                    if fy < fx:
                        x, fx = y, fy
                        try_update(x, fx)
                        sigma = min(sigma * 1.2, sigma_max)
                    else:
                        sigma = max(sigma * 0.9, sigma_min)

        return best_x, best_f
# EVOLVE-BLOCK-END
