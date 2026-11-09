"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start optimizer with adaptive (1+lambda)-ES refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin before 300s kill

        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls = 0

        def evaluate(x):
            nonlocal calls, best_x, best_f
            if time.time() > deadline:
                return np.inf
            calls += 1
            f_val = func(x)
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # Phase 1: Multi-start coarse exploration
        # Weierstrass has many local minima; sampling broadly helps find the global basin
        n_init = min(25, budget // 10)
        for i in range(n_init):
            if calls >= budget or time.time() > deadline:
                break
            x = rng.uniform(-4.5, 4.5, dim)
            evaluate(x)

        if calls >= budget or time.time() > deadline:
            return best_x, best_f

        # Phase 2: Population-based local refinement from best point
        # Use (1+lambda)-ES with 1/5 success rule for adaptive step size
        x = best_x.copy() if best_x is not None else rng.uniform(-1, 1, dim)
        fx = best_f

        sigma = 1.0
        pop_size = 6
        successes = []
        window_size = 12
        stagnation = 0
        max_stagnation = 35

        while calls < budget and time.time() < deadline:
            # Generate population of offspring
            improvements = 0
            for j in range(pop_size):
                if calls >= budget or time.time() > deadline:
                    break

                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = evaluate(y)

                if fy < fx:
                    improvements += 1
                    x, fx = y, fy
                    successes.append(1)
                    stagnation = 0
                else:
                    successes.append(0)
                    stagnation += 1

            # Trim success window
            if len(successes) > window_size:
                successes = successes[-window_size:]

            # 1/5 success rule: adapt sigma based on success rate
            if len(successes) >= window_size:
                rate = sum(successes) / window_size
                if rate > 0.25:
                    sigma *= 1.15
                elif rate < 0.10:
                    sigma *= 0.80
                else:
                    sigma *= 0.95

            # Clamp sigma
            sigma = np.clip(sigma, 0.005, 4.0)

            # Restart mechanism: if stuck, try larger jumps or reset
            if stagnation > max_stagnation:
                if sigma < 0.15:
                    # Very small step: likely in a sharp local min, restart with larger sigma
                    x = best_x.copy()
                    fx = best_f
                    sigma = 2.0
                    stagnation = 0
                    successes = []
                else:
                    # Moderate step: try a few larger jumps to escape plateau
                    for _ in range(min(5, budget - calls)):
                        if calls >= budget or time.time() > deadline:
                            break
                        y = x + sigma * 2.0 * rng.standard_normal(dim)
                        y = np.clip(y, -5, 5)
                        fy = evaluate(y)
                        if fy < fx:
                            x, fx = y, fy
                            stagnation = 0
                            break
                    else:
                        # No improvement from jumps, shrink and continue
                        sigma *= 0.7
                        stagnation = 0
                    successes = []

        # Phase 3: Fine-tuning with small steps
        if calls < budget and time.time() < deadline and best_x is not None:
            x = best_x.copy()
            fx = best_f
            fine_sigma = 0.05

            for _ in range(min(50, budget - calls)):
                if calls >= budget or time.time() > deadline:
                    break
                y = x + fine_sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = evaluate(y)
                if fy < fx:
                    x, fx = y, fy
                    fine_sigma *= 1.1
                else:
                    fine_sigma *= 0.85

                if fine_sigma < 0.001:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
