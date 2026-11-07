"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Rastrigin-aware optimizer exploiting periodicity: coarse basin jump + 1/1-ES local."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(12345)
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 270  # 30s safety margin

        def clip(v):
            return np.clip(v, -5.0, 5.0)

        # Initial evaluation
        x0 = rng.uniform(-5.0, 5.0, dim)
        x0 = clip(x0)
        fx0 = func(x0)
        best_x = x0.copy()
        best_f = fx0
        calls = 1

        # Rastrigin: f(x) = 10*dim + sum(x_i^2 - 10*cos(2*pi*x_i))
        # Global minimum at x=0 (before shift). Period is 2 in each coordinate.
        # Strategy: coarse grid over integer basins, then local ES.

        # Phase 1: Coarse basin exploration
        # The Rastrigin function has minima at integer multiples of 2 (before shift).
        # After hidden shift/rotation, the landscape is still periodic-ish.
        # Try a set of candidate starting points spread across the box.

        # Generate candidates at multiples of 2 (potential basin centers)
        # with some randomization
        candidates = []
        for i in range(dim):
            candidates.append(-4.0)
            candidates.append(0.0)
            candidates.append(4.0)

        # Try a few structured starting points
        structured_starts = []
        # All zeros
        structured_starts.append(np.zeros(dim))
        # Integer combinations
        for val in [-4, -2, 0, 2, 4]:
            structured_starts.append(np.full(dim, float(val)))

        # Add some random starts
        for _ in range(10):
            structured_starts.append(rng.uniform(-5, 5, dim))

        # Evaluate structured starts, keeping best
        for start in structured_starts:
            if calls >= budget:
                break
            s = clip(start)
            fs = func(s)
            calls += 1
            if fs < best_f:
                best_f = fs
                best_x = s.copy()

        # Phase 2: Multi-start 1/1-ES from the best point, with periodic-aware jumps
        # Use a (1+1)-ES with adaptive step size and periodic jumps when stagnated

        x = best_x.copy()
        fx = best_f
        sigma = 1.0
        stagnation = 0
        max_stagnation = 15
        jump_cooldown = 0

        for _ in range(budget - calls):
            if calls >= budget:
                break
            if time.time() > deadline:
                break

            # Periodic jump: when stagnated, jump to nearby integer basin
            if stagnation >= max_stagnation and jump_cooldown <= 0:
                # Jump: pick a random dimension, shift by ±2 (one period)
                dim_to_jump = rng.integers(dim)
                direction = rng.choice([-1, 1])
                x = x.copy()
                x[dim_to_jump] += direction * 2.0
                x = clip(x)
                fx = func(x)
                calls += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                stagnation = 0
                jump_cooldown = 10
                sigma = 2.0
                continue

            if jump_cooldown > 0:
                jump_cooldown -= 1

            # Standard (1+1)-ES step
            y = clip(x + sigma * rng.standard_normal(dim))
            fy = func(y)
            calls += 1

            if fy < fx:
                x, fx = y, fy
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                sigma *= 1.3
                stagnation = 0
            else:
                sigma *= 0.85
                stagnation += 1

            # Keep sigma in reasonable range
            sigma = max(1e-4, min(sigma, 5.0))

        return best_x, best_f
# EVOLVE-BLOCK-END
