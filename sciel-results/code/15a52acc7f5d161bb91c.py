"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with adaptive sigma, restarts, and final polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        # Phase 1: Initial exploration - evaluate a few diverse points
        # to find a promising region
        candidates = []
        for i in range(min(10, self.budget // 5)):
            x = rng.uniform(-4, 4, self.dim)
            fx = func(x)
            candidates.append((fx, x))
        
        if not candidates:
            x = np.zeros(self.dim)
            fx = func(x)
            return x, fx

        candidates.sort(key=lambda t: t[0])
        x = candidates[0][1].copy()
        fx = candidates[0][0]
        sigma = 1.0
        calls_made = len(candidates)

        # Phase 2: Main (1+1)-ES with stagnation detection and restarts
        best_fx = fx
        best_x = x.copy()
        stagnation_count = 0
        max_stagnation = 50
        sigma_min = 1e-4
        sigma_max = 10.0

        for i in range(self.budget - calls_made):
            if time.time() > deadline - 5.0:  # leave time for polish
                break

            # Generate candidate
            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)
            calls_made += 1

            if fy < fx:
                x, fx = y, fy
                sigma = min(sigma * 1.2, sigma_max)
                stagnation_count = 0
                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()
            else:
                sigma = max(sigma * 0.9, sigma_min)
                stagnation_count += 1

            # Restart if stagnated
            if stagnation_count >= max_stagnation:
                # Try restarting from best with larger sigma
                x = best_x + sigma * rng.standard_normal(self.dim)
                x = np.clip(x, -5, 5)
                fx = func(x)
                calls_made += 1
                sigma = max(sigma * 2.0, 0.5)
                stagnation_count = 0
                if calls_made >= self.budget:
                    break

        # Phase 3: Final polish - use remaining budget for fine-tuning
        remaining = self.budget - calls_made
        if remaining > 0 and time.time() < deadline:
            sigma = max(sigma * 0.1, 1e-5)
            for j in range(remaining):
                if time.time() > deadline:
                    break
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                if fy < fx:
                    x, fx = y, fy
                    if fx < best_fx:
                        best_fx = fx
                        best_x = x.copy()
                    sigma = min(sigma * 1.1, 1e-3)
                else:
                    sigma = max(sigma * 0.95, 1e-6)

        return best_x, best_fx
# EVOLVE-BLOCK-END
