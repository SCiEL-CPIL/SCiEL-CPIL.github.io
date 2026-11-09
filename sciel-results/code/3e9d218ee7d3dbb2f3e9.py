"""Deterministic multi-start with adaptive (1+1)-ES refinement and deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start deterministic initialization + adaptive (1+1)-ES refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    @staticmethod
    def _sobol_sequence(n, dim, seed=12345):
        """Generate n Sobol-like low-discrepancy points in [0,1]^dim."""
        # Use a simple Halton sequence as a low-discrepancy alternative
        # (Sobol from scipy might not be available in all environments, but Halton is easy)
        points = np.zeros((n, dim))
        for i in range(n):
            for j in range(dim):
                # Halton base = j + 2
                base = j + 2
                f = 1.0 / base
                result = 0.0
                i_val = i + 1
                while i_val > 0:
                    f /= base
                    result += f * (i_val % base)
                    i_val //= base
                points[i, j] = result
        return points

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 320.0  # 10s safety margin before 330s kill

        # Phase 1: Deterministic multi-start with Halton sequence
        # Use a good fraction of budget for broad coverage
        n_starts = min(100, budget // 5)
        halton = self._sobol_sequence(n_starts, dim)
        # Map to [-5, 5]^dim
        candidates = halton * 10.0 - 5.0

        best_x = np.zeros(dim)
        best_f = np.inf
        call_count = 0

        for i in range(n_starts):
            x = candidates[i]
            fx = func(x)
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            # Check deadline
            if time.time() > deadline:
                break

        # Phase 2: Adaptive (1+1)-ES refinement on the best candidate
        # Also periodically restart from other good candidates
        x = best_x.copy()
        fx = best_f
        sigma = 1.0
        sigma_min = 1e-6
        sigma_max = 10.0

        # Keep track of top candidates for potential restarts
        # Sort candidates by function value (we only have the best, but let's re-evaluate a few)
        # Actually, let's just do a local search from the best point, with occasional
        # jumps to explore nearby regions

        rng = np.random.default_rng(42)

        # Adaptive step size with 1/5 rule
        success_count = 0
        last_adjustment = 0

        while call_count < budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Generate trial point
            step = sigma * rng.standard_normal(dim)
            y = np.clip(x + step, -5, 5)
            fy = func(y)
            call_count += 1

            if fy < fx:
                x = y
                fx = fy
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                success_count += 1
            else:
                pass

            # Adjust sigma every 10 steps using 1/5 rule
            if call_count - last_adjustment >= 10:
                if success_count > 1:  # more than 1/5
                    sigma = min(sigma * 1.2, sigma_max)
                elif success_count < 2:  # less than 1/5
                    sigma = max(sigma * 0.8, sigma_min)
                success_count = 0
                last_adjustment = call_count

            # If sigma is very small, try a larger jump to escape local minima
            if sigma < 0.01:
                # Occasional large restart
                if rng.random() < 0.1:
                    jump = rng.standard_normal(dim) * 2.0
                    y = np.clip(x + jump, -5, 5)
                    fy = func(y)
                    call_count += 1
                    if fy < fx:
                        x = y
                        fx = fy
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()

            # Check deadline again after each call to be safe
            if time.time() > deadline:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
