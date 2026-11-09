"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Two-phase optimizer for BBOB f13 (sharp ridge):
    Phase 1: Deterministic multi-start (Halton + random) for broad coverage.
    Phase 2: Adaptive (1+1)-ES with 1/5 rule on the best starting point.
    Deadline guard ensures we return well before the 300s limit.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _halton(self, idx, base):
        """Generate the idx-th Halton sequence point in [0,1) for given base."""
        f = 1.0
        r = 0.0
        i = idx
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r

    def _halton_point(self, idx, dim):
        """Generate a Halton point in [0,1)^dim."""
        bases = self._prime_bases(dim)
        return np.array([self._halton(idx + 1, b) for b in bases])

    def _prime_bases(self, dim):
        """Return the first 'dim' prime numbers."""
        primes = []
        candidate = 2
        while len(primes) < dim:
            is_prime = True
            for p in primes:
                if p * p > candidate:
                    break
                if candidate % p == 0:
                    is_prime = False
                    break
            if is_prime:
                primes.append(candidate)
            candidate += 1
        return primes

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 20s safety margin before 300s kill

        dim = self.dim
        budget = self.budget

        # Phase 1: Deterministic multi-start with Halton sequence + some random points
        # Use about 20% of budget for exploration
        n_explore = max(10, min(50, budget // 5))
        
        # Generate Halton points mapped to [-5, 5]^dim
        halton_points = []
        for i in range(n_explore):
            p = self._halton_point(i, dim)
            mapped = -5.0 + 10.0 * p
            halton_points.append(mapped)

        # Add a few random points for diversity
        rng = np.random.default_rng(42)
        n_random = max(5, n_explore // 4)
        random_points = [rng.uniform(-5, 5, dim) for _ in range(n_random)]

        all_starts = halton_points + random_points

        # Evaluate all starting points
        best_x = None
        best_f = np.inf
        eval_count = 0

        for pt in all_starts:
            if eval_count >= budget:
                break
            if time.time() > deadline - 10.0:  # Stop exploring if close to deadline
                break
            f_val = func(pt)
            eval_count += 1
            if f_val < best_f:
                best_f = f_val
                best_x = pt.copy()

        # Phase 2: Adaptive (1+1)-ES from the best point
        if best_x is None:
            # Fallback if something went wrong
            x = np.zeros(dim)
            fx = func(x)
            eval_count += 1
        else:
            x = best_x
            fx = best_f

        # (1+1)-ES with 1/5 success rule
        sigma = 1.0  # Initial step size
        # 1/5 rule: success rate should be ~0.2
        # If success rate > 0.5, increase sigma; if < 0.2, decrease
        # Simple adaptation: use a running estimate

        success_window = []
        window_size = 10

        while eval_count < budget:
            if time.time() > deadline:
                break

            # Generate trial point
            y = x + sigma * rng.standard_normal(dim)
            y = np.clip(y, -5, 5)
            fy = func(y)
            eval_count += 1

            # Update best if improved
            if fy < fx:
                x = y
                fx = fy
                success_window.append(1)
            else:
                success_window.append(0)

            # Adapt sigma using 1/5 rule
            if len(success_window) >= window_size:
                success_rate = sum(success_window) / len(success_window)
                if success_rate > 0.5:
                    sigma *= 1.5
                elif success_rate < 0.2:
                    sigma *= 0.7
                else:
                    # Keep sigma roughly stable
                    sigma *= 0.95 + 0.1 * success_rate
                # Trim window
                success_window = success_window[-window_size:]

            # Ensure sigma stays in reasonable bounds
            sigma = np.clip(sigma, 1e-4, 10.0)

        return x, fx
# EVOLVE-BLOCK-END
