"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start (1+1)-ES with adaptive step size for BBOB f24."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290.0  # safety margin before 300s limit

        # Track best solution found
        best_x = None
        best_f = np.inf

        # Count function calls
        call_count = 0

        def safe_call(x):
            nonlocal call_count
            if call_count >= budget:
                return None
            call_count += 1
            try:
                fx = func(x)
                return float(fx)
            except Exception:
                return None

        # Multi-start initialization targeting Lunacek bi-Rastrigin basins
        # The Lunacek function has two global minima: one near the origin
        # and one at a shifted location. We initialize around both.
        rng = np.random.default_rng(12345)

        # Generate initial points: mix of origin-centered and shifted
        n_starts = 8
        starts = []

        # 4 starts near origin (one basin)
        for _ in range(4):
            x0 = rng.uniform(-1.5, 1.5, dim)
            starts.append(x0)

        # 4 starts near the shifted basin (typical shift for f24 is around 4-5 in some dims)
        # The Lunacek bi-Rastrigin has a secondary minimum offset by s*(1+5/|y|) in each dim
        # where s is typically 4.982943678... We try shifts in various directions
        for i in range(4):
            # Try different shift patterns
            if i == 0:
                x0 = np.full(dim, 4.0) + rng.uniform(-0.5, 0.5, dim)
            elif i == 1:
                x0 = np.full(dim, -4.0) + rng.uniform(-0.5, 0.5, dim)
            elif i == 2:
                x0 = rng.uniform(3.0, 5.0, dim)
            else:
                x0 = rng.uniform(-5.0, -3.0, dim)
            starts.append(x0)

        # Also add a few random starts
        for _ in range(4):
            x0 = rng.uniform(-5, 5, dim)
            starts.append(x0)

        # Evaluate all starts and pick the best few to continue
        start_results = []
        for x0 in starts:
            fx = safe_call(x0)
            if fx is None:
                continue
            start_results.append((fx, x0))
            if fx < best_f:
                best_f = fx
                best_x = x0.copy()

        if not start_results:
            return best_x, best_f

        # Sort by fitness
        start_results.sort(key=lambda t: t[0])

        # Use top 3 starts as bases for local search
        n_local = min(3, len(start_results))

        for idx in range(n_local):
            if call_count >= budget or time.time() > deadline:
                break

            fx, x = start_results[idx]
            x = x.copy()

            # (1+1)-ES with window-based step size adaptation
            sigma = 1.0
            window_size = 10
            successes = 0
            sigma_min = 1e-6
            sigma_max = 10.0

            while call_count < budget and time.time() < deadline:
                # Generate trial point
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)

                fy = safe_call(y)
                if fy is None:
                    break

                # Update success count
                if fy < fx:
                    successes += 1
                    x = y.copy()
                    fx = fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    successes -= 1

                # Adapt step size based on recent window
                # Use a simple success rate over last few steps
                # If success rate is around 0.2-0.3, step size is well-calibrated
                # We'll use a simpler adaptive scheme

                # Check if we should adapt sigma based on last window
                # For simplicity, use a running average of successes
                # If too many successes, increase sigma; if too few, decrease
                if successes > window_size * 0.5:
                    sigma *= 1.2
                elif successes < window_size * 0.2:
                    sigma *= 0.8

                # Reset success counter periodically
                if call_count % (window_size * 2) == 0:
                    successes = 0

                # Clamp sigma
                sigma = np.clip(sigma, sigma_min, sigma_max)

                # Stagnation detection: if no improvement in last 50 calls, restart locally
                # (simplified: just continue)

            # After local search from this start, if we have budget left,
            # try a finer-grained polish around the best found so far
            if call_count < budget and time.time() < deadline and best_x is not None:
                # Fine polish with small sigma
                x_polish = best_x.copy()
                fx_polish = best_f
                sigma_polish = 0.1
                polish_calls = min(50, budget - call_count)

                for _ in range(polish_calls):
                    if time.time() > deadline:
                        break
                    y = x_polish + sigma_polish * rng.standard_normal(dim)
                    y = np.clip(y, -5, 5)
                    fy = safe_call(y)
                    if fy is None:
                        break
                    if fy < fx_polish:
                        x_polish = y.copy()
                        fx_polish = fy
                        if fx_polish < best_f:
                            best_f = fx_polish
                            best_x = x_polish.copy()
                    # Slowly decrease sigma
                    sigma_polish *= 0.95

        return best_x, best_f
# EVOLVE-BLOCK-END
