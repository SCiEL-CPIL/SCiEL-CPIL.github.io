"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES-like optimizer with adaptive per-dim steps and local polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 280.0  # safety margin before 300s kill
        budget = self.budget
        dim = self.dim
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_x = x.copy()
                best_f = f

        def time_ok():
            return time.time() < deadline

        # Phase 1: Broad multi-start to find a good basin
        # Use 12 diverse starting points
        n_starts = 12
        calls_made = 0

        # Generate diverse starting points: corners, center, random
        starts = []
        # Center
        starts.append(np.zeros(dim))
        # Corners of the box
        for i in range(2 ** min(dim, 5)):
            corner = np.array([(i >> j) & 1 for j in range(dim)], dtype=float)
            corner = corner * 10 - 5  # map {0,1} -> {-5,5}
            starts.append(corner)
        # Random points
        while len(starts) < n_starts:
            starts.append(rng.uniform(-5, 5, dim))

        # Evaluate all starts
        for s in starts:
            if calls_made >= budget or not time_ok():
                break
            s_clipped = np.clip(s, -5, 5)
            fs = func(s_clipped)
            calls_made += 1
            update_best(s_clipped, fs)

        # Phase 2: Local CMA-ES-like refinement from the best start
        if best_x is not None and calls_made < budget and time_ok():
            x = best_x.copy()
            fx = best_f
            # Per-dimension step sizes, initialized based on problem scale
            sigma = np.full(dim, 1.0)
            # Covariance diagonal (per-dim variances)
            C = np.eye(dim)

            # Success counter for step size adaptation
            successes = 0
            n_success_window = 5
            success_count = 0

            # Track recent successes for 1/5 rule
            success_history = []

            remaining = budget - calls_made
            for it in range(remaining):
                if not time_ok():
                    break
                if calls_made >= budget:
                    break

                # Generate perturbation
                # Use diagonal CMA-ES: sample from N(0, diag(sigma^2))
                z = rng.standard_normal(dim)
                step = sigma * z
                y = x + step
                y = np.clip(y, -5, 5)

                fy = func(y)
                calls_made += 1

                if fy < fx:
                    x = y
                    fx = fy
                    success_count += 1
                    success_history.append(1)
                    # Increase step sizes
                    sigma *= 1.2
                else:
                    success_history.append(0)
                    # Decrease step sizes
                    sigma *= 0.9

                # Keep sigma in reasonable range
                sigma = np.clip(sigma, 0.01, 5.0)

                # 1/5 success rule: adapt global scale
                if len(success_history) >= n_success_window:
                    success_rate = sum(success_history[-n_success_window:]) / n_success_window
                    if success_rate > 0.5:
                        sigma *= 1.1
                    elif success_rate < 0.2:
                        sigma *= 0.85
                    success_history = success_history[-(n_success_window - 1):]

                update_best(x, fx)

                # If we've been stuck for a while, try a larger jump
                if it > 50 and it % 100 == 0 and success_count < 5:
                    # Random restart from current best with larger perturbation
                    z_big = rng.standard_normal(dim) * 3.0
                    x_try = x + z_big
                    x_try = np.clip(x_try, -5, 5)
                    f_try = func(x_try)
                    calls_made += 1
                    if f_try < fx:
                        x = x_try
                        fx = f_try
                        update_best(x, fx)

        # Phase 3: Fine polishing with very small steps if budget remains
        if best_x is not None and calls_made < budget and time_ok():
            x = best_x.copy()
            fx = best_f
            # Very small per-dim steps for polishing
            for _ in range(min(50, budget - calls_made)):
                if not time_ok() or calls_made >= budget:
                    break
                for d in range(dim):
                    if not time_ok() or calls_made >= budget:
                        break
                    for direction in [-1, 1]:
                        if calls_made >= budget:
                            break
                        y = x.copy()
                        y[d] += direction * 0.1
                        y = np.clip(y, -5, 5)
                        fy = func(y)
                        calls_made += 1
                        if fy < fx:
                            x = y
                            fx = fy
                            update_best(x, fx)

        return best_x, best_f
# EVOLVE-BLOCK-END
