"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid optimizer: stratified multi-start + adaptive (1+1)-ES + coordinate polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(42)

        best_x = None
        best_f = float('inf')
        calls_made = 0

        def clip(x):
            return np.clip(x, -5.0, 5.0)

        def evaluate(x):
            nonlocal calls_made, best_f, best_x
            x = clip(np.asarray(x, dtype=float))
            fx = float(func(x))
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def remaining_time():
            return deadline - time.time()

        # Phase 1: Stratified diverse initial evaluation
        initial_points = []

        # Center
        initial_points.append(np.zeros(dim))

        # Stratified sampling: divide each dimension into 3 bins
        n_strat = 12
        for _ in range(n_strat):
            bins = rng.integers(0, 3, dim)
            x = np.zeros(dim)
            for d in range(dim):
                bin_start = -5.0 + bins[d] * (10.0 / 3.0)
                bin_end = bin_start + (10.0 / 3.0)
                x[d] = rng.uniform(bin_start, bin_end)
            initial_points.append(x)

        # A few pure random points
        for _ in range(5):
            initial_points.append(rng.uniform(-4.0, 4.0, dim))

        # Evaluate all initial points
        evaluated = []
        for x0 in initial_points:
            if calls_made >= budget or remaining_time() < 10.0:
                break
            x0 = clip(x0)
            fx = evaluate(x0)
            evaluated.append((fx, x0.copy()))

        if not evaluated:
            x0 = np.zeros(dim)
            evaluate(x0)
            evaluated.append((best_f, best_x.copy()))

        # Sort by fitness
        evaluated.sort(key=lambda t: t[0])

        # Phase 2: Local refinement using Adaptive (1+1)-ES on top candidates
        top_k = min(5, len(evaluated))
        reserve_budget = 30
        available_for_refinement = budget - calls_made - reserve_budget

        if available_for_refinement > 0:
            calls_per_candidate = max(15, available_for_refinement // top_k)

            for k in range(top_k):
                if calls_made >= budget or remaining_time() < 5.0:
                    break

                fx_init, x_init = evaluated[k]

                # Adaptive (1+1)-ES
                x = x_init.copy()
                fx = fx_init
                sigma = 0.8
                min_sigma = 1e-6
                max_sigma = 5.0

                # Success rate tracking with window
                window_size = 10
                successes = []

                for _ in range(calls_per_candidate):
                    if calls_made >= budget or remaining_time() < 2.0:
                        break

                    # Generate trial
                    y = x + sigma * rng.standard_normal(dim)
                    y = clip(y)
                    fy = evaluate(y)

                    # Update success window
                    if fy < fx:
                        successes.append(1)
                        x = y
                        fx = fy
                    else:
                        successes.append(0)

                    # Keep window size constant
                    if len(successes) > window_size:
                        successes.pop(0)

                    # Adapt sigma based on success rate in window
                    if len(successes) == window_size:
                        sr = sum(successes) / window_size
                        target_sr = 0.2
                        if sr > target_sr + 0.1:
                            sigma *= 1.05
                        elif sr < target_sr - 0.1:
                            sigma *= 0.95

                    sigma = np.clip(sigma, min_sigma, max_sigma)

                    if sigma < min_sigma * 10:
                        break

                # Update global best
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

        # Phase 3: Coordinate-wise polish on the global best
        if calls_made < budget and remaining_time() > 5.0 and best_x is not None:
            x = best_x.copy()
            fx = best_f

            # Try coordinate-wise steps with decreasing magnitudes
            step_sizes = [0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002]

            for step in step_sizes:
                if calls_made >= budget or remaining_time() < 2.0:
                    break

                for d in range(dim):
                    if calls_made >= budget or remaining_time() < 1.0:
                        break

                    # Try positive and negative steps along each coordinate
                    for direction in [1.0, -1.0]:
                        if calls_made >= budget or remaining_time() < 1.0:
                            break

                        y = x.copy()
                        y[d] += direction * step
                        y = clip(y)
                        fy = evaluate(y)

                        if fy < fx:
                            x = y
                            fx = fy

            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 4: Final fine-tuning with very small random steps
        if calls_made < budget and remaining_time() > 3.0 and best_x is not None:
            x = best_x.copy()
            fx = best_f
            sigma = 0.02

            while calls_made < budget and remaining_time() > 1.0:
                y = clip(x + sigma * rng.standard_normal(dim))
                fy = evaluate(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.01
                else:
                    sigma *= 0.98
                    if sigma < 1e-8:
                        break

            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
