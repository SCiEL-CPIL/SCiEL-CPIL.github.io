"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Portfolio search for BBOB f20 (Schwefel) on [-5,5]^5.
    
    Strategy:
      1. Multiple (1+1)-ES restarts with diverse initializations and step sizes.
      2. Adaptive step-size control based on success rate (1/5 rule variant).
      3. Final local refinement using a finer-grained search around the best point.
      4. Wall-clock deadline guard to stay within the 330s limit with safety margin.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # safety margin before 330s kill

        # Call counter to track budget usage
        call_count = [0]

        def make_call(x):
            """Wrap func call with budget tracking."""
            if call_count[0] >= budget:
                return None
            call_count[0] += 1
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            return func(x)

        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 1: Multiple (1+1)-ES restarts with diverse parameters
        # We allocate budget across restarts, with more budget for early restarts
        n_restarts = 8
        # Budget allocation: decreasing allocation per restart
        # First restart gets more budget, later ones get less
        allocations = []
        remaining = budget - 50  # reserve 50 for final refinement
        for i in range(n_restarts):
            if i == n_restarts - 1:
                alloc = remaining
            else:
                frac = 0.5 ** i * 0.25  # geometric decay
                alloc = max(10, int(remaining * frac))
                remaining -= alloc
            allocations.append(alloc)
        # Adjust last to use all remaining
        allocations[-1] = max(1, remaining)

        # Different initialization strategies for diversity
        init_strategies = [
            lambda: np.zeros(dim),
            lambda: np.full(dim, 4.20968746),  # known Schwefel minimum vicinity
            lambda: rng.uniform(-5, 5, dim),
            lambda: rng.uniform(-5, 5, dim),
            lambda: rng.uniform(-5, 5, dim),
            lambda: np.full(dim, -4.20968746),
            lambda: rng.uniform(-5, 5, dim),
            lambda: np.zeros(dim) + 0.1 * rng.standard_normal(dim),
        ]

        # Different initial step sizes for diversity
        initial_sigmas = [2.0, 1.0, 3.0, 1.5, 2.5, 1.0, 3.0, 0.5]

        for r in range(n_restarts):
            if call_count[0] >= budget - 50:
                break
            if time.time() > deadline - 30:
                break

            alloc = allocations[r]
            if alloc < 5:
                break

            x = init_strategies[r % len(init_strategies)]()
            fx = make_call(x)
            if fx is None:
                break
            update_best(x, fx)

            sigma = initial_sigmas[r % len(initial_sigmas)]
            success_count = 0
            total_count = 0

            for step in range(alloc):
                if call_count[0] >= budget - 50:
                    break
                if time.time() > deadline - 30:
                    break

                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5.0, 5.0)
                fy = make_call(y)
                if fy is None:
                    break

                total_count += 1
                if fy < fx:
                    x = y
                    fx = fy
                    success_count += 1
                    update_best(x, fx)
                    # Increase step size on success
                    sigma *= 1.3
                else:
                    # Decrease step size on failure
                    sigma *= 0.85

                # Adaptive step size adjustment based on recent success rate
                if total_count >= 10:
                    recent_success_rate = success_count / total_count
                    if recent_success_rate > 0.3:
                        sigma *= 1.1
                    elif recent_success_rate < 0.1:
                        sigma *= 0.9
                    # Reset counters to track recent performance
                    success_count = 0
                    total_count = 0

                # Keep sigma within reasonable bounds
                sigma = np.clip(sigma, 0.01, 10.0)

        # Phase 2: Final local refinement around the best point
        if best_x is not None and call_count[0] < budget - 5:
            if time.time() < deadline - 10:
                x = best_x.copy()
                fx = best_f
                sigma = 0.1
                for step in range(budget - call_count[0] - 5):
                    if call_count[0] >= budget - 5:
                        break
                    if time.time() > deadline - 5:
                        break

                    y = x + sigma * rng.standard_normal(dim)
                    y = np.clip(y, -5.0, 5.0)
                    fy = make_call(y)
                    if fy is None:
                        break

                    if fy < fx:
                        x = y
                        fx = fy
                        update_best(x, fx)
                        sigma *= 1.1
                    else:
                        sigma *= 0.8

                    sigma = np.clip(sigma, 0.001, 1.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
