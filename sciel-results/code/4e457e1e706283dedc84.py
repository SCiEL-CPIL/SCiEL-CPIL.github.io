"""Multi-restart (1+1)-ES with adaptive sigma for Rastrigin (BBOB f3)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-restart (1+1)-ES optimizer for BBOB f3 (Rastrigin).
    Uses structured initial points and adaptive step size with restarts.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        budget = self.budget
        dim = self.dim
        calls = 0
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf

        def try_point(x):
            nonlocal calls, best_x, best_f
            if calls >= budget or time.time() > deadline:
                return False
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return True

        # Phase 1: Structured initial exploration
        # Evaluate at origin and points at integer coordinates
        init_points = [np.zeros(dim)]
        # Add points at +/-1, +/-2, +/-3, +/-4 in each dimension (structured)
        for val in [1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0]:
            init_points.append(np.full(dim, val))
        # Add some mixed points
        for i in range(dim):
            for j in range(i + 1, dim):
                pt = np.zeros(dim)
                pt[i] = 1.0
                pt[j] = 1.0
                init_points.append(pt)
                pt2 = np.zeros(dim)
                pt2[i] = -1.0
                pt2[j] = -1.0
                init_points.append(pt2)

        # Limit initial points to ~30
        n_init = min(30, len(init_points))
        for i in range(n_init):
            if not try_point(init_points[i]):
                break

        # Phase 2: Multi-restart (1+1)-ES
        # Use the best point found so far as the first starting point
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls += 1

        # Generate restart points: mix of random and structured
        restart_points = [best_x.copy()]
        # Add random points
        for _ in range(20):
            restart_points.append(rng.uniform(-4, 4, dim))
        # Add structured points near best
        for scale in [0.5, 1.0, 2.0]:
            for _ in range(5):
                restart_points.append(best_x + scale * rng.standard_normal(dim))

        # Shuffle restart points
        rng.shuffle(restart_points)

        # Run (1+1)-ES from each restart point
        for start_pt in restart_points:
            if calls >= budget or time.time() > deadline:
                break

            x = np.clip(start_pt.copy(), -5, 5)
            f = func(x)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()

            sigma = 0.5
            no_improve = 0
            max_no_improve = 30  # Restart after 30 failed steps
            local_calls = 0
            max_local_calls = 40  # Limit calls per restart

            while local_calls < max_local_calls and calls < budget and time.time() < deadline:
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls += 1
                local_calls += 1

                if fy < f:
                    x, f = y, fy
                    no_improve = 0
                    sigma *= 1.05
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
                else:
                    sigma *= 0.95
                    no_improve += 1
                    if no_improve > max_no_improve:
                        break

        # Phase 3: Final polishing with very small steps
        if calls < budget and time.time() < deadline:
            x = best_x.copy()
            f = best_f
            sigma = 0.01
            no_improve = 0
            max_no_improve = 20
            remaining = budget - calls

            for _ in range(remaining):
                if time.time() > deadline:
                    break
                if calls >= budget:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls += 1
                if fy < f:
                    x, f = y, fy
                    no_improve = 0
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
                else:
                    sigma *= 0.9
                    no_improve += 1
                    if no_improve > max_no_improve:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
