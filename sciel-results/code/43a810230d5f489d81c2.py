"""Portfolio search optimizer for BBOB f20 (Schwefel) with explicit deadline guard."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """
    Multi-start (1+1)-ES with 1/5 success rule and final polish phase.
    Designed for BBOB f20 (Schwefel) on [-5,5]^5 with budget 500.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # safety margin before 330s kill
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf

        def update_best(x, fx):
            nonlocal best_x, best_f
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Multi-start with diverse initial points
        n_starts = 8
        starts = []
        starts.append(np.zeros(self.dim))
        for _ in range(n_starts - 1):
            starts.append(rng.uniform(-4.5, 4.5, self.dim))

        # Reserve last 40 evaluations for a final polish around best_x
        polish_budget = min(40, self.budget // 5)
        search_budget = self.budget - polish_budget

        # Allocate search budget across starts
        allocations = [search_budget // n_starts] * n_starts
        allocations[0] += search_budget % n_starts  # first start gets remainder

        call_count = 0

        for i, x0 in enumerate(starts):
            if call_count >= search_budget:
                break
            if time.time() > deadline:
                break

            alloc = min(allocations[i], search_budget - call_count)
            x = x0.copy()
            fx = func(x)
            call_count += 1
            update_best(x, fx)

            # 1/5 success rule parameters
            sigma = 1.5
            c_sigma = 1.0 + 0.35 * (1 - 1.0 / (self.dim + 1.7))  # learning rate
            c_cs = 1.0 + 0.35 * (1 - 1.0 / (self.dim + 1.7))
            chi_n = np.sqrt(self.dim) * (1 - 1.0 / (4 * self.dim) + 1.0 / (21 * self.dim**2))
            cs = (1.5 + 0.5 / self.dim) / (self.dim + 1.5 + 2.0 / self.dim)  # path length smoothing
            c_c = 1.0 + 2.0 * max(0, np.sqrt((cs + 1) / (self.dim + 1)) - 1)  # covariance adaptation
            c1 = 2 / ((self.dim + 1.3)**2 + 10)  # rank-one update
            cmu = min(c1 * 3 / 2, 0.5)  # rank-mu update (simplified)
            ps = np.zeros(self.dim)  # success path
            pc = np.zeros(self.dim)  # covariance adaptation path
            C = np.eye(self.dim)  # covariance matrix
            inv_sqrt_C = np.eye(self.dim)  # inverse square root of C
            dcor = 1 + 2 * max(0, np.sqrt((cs + 1) / (self.dim + 1)) - 1) + c1
            alpha_sigma = (1 + 2 * max(0, np.sqrt((cs + 1) / (self.dim + 1)) - 1)) * c_sigma
            kappa_sigma = 0.3 * (self.dim + 1)
            chi_n_inv = 1.0 / chi_n

            # Simplified: use only sigma adaptation with 1/5 rule
            # Track success rate over window of 10
            success_window = []
            window_size = 10

            for step in range(alloc - 1):
                if call_count >= search_budget:
                    break
                if time.time() > deadline:
                    break

                # Generate trial point
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                call_count += 1

                success = fy < fx
                if success:
                    x = y
                    fx = fy
                    update_best(x, fx)

                success_window.append(1 if success else 0)
                if len(success_window) > window_size:
                    success_window.pop(0)

                # Adapt sigma based on success rate (target ~1/5 or ~0.2)
                if len(success_window) >= window_size:
                    sr = np.mean(success_window)
                    # If success rate too high, increase sigma; too low, decrease
                    if sr > 0.3:
                        sigma *= 1.15
                    elif sr < 0.1:
                        sigma *= 0.85
                    else:
                        sigma *= 1.0  # keep same

                # If stuck for a while, try a larger jump
                if len(success_window) >= window_size and np.mean(success_window) < 0.05:
                    y = np.clip(x + sigma * 3 * rng.standard_normal(self.dim), -5, 5)
                    fy = func(y)
                    call_count += 1
                    if fy < fx:
                        x = y
                        fx = fy
                        update_best(x, fx)
                        success_window = [1]
                    else:
                        success_window = [0]
                        sigma *= 0.5

                sigma = np.clip(sigma, 0.005, 5.0)

        # Final polish phase: tight local search around best_x
        if best_x is not None and call_count < self.budget:
            remaining = self.budget - call_count
            x = best_x.copy()
            fx = best_f
            sigma = 0.1  # small step for polishing

            for step in range(remaining):
                if time.time() > deadline:
                    break

                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                call_count += 1

                if fy < fx:
                    x = y
                    fx = fy
                    update_best(x, fx)
                    sigma *= 1.05  # slight increase
                else:
                    sigma *= 0.9  # decrease

                sigma = np.clip(sigma, 0.001, 1.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
