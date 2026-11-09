"""Combined optimizer: multi-start + adaptive (1+1)-ES with CMA-ES refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start adaptive (1+1)-ES with CMA-ES refinement for BBOB f21."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 300.0  # 30s safety margin before 330s kill

        rng = np.random.default_rng(12345)
        dim = self.dim
        budget = self.budget
        n = dim

        best_x = None
        best_f = np.inf
        call_count = 0

        def check_time():
            return time.time() > deadline - 5

        # Phase 1: Multi-start (1+1)-ES exploration
        n_restarts = 15
        # Use ~40% of budget for exploration
        exploration_budget = int(budget * 0.4)
        calls_per_restart = max(1, exploration_budget // n_restarts)

        # Create diverse starting points using a combination of grid and random
        start_points = []
        # Add some structured points
        for i in range(n):
            pt = np.zeros(dim)
            pt[i] = -4.0
            start_points.append(pt.copy())
            pt[i] = 4.0
            start_points.append(pt.copy())
        # Add center
        start_points.append(np.zeros(dim))
        # Add random points to fill up
        while len(start_points) < n_restarts:
            start_points.append(rng.uniform(-4, 4, dim))

        # Shuffle
        idx = rng.permutation(len(start_points))
        start_points = [start_points[i] for i in idx[:n_restarts]]

        for sp in start_points:
            if call_count >= budget - 1 or check_time():
                break

            x = np.clip(sp.copy(), -5, 5)
            fx = func(x)
            call_count += 1

            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            # Run (1+1)-ES with adaptive sigma using 1/5 success rule
            sigma = 1.0
            successes = 0
            attempts = 0

            for _ in range(calls_per_restart - 1):
                if call_count >= budget - 1 or check_time():
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                call_count += 1
                attempts += 1

                if fy < fx:
                    x, fx = y, fy
                    successes += 1
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    # Stagnation: try a larger jump occasionally
                    if sigma < 0.1 and attempts > 10:
                        sigma = 0.5
                        attempts = 0

                # Adaptive sigma: 1/5 success rule
                if attempts >= 5:
                    if successes / attempts > 0.5:
                        sigma *= 1.2
                    elif successes / attempts < 0.2:
                        sigma *= 0.8
                    successes = 0
                    attempts = 0

        # Phase 2: CMA-ES refinement on the best point
        if best_x is not None:
            refinement_budget = budget - call_count - 1
            if refinement_budget > 10 and not check_time():
                # Simple CMA-ES implementation
                x = best_x.copy()
                sigma = 0.5
                C = np.eye(dim)
                # Simple eigenvalue update
                lambda_ = 4 + int(3 * np.log(dim))  # population size
                c_sigma = (lambda_ + 2) / (dim + lambda_ + 5)
                c_c = (4 + dim / lambda_) / (dim + 4 + 2 * lambda_ / dim)
                c_1 = 2 / ((dim + 1.3) ** 2 + lambda_)
                c_mu = min(1 - c_1, 2 * ((lambda_ + 2) / (dim + 2) - 1))

                p_sigma = np.zeros(dim)
                p_c = np.zeros(dim)
                # Simple path for CMA-ES
                # For simplicity, we'll use a simplified version
                for _ in range(refinement_budget):
                    if call_count >= budget or check_time():
                        break

                    # Generate lambda_ samples
                    samples = []
                    for _ in range(lambda_):
                        # Simplified: just use standard normal
                        y = np.clip(x + sigma * rng.multivariate_normal(np.zeros(dim), C), -5, 5)
                        samples.append((y, func(y)))
                        call_count += 1
                        if call_count >= budget or check_time():
                            break

                    if not samples:
                        break

                    # Sort by fitness
                    samples.sort(key=lambda s: s[1])

                    # Update best
                    if samples[0][1] < best_f:
                        best_f = samples[0][1]
                        best_x = samples[0][0].copy()

                    # Update mean
                    weights = np.ones(lambda_)
                    weights[0] = 2.0  # Give more weight to best
                    w_sum = weights.sum()
                    weights = weights / w_sum

                    new_mean = np.zeros(dim)
                    for i, (y, _) in enumerate(samples):
                        new_mean += weights[i] * y
                    x = new_mean

                    # Update sigma
                    sigma *= (1 + c_sigma) ** 0.5

                    # Update covariance (simplified)
                    # For now, keep C as identity but adapt sigma
                    if sigma < 0.01:
                        sigma = 0.05
                    elif sigma > 5.0:
                        sigma = 2.0

        return best_x, best_f
# EVOLVE-BLOCK-END
