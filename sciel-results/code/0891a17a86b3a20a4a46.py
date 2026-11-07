import numpy as np
import time


class Optimizer:
    """Hybrid optimizer: LHS global sampling + (1+1)-ES local refinement with final polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        max_time = 325.0  # Safety margin before 330s kill

        rng = np.random.default_rng(0)

        # Phase 1: Global Sampling with LHS
        # 100 samples provides good coverage for dim=5 without exhausting budget
        n_samples = 100
        samples = np.zeros((n_samples, dim))
        for d in range(dim):
            intervals = np.linspace(-5, 5, n_samples + 1)
            points = rng.uniform(intervals[:-1], intervals[1:])
            rng.shuffle(points)
            samples[:, d] = points

        f_samples = np.array([func(x) for x in samples])
        calls = n_samples

        if time.time() - start_time > max_time:
            idx = np.argmin(f_samples)
            return samples[idx], f_samples[idx]

        # Sort by fitness
        sorted_idx = np.argsort(f_samples)
        best_x = samples[sorted_idx[0]].copy()
        best_f = f_samples[sorted_idx[0]]

        # Phase 2: Local Refinement with (1+1)-ES on top candidates
        # Use top 5 candidates to balance exploration of different basins
        n_candidates = 5
        remaining_budget = budget - calls
        # Reserve a small safety margin, but use most of the budget
        safety_budget = int(remaining_budget * 0.05)
        alloc_budget = remaining_budget - safety_budget
        budget_per_candidate = max(10, alloc_budget // n_candidates)

        for k in range(n_candidates):
            if calls >= budget or time.time() - start_time > max_time:
                break

            x_start = samples[sorted_idx[k]].copy()
            f_start = f_samples[sorted_idx[k]]

            # (1+1)-ES refinement
            x = x_start.copy()
            fx = f_start
            sigma = 0.5  # Initial step size
            local_budget = min(budget_per_candidate, budget - calls)

            for _ in range(local_budget):
                if calls >= budget or time.time() - start_time > max_time:
                    break

                # Generate offspring
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls += 1

                # Update
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.1  # Increase step size on success
                else:
                    sigma *= 0.9  # Decrease step size on failure

            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 3: Final polish if budget remains
        # Use a smaller sigma for fine-tuning the best point found
        if calls < budget and time.time() - start_time < max_time:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            final_budget = budget - calls

            for _ in range(final_budget):
                if calls >= budget or time.time() - start_time > max_time:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls += 1
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.05
                else:
                    sigma *= 0.95

            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        return best_x, best_f
