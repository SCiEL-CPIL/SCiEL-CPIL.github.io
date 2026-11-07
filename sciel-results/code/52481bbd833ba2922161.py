import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        start_time = time.time()
        deadline = start_time + 285.0  # 45s margin before 330s limit

        def time_ok():
            return time.time() < deadline

        # Initial point
        x_best = rng.uniform(-5, 5, self.dim)
        f_best = func(x_best)
        calls = 1

        x_ever = x_best.copy()
        f_ever = f_best

        # Adaptive step size
        sigma = 1.5
        lambda_pop = 6
        stall_counter = 0
        stall_threshold = 12  # 12 generations * 6 offspring = 72 calls without improvement

        # Track best few points for diversity
        archive = [(f_best, x_best.copy())]

        while calls + lambda_pop <= self.budget and time_ok():
            # Generate offspring
            noise = rng.standard_normal((lambda_pop, self.dim))
            offspring = x_best + sigma * noise
            offspring = np.clip(offspring, -5, 5)

            f_vals = np.array([func(x) for x in offspring])
            calls += lambda_pop

            # Update archive with best offspring
            for i in range(lambda_pop):
                if f_vals[i] < f_ever:
                    f_ever = f_vals[i]
                    x_ever = offspring[i].copy()
                archive.append((f_vals[i], offspring[i].copy()))
            archive.sort(key=lambda t: t[0])
            if len(archive) > 10:
                archive = archive[:10]

            best_idx = np.argmin(f_vals)
            best_val = f_vals[best_idx]

            if best_val < f_best:
                x_best = offspring[best_idx]
                f_best = best_val
                stall_counter = 0
                # Increase sigma on success
                sigma *= 1.12
            else:
                stall_counter += 1
                # Decrease sigma on failure
                sigma *= 0.82

            # Clamp sigma
            sigma = np.clip(sigma, 1e-7, 6.0)

            # Restart if stalled - pick from archive for better diversity
            if stall_counter > stall_threshold and calls < self.budget - lambda_pop - 2:
                # Pick a random point from archive (not the best, to encourage exploration)
                if len(archive) > 1:
                    idx = rng.integers(1, min(10, len(archive)))
                    x_best = archive[idx][1].copy()
                    f_best = archive[idx][0]
                else:
                    x_best = rng.uniform(-5, 5, self.dim)
                    f_best = func(x_best)
                    calls += 1
                stall_counter = 0
                sigma = 1.5
                if f_best < f_ever:
                    f_ever = f_best
                    x_ever = x_best.copy()

        # Use remaining budget with (1+1) steps
        while calls < self.budget and time_ok():
            # Generate a few candidates and pick the best
            n_cand = min(3, self.budget - calls)
            candidates = x_best + sigma * rng.standard_normal((n_cand, self.dim))
            candidates = np.clip(candidates, -5, 5)

            f_cands = np.array([func(x) for x in candidates])
            calls += n_cand

            best_i = np.argmin(f_cands)
            if f_cands[best_i] < f_best:
                x_best = candidates[best_i]
                f_best = f_cands[best_i]
                if f_best < f_ever:
                    f_ever = f_best
                    x_ever = x_best.copy()
                sigma *= 1.1
            else:
                sigma *= 0.85

            sigma = np.clip(sigma, 1e-7, 6.0)

        return x_ever, f_ever
