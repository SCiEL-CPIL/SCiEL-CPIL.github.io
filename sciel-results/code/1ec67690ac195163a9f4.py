"""Improved optimizer: (1+1)-ES with 1/5 success rule + coordinate-descent polish."""
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule, adaptive window, and coordinate-descent polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 310  # safety margin before 330s kill

        dim = self.dim
        budget = self.budget

        # Phase 1: Initial exploration with several random starts
        # Start at center and try a few random points
        candidates = [np.zeros(dim)]
        for _ in range(8):
            candidates.append(np.random.uniform(-4, 4, dim))

        best_x = candidates[0].copy()
        best_fx = func(best_x)
        total_calls = 1

        for c in candidates[1:]:
            if total_calls >= budget or time.time() >= deadline - 200:
                break
            fc = func(c)
            total_calls += 1
            if fc < best_fx:
                best_fx = fc
                best_x = c.copy()

        # Phase 2: (1+1)-ES with 1/5 success rule
        x = best_x.copy()
        fx = best_fx
        sigma = 3.0

        # 1/5 success rule parameters
        window_size = 10
        successes = 0
        total_in_window = 0
        es_calls = 0
        es_stagnation = 0

        # Track per-dimension step sizes for adaptive coordinate descent later
        dim_stages = np.ones(dim) * sigma

        es_budget = budget - total_calls - (dim * 10)  # reserve some for polish
        es_budget = max(es_budget, 100)

        while es_calls < es_budget and total_calls < budget and time.time() < deadline - 100:
            if time.time() >= deadline - 100:
                break

            # Generate offspring
            y = np.clip(x + sigma * np.random.randn(dim), -5, 5)
            fy = func(y)
            total_calls += 1
            es_calls += 1

            if fy < fx:
                x = y
                fx = fy
                if fy < best_fx:
                    best_fx = fy
                    best_x = x.copy()
                    es_stagnation = 0
                successes += 1
            else:
                es_stagnation += 1

            total_in_window += 1

            # Adapt sigma using 1/5 success rule
            if total_in_window >= window_size:
                success_rate = successes / total_in_window
                if success_rate > 0.2:
                    sigma *= 1.05
                elif success_rate < 0.2:
                    sigma *= 0.95
                # Reset window
                successes = 0
                total_in_window = 0

            sigma = max(1e-4, min(5.0, sigma))

            # If heavily stagnated, try a random restart perturbation
            if es_stagnation > 80 and time.time() < deadline - 120:
                # Small random perturbation to escape local minima
                x = np.clip(best_x + sigma * np.random.randn(dim) * 0.5, -5, 5)
                fx = func(x)
                total_calls += 1
                es_calls += 1
                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()
                es_stagnation = 0
                sigma = max(sigma * 1.2, 0.5)

        # Phase 3: Coordinate descent polish with adaptive step sizes
        if total_calls < budget and time.time() < deadline - 20:
            cd_x = best_x.copy()
            cd_fx = best_fx
            cd_steps = np.ones(dim) * max(sigma * 0.5, 0.1)

            # Multiple rounds of coordinate descent with shrinking steps
            max_cd_rounds = min(15, (budget - total_calls) // (2 * dim))

            for rnd in range(max_cd_rounds):
                if time.time() >= deadline - 20:
                    break

                for d in range(dim):
                    if total_calls >= budget or time.time() >= deadline - 20:
                        break

                    improved = False

                    # Try positive direction
                    trial = cd_x.copy()
                    trial[d] += cd_steps[d]
                    trial = np.clip(trial, -5, 5)
                    ftrial = func(trial)
                    total_calls += 1

                    if ftrial < cd_fx:
                        cd_x = trial
                        cd_fx = ftrial
                        if cd_fx < best_fx:
                            best_fx = cd_fx
                            best_x = cd_x.copy()
                        improved = True
                    else:
                        # Try negative direction
                        trial_neg = cd_x.copy()
                        trial_neg[d] -= cd_steps[d]
                        trial_neg = np.clip(trial_neg, -5, 5)
                        fneg = func(trial_neg)
                        total_calls += 1

                        if fneg < cd_fx:
                            cd_x = trial_neg
                            cd_fx = fneg
                            if cd_fx < best_fx:
                                best_fx = cd_fx
                                best_x = cd_x.copy()
                            improved = True

                    # Adapt step size for this dimension
                    if improved:
                        cd_steps[d] *= 1.1
                    else:
                        cd_steps[d] *= 0.7

                    cd_steps[d] = max(1e-4, min(2.0, cd_steps[d]))

                # Global step shrink
                cd_steps *= 0.85

            # Final fine-tuning with very small steps
            if total_calls < budget and time.time() < deadline - 10:
                final_steps = np.ones(dim) * max(cd_steps.min(), 0.01)
                for d in range(dim):
                    if total_calls >= budget or time.time() >= deadline - 10:
                        break
                    for direction in [1, -1]:
                        if total_calls >= budget or time.time() >= deadline - 10:
                            break
                        trial = cd_x.copy()
                        trial[d] += direction * final_steps[d]
                        trial = np.clip(trial, -5, 5)
                        ftrial = func(trial)
                        total_calls += 1
                        if ftrial < cd_fx:
                            cd_x = trial
                            cd_fx = ftrial
                            if cd_fx < best_fx:
                                best_fx = cd_fx
                                best_x = cd_x.copy()

        return best_x, best_fx
