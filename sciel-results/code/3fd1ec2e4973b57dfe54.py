import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 290.0  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Start at box center for maximum early convergence
        x_best = np.zeros(dim)
        f_best = func(x_best)
        calls = 1

        step_size = 2.0  # initial step size for 5D box [-5, 5]
        gen = 0

        while calls < budget:
            if time.time() > deadline - 2.0:
                break

            gen += 1

            # --- Main (1+1)-ES step ---
            y = x_best + step_size * rng.standard_normal(dim)
            y = np.clip(y, -5, 5)
            fy = func(y)
            calls += 1

            if fy < f_best:
                x_best = y.copy()
                f_best = fy
                step_size *= 1.5
            else:
                step_size *= 0.87

            step_size = max(1e-4, min(step_size, 8.0))

            # --- Lightweight recombination every 5 generations ---
            # Evaluate 3 additional candidates around the best to build a
            # weighted recombination point. This helps escape flat ridge
            # directions without the overhead of a full population.
            if gen % 5 == 0 and calls + 3 < budget and time.time() < deadline - 5.0:
                candidates = []
                for i in range(3):
                    scale = [0.5, 1.0, 2.0][i]
                    cand = np.clip(x_best + step_size * scale * rng.standard_normal(dim), -5, 5)
                    fc = func(cand)
                    calls += 1
                    candidates.append((fc, cand))
                    if fc < f_best:
                        f_best = fc
                        x_best = cand.copy()

                # Weighted recombination: weight by inverse rank
                candidates.sort(key=lambda t: t[0])
                weights = np.array([0.5, 0.3, 0.2])
                rec = np.zeros(dim)
                for w, (_, c) in zip(weights, candidates):
                    rec += w * c
                # Blend recombination with current best to avoid jumping too far
                x_best = 0.6 * x_best + 0.4 * rec
                x_best = np.clip(x_best, -5, 5)

        # --- Final fine-tune phase: small adaptive steps ---
        remaining = budget - calls
        if remaining > 0 and time.time() < deadline - 1.0:
            local_sigma = max(step_size * 0.1, 0.01)
            for _ in range(remaining):
                if time.time() > deadline - 0.5:
                    break
                y = np.clip(x_best + local_sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls += 1
                if fy < f_best:
                    f_best = fy
                    x_best = y.copy()
                    local_sigma *= 1.2
                else:
                    local_sigma *= 0.9

        return x_best, f_best
