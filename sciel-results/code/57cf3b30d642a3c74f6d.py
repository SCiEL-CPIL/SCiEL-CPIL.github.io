import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_used = 0

        def track_call(x, f):
            nonlocal calls_used, best_x, best_f
            calls_used += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()

        def time_ok():
            return time.time() < deadline and calls_used < budget

        rng = np.random.default_rng(42)

        # Phase 1: Initial random exploration
        n_init = min(10, budget)
        for _ in range(n_init):
            if not time_ok():
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            track_call(x, fx)

        if best_x is None:
            return np.zeros(dim), np.inf

        # Phase 2: Differential Evolution with adaptive restarts
        pop_size = 12  # population size
        F = 0.8  # differential weight
        CR = 0.5  # crossover rate
        max_generations_no_improve = 15  # restart threshold
        generations_no_improve = 0
        best_f_history = []

        # Initialize population around best point
        pop = []
        for i in range(pop_size):
            if i == 0:
                x = best_x.copy()
            else:
                x = best_x + rng.normal(0, 0.5, dim)
                x = np.clip(x, -5, 5)
            fx = func(x)
            track_call(x, fx)
            pop.append((x.copy(), fx))

        # Sort population
        pop.sort(key=lambda t: t[1])

        while time_ok():
            if calls_used >= budget:
                break

            # Check stagnation
            if best_f_history and best_f <= best_f_history[-1] + 1e-10:
                generations_no_improve += 1
            else:
                generations_no_improve = 0
            best_f_history.append(best_f)

            # If stagnated, restart with new population centered on best
            if generations_no_improve >= max_generations_no_improve:
                pop = []
                for i in range(pop_size):
                    if i == 0:
                        x = best_x.copy()
                    else:
                        # Mix of local and global exploration
                        if rng.random() < 0.5:
                            x = best_x + rng.normal(0, 0.3, dim)
                        else:
                            x = rng.uniform(-5, 5, dim)
                        x = np.clip(x, -5, 5)
                    fx = func(x)
                    track_call(x, fx)
                    pop.append((x.copy(), fx))
                pop.sort(key=lambda t: t[1])
                generations_no_improve = 0
                best_f_history = []
                continue

            # Create offspring
            new_pop = []
            for i in range(pop_size):
                if calls_used >= budget or not time_ok():
                    break

                # Select three distinct random individuals
                r1, r2, r3 = rng.choice(pop_size, 3, replace=False)
                while r1 == i or r2 == i or r3 == i:
                    r1, r2, r3 = rng.choice(pop_size, 3, replace=False)

                # DE/rand/1 mutation
                v = pop[r1][0] + F * (pop[r2][0] - pop[r3][0])
                v = np.clip(v, -5, 5)

                # Crossover
                j_rand = rng.integers(dim)
                trial = v.copy()
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                    else:
                        trial[j] = pop[i][0][j]
                trial = np.clip(trial, -5, 5)

                # Evaluate
                ft = func(trial)
                track_call(trial, ft)

                # Selection
                if ft <= pop[i][1]:
                    new_pop.append((trial.copy(), ft))
                else:
                    new_pop.append(pop[i])

            if len(new_pop) < pop_size:
                # Pad with existing population members
                for i in range(len(new_pop), pop_size):
                    new_pop.append(pop[i])

            pop = new_pop
            pop.sort(key=lambda t: t[1])

            # Update global best
            if pop[0][1] < best_f:
                best_f = pop[0][1]
                best_x = pop[0][0].copy()

        # Phase 3: Local polish with adaptive step size
        remaining = budget - calls_used
        if remaining > 0 and best_x is not None:
            polish_sigma = 0.05
            min_sigma = 1e-6
            for _ in range(min(remaining, 80)):
                if not time_ok():
                    break
                z = rng.standard_normal(dim)
                y = best_x + polish_sigma * z
                y = np.clip(y, -5, 5)
                fy = func(y)
                track_call(y, fy)
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                    polish_sigma = max(polish_sigma * 0.95, min_sigma)
                else:
                    polish_sigma = max(polish_sigma * 0.8, min_sigma)

        return best_x, best_f
