"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Adaptive Differential Evolution with local refinement for BBOB f17."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s/330s limit

        dim = self.dim
        budget = self.budget
        calls_used = 0
        best_x = np.zeros(dim)
        best_f = np.inf

        rng = np.random.default_rng(777)

        def try_update(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        def check_time():
            return calls_used < budget and time.time() < deadline - 5.0

        def check_time_early():
            return calls_used < budget and time.time() < deadline - 20.0

        # Phase 1: Build diverse initial population
        n_init = 2 * dim + 4
        pop = []
        pop.append(np.zeros(dim))
        for axis in range(dim):
            for sign in [-1, 1]:
                p = np.zeros(dim)
                p[axis] = sign * 3.5
                pop.append(p)
        while len(pop) < n_init:
            pop.append(rng.uniform(-4.5, 4.5, dim))

        # Evaluate initial population
        pop_evaluated = []
        for x in pop:
            if not check_time_early():
                break
            fx = func(x)
            calls_used += 1
            pop_evaluated.append((x, fx))
            try_update(x, fx)

        if len(pop_evaluated) >= 4:
            pop = [x for x, f in pop_evaluated]
            pop_f = [f for x, f in pop_evaluated]

            # Adaptive DE parameters
            F = 0.5
            CR = 0.5
            tau1 = 0.1
            tau2 = 0.1
            p_pbest = 0.1
            no_improve = 0
            max_no_improve = 10  # generations without improvement triggers restart

            while check_time():
                # Sort by fitness
                indices = np.argsort(pop_f)
                pop_sorted = [pop[i] for i in indices]
                pop_f_sorted = [pop_f[i] for i in indices]

                gen_best = pop_f_sorted[0]
                if gen_best < best_f - 1e-12:
                    no_improve = 0
                else:
                    no_improve += 1

                # If stagnating, inject diversity
                if no_improve >= max_no_improve:
                    # Replace worst half with random points around best
                    half = max(2, len(pop) // 2)
                    for i in range(half, len(pop)):
                        new_x = np.clip(best_x + rng.standard_normal(dim) * 1.5, -5, 5)
                        fx = func(new_x)
                        calls_used += 1
                        if not check_time():
                            break
                        pop[i] = new_x
                        pop_f[i] = fx
                        try_update(new_x, fx)
                    no_improve = 0
                    continue

                # Keep best half
                half = max(2, len(pop) // 2)
                new_pop = []
                new_pop_f = []

                for i in range(min(half, len(pop_sorted))):
                    new_pop.append(pop_sorted[i].copy())
                    new_pop_f.append(pop_f_sorted[i])

                # Generate new individuals using JADE-style mutation
                while len(new_pop) < len(pop) and check_time():
                    r_victim = rng.integers(len(pop))
                    victim = pop[r_victim].copy()
                    victim_f = pop_f[r_victim]

                    # pbest selection
                    if rng.random() < p_pbest:
                        subset_size = max(1, int(p_pbest * len(pop)))
                        subset_indices = rng.choice(len(pop), size=subset_size, replace=False)
                        pbest_idx = subset_indices[np.argmin([pop_f[i] for i in subset_indices])]
                        x_pbest = pop[pbest_idx]
                    else:
                        x_pbest = pop_sorted[0]

                    # Random distinct indices
                    r3 = rng.integers(len(pop))
                    r4 = rng.integers(len(pop))
                    attempts = 0
                    while (r3 == r4 or r3 == r_victim or r4 == r_victim) and attempts < 10:
                        r3 = rng.integers(len(pop))
                        r4 = rng.integers(len(pop))
                        attempts += 1
                    if r3 == r4 or r3 == r_victim or r4 == r_victim:
                        r3 = (r3 + 1) % len(pop)
                        r4 = (r4 + 2) % len(pop)
                        if r3 == r_victim:
                            r3 = (r3 + 1) % len(pop)
                        if r4 == r_victim:
                            r4 = (r4 + 1) % len(pop)
                        if r3 == r4:
                            r4 = (r4 + 1) % len(pop)

                    # Adaptive F
                    F1 = rng.uniform(0.1, 1.0)
                    F2 = rng.uniform(0.1, 1.0)

                    # JADE mutation
                    mutant = victim + F1 * (x_pbest - victim) + F2 * (pop[r3] - pop[r4])
                    mutant = np.clip(mutant, -5, 5)

                    f_mutant = func(mutant)
                    calls_used += 1

                    if f_mutant < victim_f:
                        new_pop.append(mutant)
                        new_pop_f.append(f_mutant)
                        try_update(mutant, f_mutant)
                    else:
                        # Keep a random individual from current population
                        keep_idx = rng.integers(len(pop))
                        new_pop.append(pop[keep_idx].copy())
                        new_pop_f.append(pop_f[keep_idx])

                pop = new_pop
                pop_f = new_pop_f

                # Slight CR adaptation
                CR = max(0.3, CR * 0.999)

        # Phase 2: Local refinement around best point
        if check_time():
            x = best_x.copy()
            fx = best_f
            sigma = 0.5
            no_improve = 0
            max_no_improve = 30

            while check_time() and no_improve < max_no_improve:
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_used += 1

                if fy < fx:
                    x, fx = y, fy
                    try_update(x, fx)
                    sigma *= 1.1
                    no_improve = 0
                else:
                    sigma *= 0.85
                    no_improve += 1

        # Phase 3: Coordinate-wise refinement
        if check_time():
            x = best_x.copy()
            fx = best_f
            step = 0.5
            for coord in range(dim):
                if not check_time():
                    break
                for _ in range(10):
                    if not check_time():
                        break
                    y = x.copy()
                    y[coord] += step
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    calls_used += 1
                    if fy < fx:
                        x, fx = y, fy
                        try_update(x, fx)
                    else:
                        y = x.copy()
                        y[coord] -= step
                        y = np.clip(y, -5, 5)
                        fy = func(y)
                        calls_used += 1
                        if fy < fx:
                            x, fx = y, fy
                            try_update(x, fx)
                        else:
                            break
                step *= 0.5

        return best_x, best_f
# EVOLVE-BLOCK-END
