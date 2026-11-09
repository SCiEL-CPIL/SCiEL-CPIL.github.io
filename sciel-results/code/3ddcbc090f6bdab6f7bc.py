import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # 15s margin before 300s limit

        n = self.dim
        budget = self.budget
        lb = -5.0
        ub = 5.0

        best_x = None
        best_f = np.inf
        eval_count = 0

        def safe_func(x):
            nonlocal best_x, best_f, eval_count
            if time.time() > deadline:
                raise TimeoutError
            if eval_count >= budget:
                raise BudgetExceeded
            f_val = func(np.asarray(x))
            eval_count += 1
            if f_val < best_f:
                best_f = f_val
                best_x = np.array(x, dtype=np.float64)
            return f_val

        rng = np.random.default_rng(7)

        # Phase 1: Initial population (LHS-inspired)
        # Use a stratified sample for good initial coverage
        n_init = 25
        init_xs = []
        init_fs = []
        # Stratified: divide each dim into 5 intervals, pick one per interval combo is too expensive
        # Instead use quasi-random (Halton-like via simple LHS)
        perm = rng.permutation(n_init)
        for i in range(n_init):
            x = np.empty(n)
            for d in range(n):
                x[d] = lb + (ub - lb) * (perm[i % n_init] + rng.random()) / n_init
            x = np.clip(x, lb, ub)
            init_xs.append(x)
            init_fs.append(safe_func(x))

        if best_x is None:
            return np.zeros(n), 0.0

        # L-SHADE parameters
        # Population size: start at 2n, shrink linearly to 4
        pop_size_start = max(2 * n, 10)  # 10 for n=5
        pop_size_min = 4
        F_min, F_max = 0.4, 0.9
        CR_min, CR_max = 0.2, 0.9

        # Success history archive
        H_F = []
        H_CR = []
        H_size = 50

        # Current population
        # Sort initial by fitness
        init_indices = np.argsort(init_fs)
        pop_x = [init_xs[i].copy() for i in init_indices[:pop_size_start]]
        pop_f = [init_fs[i] for i in init_indices[:pop_size_start]]

        # Track for adaptation
        prev_pop_size = pop_size_start
        generation = 0

        # Main loop
        while eval_count < budget:
            if time.time() > deadline:
                break

            # Compute current population size (linear shrink)
            total_generations_est = max(1, (budget - n_init) // pop_size_start)
            # Actually, let's use a fixed schedule based on remaining budget
            # Simpler: shrink every few generations
            # Let's use: pop_size = max(pop_size_min, pop_size_start - generation * shrink_rate)
            shrink_rate = max(1, (pop_size_start - pop_size_min) // max(1, (budget - n_init) // 5))
            cur_pop_size = max(pop_size_min, pop_size_start - generation * shrink_rate)

            # Sample F and CR from success history or random
            if len(H_F) >= 5:
                F = rng.cauchy(F_min + (F_max - F_min) * (np.random.rand() ** 0.5))
                F = np.clip(F, F_min, F_max)
                CR = rng.cauchy(CR_min + (CR_max - CR_min) * (np.random.rand() ** 0.5))
                CR = np.clip(CR, CR_min, CR_max)
            else:
                F = rng.uniform(F_min, F_max)
                CR = rng.uniform(CR_min, CR_max)

            # Generate trial population
            trial_xs = []
            trial_fs = []
            successful_F = []
            successful_CR = []

            for i in range(cur_pop_size):
                if eval_count >= budget or time.time() > deadline:
                    break

                # Select r1, r2, r3 different from i
                candidates = list(range(cur_pop_size))
                candidates.remove(i)
                r1, r2, r3 = rng.choice(candidates, size=3, replace=False)

                # DE/rand/1 mutation
                v = pop_x[r1] + F * (pop_x[r2] - pop_x[r3])
                # Clip to bounds
                v = np.clip(v, lb, ub)

                # Crossover
                j_rand = rng.integers(0, n)
                u = np.empty(n)
                for j in range(n):
                    if rng.random() <= CR or j == j_rand:
                        u[j] = v[j]
                    else:
                        u[j] = pop_x[i][j]

                f_u = safe_func(u)
                trial_xs.append(u)
                trial_fs.append(f_u)

                if f_u <= pop_f[i]:
                    successful_F.append(F)
                    successful_CR.append(CR)

            if not trial_xs:
                break

            # Update population (selection)
            new_pop_x = []
            new_pop_f = []
            for i in range(len(trial_xs)):
                if trial_fs[i] <= pop_f[i]:
                    new_pop_x.append(trial_xs[i])
                    new_pop_f.append(trial_fs[i])
                else:
                    new_pop_x.append(pop_x[i])
                    new_pop_f.append(pop_f[i])

            # Handle case where trial_xs might be shorter than pop_x (if we broke early)
            while len(new_pop_x) < cur_pop_size:
                idx = len(new_pop_x) % len(pop_x)
                new_pop_x.append(pop_x[idx])
                new_pop_f.append(pop_f[idx])

            pop_x = new_pop_x[:cur_pop_size]
            pop_f = new_pop_f[:cur_pop_size]

            # Update success history
            if successful_F:
                H_F.extend(successful_F)
                H_CR.extend(successful_CR)
                # Keep only last H_size
                if len(H_F) > H_size:
                    H_F = H_F[-H_size:]
                    H_CR = H_CR[-H_size:]

            generation += 1

            # If population shrank, we might have extra budget for more generations
            # This is handled naturally by the while loop

        return best_x, best_f
