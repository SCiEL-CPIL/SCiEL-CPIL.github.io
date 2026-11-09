"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Differential Evolution with adaptive parameters for rotated Rastrigin in 5D."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        start_time = time.time()
        deadline = start_time + 320  # Safety margin before 330s kill

        best_x = None
        best_f = np.inf
        call_count = 0

        def safe_func(x):
            nonlocal call_count
            if call_count >= budget:
                return np.inf
            call_count += 1
            try:
                val = func(np.asarray(x, dtype=float).reshape(dim))
                if np.isfinite(val):
                    return float(val)
                return 1e10
            except Exception:
                return 1e10

        def time_ok():
            return call_count < budget and time.time() < deadline

        # Initialize population
        pop_size = 20
        pop = rng.uniform(-5, 5, size=(pop_size, dim))
        fvals = np.array([safe_func(x) for x in pop])
        
        for i in range(pop_size):
            if fvals[i] < best_f:
                best_f = fvals[i]
                best_x = pop[i].copy()

        # DE parameters
        F = 0.8  # Mutation scale factor
        CR = 0.9  # Crossover rate

        # Adaptive strategy: reduce F and CR as we converge
        generation = 0
        no_improve = 0
        prev_best = best_f

        while time_ok():
            generation += 1
            # Adaptive parameters
            if no_improve > 10:
                F = max(0.3, F * 0.95)
                CR = min(0.95, CR * 1.02)
            elif best_f < prev_best:
                no_improve = 0
                F = min(0.8, F * 1.05)
                CR = max(0.5, CR * 0.98)
            else:
                no_improve += 1

            prev_best = best_f

            # Generate offspring for each member
            for i in range(pop_size):
                if not time_ok():
                    break

                # DE/rand/1 mutation
                a = rng.choice([j for j in range(pop_size) if j != i])
                b = rng.choice([j for j in range(pop_size) if j != i and j != a])
                c = rng.choice([j for j in range(pop_size) if j != i and j != a and j != b])

                v = pop[a] + F * (pop[b] - pop[c])
                v = np.clip(v, -5, 5)

                # Binomial crossover
                trial = pop[i].copy()
                j_rand = rng.integers(dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]

                # Evaluate trial
                f_trial = safe_func(trial)

                # Selection
                if f_trial <= fvals[i]:
                    pop[i] = trial
                    fvals[i] = f_trial
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
