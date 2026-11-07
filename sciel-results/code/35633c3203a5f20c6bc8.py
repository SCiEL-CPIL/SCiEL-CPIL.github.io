"""Differential evolution optimizer for BBOB-style f6 (Attractive sector)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Differential evolution with small population for anisotropic landscapes."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0  # 20s safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_made = 0

        def eval_func(x):
            nonlocal calls_made, best_f, best_x
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def differential_evolution(x0, pop_size=5, F=0.8, CR=0.5, max_iter=100):
            nonlocal calls_made, best_f, best_x

            # Initialize population around x0
            population = []
            for i in range(pop_size):
                if i == 0:
                    pop_x = x0.copy()
                else:
                    pop_x = x0 + rng.uniform(-2.0, 2.0, dim)
                    pop_x = np.clip(pop_x, -5.0, 5.0)
                population.append(pop_x)

            # Evaluate initial population
            f_vals = [eval_func(x) for x in population]

            for iteration in range(max_iter):
                if calls_made >= budget - pop_size or time.time() > deadline:
                    return

                # Sort population by fitness
                sorted_idx = np.argsort(f_vals)
                population = [population[i] for i in sorted_idx]
                f_vals = [f_vals[i] for i in sorted_idx]

                # Track best
                if f_vals[0] < best_f:
                    best_f = f_vals[0]
                    best_x = population[0].copy()

                # Generate offspring using DE/best/1/bin strategy
                for i in range(pop_size):
                    if calls_made >= budget - 1 or time.time() > deadline:
                        return

                    # Select three distinct random indices different from i
                    while True:
                        a, b, c = rng.choice(pop_size, 3, replace=False)
                        if a != i and b != i and c != i:
                            break

                    # Mutation: best + F * (a - b)
                    mutant = population[0] + F * (population[a] - population[b])

                    # Crossover: binomial crossover
                    trial = population[i].copy()
                    j_rand = rng.integers(dim)
                    for j in range(dim):
                        if rng.random() < CR or j == j_rand:
                            trial[j] = mutant[j]
                    trial = np.clip(trial, -5.0, 5.0)

                    # Evaluate trial
                    trial_f = eval_func(trial)

                    # Selection
                    if trial_f <= f_vals[i]:
                        population[i] = trial
                        f_vals[i] = trial_f

                # Convergence check: if best hasn't improved in recent iterations
                if len(f_vals) >= 2 and abs(f_vals[0] - f_vals[1]) < 1e-10:
                    break

        # Initial exploration: evaluate diverse points
        initial_points = [
            np.zeros(dim),
            np.full(dim, 2.0),
            np.full(dim, -2.0),
            np.full(dim, 3.0),
            np.full(dim, -3.0),
            rng.uniform(-4, 4, dim),
        ]

        for pt in initial_points:
            if calls_made >= budget:
                break
            eval_func(pt)

        # Run DE from the best initial point with small population
        if best_x is not None and calls_made < budget - 10 and time.time() < deadline:
            differential_evolution(best_x.copy(), pop_size=5, F=0.8, CR=0.5, max_iter=80)

        # If we have budget left, try DE from another region
        if calls_made < budget - 10 and time.time() < deadline:
            alt_start = rng.uniform(-4, 4, dim)
            differential_evolution(alt_start, pop_size=5, F=1.0, CR=0.7, max_iter=50)

        # Final refinement with tighter parameters
        if best_x is not None and calls_made < budget - 5 and time.time() < deadline:
            differential_evolution(best_x.copy(), pop_size=5, F=0.5, CR=0.3, max_iter=30)

        # Ensure we have a valid result
        if best_x is None:
            x = np.zeros(dim)
            best_x = x
            best_f = func(x)
            calls_made += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
