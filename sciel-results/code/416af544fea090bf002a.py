"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """JADE-style adaptive Differential Evolution with local refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin before 330s kill

        dim = self.dim
        total_budget = self.budget
        calls_used = 0

        best_x = np.full(dim, np.nan)
        best_f = np.inf

        def try_update(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        def check_time():
            nonlocal calls_used
            return calls_used < total_budget and time.time() < deadline - 5.0

        def check_time_early():
            nonlocal calls_used
            return calls_used < total_budget and time.time() < deadline - 15.0

        rng = np.random.default_rng(42)

        # Initialize population with diverse points
        pop_size = max(10, 2 * dim)
        population = []

        # Add origin and axis-aligned points
        population.append(np.zeros(dim))
        for axis in range(dim):
            for sign in [-1, 1]:
                p = np.zeros(dim)
                p[axis] = sign * 3.5
                population.append(p)

        # Fill remaining with random points
        while len(population) < pop_size:
            population.append(rng.uniform(-4.5, 4.5, dim))

        # Evaluate initial population
        for i in range(pop_size):
            if not check_time_early():
                break
            x = np.clip(population[i], -5, 5)
            fx = func(x)
            calls_used += 1
            population[i] = x
            try_update(x, fx)

        # Sort population by fitness
        population.sort(key=lambda x: func(x) if False else None)
        # Re-evaluate to get fitness values properly
        pop_fitness = []
        for i in range(len(population)):
            if not check_time_early():
                break
            # We already have the function values from initial eval, but let's track them
            # Actually, we need to store them
            pass

        # Let's restructure to track fitness properly
        # Clear and redo
        calls_used = 0
        best_x = np.full(dim, np.nan)
        best_f = np.inf
        population = []

        # Initialize population
        for i in range(pop_size):
            if not check_time_early():
                break
            if i == 0:
                x = np.zeros(dim)
            elif i <= dim:
                x = np.zeros(dim)
                axis = i - 1
                x[axis] = 3.5
            elif i <= 2 * dim:
                x = np.zeros(dim)
                axis = i - dim - 1
                x[axis] = -3.5
            else:
                x = rng.uniform(-4.5, 4.5, dim)
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls_used += 1
            population.append((x, fx))
            try_update(x, fx)

        # Sort by fitness
        population.sort(key=lambda t: t[1])

        # JADE parameters
        F = 0.5
        CR = 0.1
        tau1 = 0.1
        tau2 = 0.1
        c = 0.1

        # Main DE loop
        while check_time():
            # Update F and CR using JADE adaptation
            # Sample from Cauchy and normal distributions
            F_new = np.clip(rng.cauchy(F, tau1) if rng.random() < 0.5 else rng.normal(F, tau1), 0.1, 0.9)
            CR_new = np.clip(rng.normal(CR, tau2), 0.0, 1.0)

            # Generate offspring
            offspring = []
            for i in range(pop_size):
                if not check_time():
                    break

                x_i = population[i][0]
                f_i = population[i][1]

                # Select r1, r2, r3 different from i
                candidates = [j for j in range(pop_size) if j != i]
                if len(candidates) < 3:
                    break
                r1, r2, r3 = rng.choice(candidates, size=3, replace=False)

                # DE/rand/1
                v = population[r1][0] + F_new * (population[r2][0] - population[r3][0])
                v = np.clip(v, -5, 5)

                # Binary crossover
                j_rand = rng.integers(0, dim)
                u = np.zeros(dim)
                for j in range(dim):
                    if rng.random() <= CR_new or j == j_rand:
                        u[j] = v[j]
                    else:
                        u[j] = x_i[j]

                f_u = func(u)
                calls_used += 1

                # Selection
                if f_u <= f_i:
                    offspring.append((u, f_u))
                else:
                    offspring.append((x_i, f_i))

                try_update(u if f_u <= f_i else x_i, min(f_u, f_i))

            # Update population
            population = offspring
            population.sort(key=lambda t: t[1])

            # Check for stagnation
            if len(population) > 0:
                # If best hasn't improved in several generations, try a restart
                pass

        # Final local refinement around best point
        if check_time() and not np.isnan(best_x).any():
            sigma = 0.5
            x = best_x.copy()
            fx = best_f
            for _ in range(min(50, total_budget - calls_used)):
                if not check_time():
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_used += 1
                if fy < fx:
                    x, fx = y, fy
                    try_update(x, fx)
                    sigma *= 0.95
                else:
                    sigma *= 0.9

        return best_x, best_f
# EVOLVE-BLOCK-END
