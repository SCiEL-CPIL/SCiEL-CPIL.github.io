"""Optimizer for BBOB-style f3 (Rastrigin) on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Differential Evolution with Nelder-Mead local refinement for Rastrigin."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 330s kill

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = np.zeros(dim)
        best_f = np.inf
        call_count = 0

        def call_func(x):
            nonlocal call_count, best_x, best_f
            if call_count >= budget:
                return None
            val = func(x)
            call_count += 1
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        # Phase 1: Differential Evolution for global exploration
        # DE is good at escaping Rastrigin's many local minima
        pop_size = 20
        F = 0.8  # mutation scale
        CR = 0.9  # crossover rate

        # Initialize population
        population = []
        pop_f = []
        for i in range(pop_size):
            if call_count >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            fx = call_func(x)
            if fx is None:
                break
            population.append(x)
            pop_f.append(fx)

        # DE parameters
        max_generations = 40
        stagnation_counter = 0
        max_stagnation = 15

        for gen in range(max_generations):
            if call_count >= budget or time.time() > deadline:
                break

            # Sort population by fitness
            sorted_idx = np.argsort(pop_f)
            population = [population[i] for i in sorted_idx]
            pop_f = [pop_f[i] for i in sorted_idx]

            # Check for stagnation
            if pop_f[0] < best_f:
                best_f = pop_f[0]
                best_x = population[0].copy()
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            if stagnation_counter > max_stagnation:
                # Reset stagnation and perturb worst individuals
                for i in range(pop_size // 2, pop_size):
                    if call_count >= budget or time.time() > deadline:
                        break
                    population[i] = rng.uniform(-5, 5, dim)
                    fx = call_func(population[i])
                    if fx is None:
                        break
                    pop_f[i] = fx
                stagnation_counter = 0
                continue

            # Generate offspring
            new_population = list(population)
            new_pop_f = list(pop_f)

            for i in range(pop_size):
                if call_count >= budget or time.time() > deadline:
                    break

                # Select three distinct random individuals
                r1, r2, r3 = rng.choice(pop_size, 3, replace=False)
                while r1 == i or r2 == i or r3 == i:
                    r1, r2, r3 = rng.choice(pop_size, 3, replace=False)

                # Mutation: v = x_r1 + F * (x_r2 - x_r3)
                v = population[r1] + F * (population[r2] - population[r3])

                # Crossover
                trial = population[i].copy()
                j_rand = rng.randint(dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]

                # Clip to bounds
                trial = np.clip(trial, -5, 5)

                # Evaluate
                ft = call_func(trial)
                if ft is None:
                    break

                # Selection
                if ft <= pop_f[i]:
                    new_population[i] = trial
                    new_pop_f[i] = ft

            population = new_population
            pop_f = new_pop_f

        # Phase 2: Nelder-Mead local refinement
        # Use Nelder-Mead to polish the solution in the basin of the best point
        if call_count < budget and time.time() < deadline:
            x0 = best_x.copy()
            fx0 = best_f

            # Create initial simplex
            simplex = [x0.copy()]
            for d in range(dim):
                perturbed = x0.copy()
                if perturbed[d] == 0:
                    perturbed[d] = 0.1
                else:
                    perturbed[d] += 0.1
                simplex.append(perturbed)

            simplex_f = [fx0]
            for i in range(1, dim + 1):
                if call_count >= budget or time.time() > deadline:
                    break
                fi = call_func(simplex[i])
                if fi is None:
                    break
                simplex_f.append(fi)

            # Nelder-Mead parameters
            alpha = 1.0  # reflection
            gamma = 2.0  # expansion
            rho = 0.5    # contraction
            sigma_nm = 0.5  # shrink

            max_nm_iterations = 50
            for nm_iter in range(max_nm_iterations):
                if call_count >= budget or time.time() > deadline:
                    break

                # Sort simplex by fitness
                sorted_idx = np.argsort(simplex_f)
                simplex = [simplex[i] for i in sorted_idx]
                simplex_f = [simplex_f[i] for i in sorted_idx]

                # Update best
                if simplex_f[0] < best_f:
                    best_f = simplex_f[0]
                    best_x = simplex[0].copy()

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5, 5)
                f_r = call_func(x_r)
                if f_r is None:
                    break

                if simplex_f[0] <= f_r < simplex_f[-2]:
                    # Accept reflection
                    simplex[-1] = x_r
                    simplex_f[-1] = f_r
                elif f_r < simplex_f[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5, 5)
                    f_e = call_func(x_e)
                    if f_e is None:
                        break
                    if f_e < f_r:
                        simplex[-1] = x_e
                        simplex_f[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        simplex_f[-1] = f_r
                else:
                    # Contraction
                    if f_r < simplex_f[-1]:
                        # Outside contraction
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        # Inside contraction
                        x_c = centroid - rho * (centroid - simplex[-1])
                    x_c = np.clip(x_c, -5, 5)
                    f_c = call_func(x_c)
                    if f_c is None:
                        break

                    if f_c < min(f_r, simplex_f[-1]):
                        simplex[-1] = x_c
                        simplex_f[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            if call_count >= budget or time.time() > deadline:
                                break
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            f_i = call_func(simplex[i])
                            if f_i is None:
                                break
                            simplex_f[i] = f_i
                        break  # Restart NM after shrink

                # Check convergence
                if np.max(simplex_f) - np.min(simplex_f) < 1e-8:
                    break

        # Phase 3: Fine-grained local search around best point
        if call_count < budget and time.time() < deadline:
            x = best_x.copy()
            fx = best_f

            for step in [0.05, 0.02, 0.01, 0.005]:
                if call_count >= budget or time.time() > deadline:
                    break

                improved = True
                while improved and call_count < budget and time.time() < deadline:
                    improved = False
                    for d in range(dim):
                        if call_count >= budget or time.time() > deadline:
                            break
                        for direction in [1, -1]:
                            x_new = x.copy()
                            x_new[d] += direction * step
                            x_new = np.clip(x_new, -5, 5)
                            fx_new = call_func(x_new)
                            if fx_new is None:
                                break
                            if fx_new < fx - 1e-10:
                                x = x_new
                                fx = fx_new
                                improved = True
                                if fx < best_f:
                                    best_f = fx
                                    best_x = x.copy()
                                break

        return best_x, best_f
# EVOLVE-BLOCK-END
