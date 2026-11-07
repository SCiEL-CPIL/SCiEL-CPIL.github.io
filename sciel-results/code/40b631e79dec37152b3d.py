"""Optimizer using population-based evolutionary strategy with Nelder-Mead refinement
for BBOB f13 (Sharp Ridge).

Strategy:
1. Population-based ES with 50 diverse candidates and dynamically shrinking sigma
   to maintain diversity and exploit the ridge's anisotropy.
2. Nelder-Mead local refinement for precise convergence once the ridge neighborhood
   is located.
3. Multi-start with budget allocation between exploration and exploitation phases.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """
    Population-based ES with Nelder-Mead refinement for Sharp Ridge functions.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t0 = time.time()
        deadline = t0 + 290.0

        def elapsed_ok():
            return time.time() < deadline

        rng = np.random.default_rng(42)

        def clip(v):
            return np.clip(v, -5.0, 5.0)

        best_x = None
        best_f = np.inf
        evaluated = 0

        def eval_point(x):
            nonlocal evaluated, best_x, best_f
            x = np.asarray(x, dtype=float)
            x = clip(x)
            f = func(x)
            evaluated += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Nelder-Mead refinement
        def run_nelder_mead(center, initial_step, max_iter=100):
            nonlocal best_x, best_f, evaluated, deadline, elapsed_ok

            if evaluated >= budget - 10 or not elapsed_ok():
                return

            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma_nm = 0.5

            # Initialize simplex
            simplex = []
            simplex_f = []

            x0 = clip(center)
            f0 = eval_point(x0)
            simplex.append(x0)
            simplex_f.append(f0)

            if evaluated >= budget - 10:
                return

            for i in range(dim):
                if evaluated >= budget - 10 or not elapsed_ok():
                    return
                x_i = center.copy()
                x_i[i] += initial_step
                x_i = clip(x_i)
                f_i = eval_point(x_i)
                simplex.append(x_i)
                simplex_f.append(f_i)

            # Main NM loop
            for _ in range(max_iter):
                if not elapsed_ok() or evaluated >= budget - 5:
                    break

                # Sort simplex by fitness
                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]

                # Check for convergence
                if max(simplex_f) - min(simplex_f) < 1e-10:
                    break

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                worst = simplex[-1]
                worst_f = simplex_f[-1]

                # Reflection
                x_r = centroid + alpha * (centroid - worst)
                x_r = clip(x_r)
                f_r = eval_point(x_r)

                if evaluated >= budget - 5:
                    break

                if f_r < simplex_f[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = clip(x_e)
                    f_e = eval_point(x_e)

                    if evaluated >= budget - 5:
                        break

                    if f_e < f_r:
                        simplex[-1] = x_e
                        simplex_f[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        simplex_f[-1] = f_r
                elif f_r < simplex_f[-2]:
                    # Accept reflection
                    simplex[-1] = x_r
                    simplex_f[-1] = f_r
                else:
                    # Contraction
                    if f_r < worst_f:
                        # Outer contraction
                        x_c = centroid + rho * (x_r - centroid)
                        x_c = clip(x_c)
                        f_c = eval_point(x_c)

                        if evaluated >= budget - 5:
                            break

                        if f_c < f_r:
                            simplex[-1] = x_c
                            simplex_f[-1] = f_c
                        else:
                            # Shrink
                            for i in range(1, dim + 1):
                                simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                                simplex_f[i] = eval_point(simplex[i])
                                if evaluated >= budget - 5:
                                    break
                    else:
                        # Inner contraction
                        x_c = centroid - rho * (centroid - worst)
                        x_c = clip(x_c)
                        f_c = eval_point(x_c)

                        if evaluated >= budget - 5:
                            break

                        if f_c < worst_f:
                            simplex[-1] = x_c
                            simplex_f[-1] = f_c
                        else:
                            # Shrink
                            for i in range(1, dim + 1):
                                simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                                simplex_f[i] = eval_point(simplex[i])
                                if evaluated >= budget - 5:
                                    break

        # Population-based evolutionary strategy
        def run_population_es(center, budget_alloc, initial_sigma=2.0):
            nonlocal evaluated, best_x, best_f, deadline, elapsed_ok

            if evaluated >= budget or not elapsed_ok():
                return

            pop_size = 50
            sigma = initial_sigma
            sigma_min = 1e-4
            sigma_max = 5.0

            # Initialize population around center
            population = []
            pop_f = []

            for _ in range(min(pop_size, budget_alloc)):
                if evaluated >= budget or not elapsed_ok():
                    break
                x_candidate = clip(center + sigma * rng.standard_normal(dim))
                f_val = eval_point(x_candidate)
                population.append(x_candidate)
                pop_f.append(f_val)

            if len(population) == 0:
                return

            # Sort by fitness
            order = np.argsort(pop_f)
            population = [population[i] for i in order]
            pop_f = [pop_f[i] for i in order]

            # Track best
            if pop_f[0] < best_f:
                best_f = pop_f[0]
                best_x = population[0].copy()

            # Main ES loop
            stagnation_counter = 0
            best_f_prev = best_f
            window_size = 50

            while evaluated < budget and elapsed_ok():
                if len(population) < 2:
                    break

                # Generate new candidates
                new_candidates = []
                new_f = []

                for _ in range(min(pop_size, budget - evaluated)):
                    if evaluated >= budget or not elapsed_ok():
                        break

                    # Select parent (tournament selection)
                    idx1 = rng.integers(0, len(population))
                    idx2 = rng.integers(0, len(population))
                    if pop_f[idx1] < pop_f[idx2]:
                        parent = population[idx1]
                    else:
                        parent = population[idx2]

                    # Gaussian perturbation with current sigma
                    x_candidate = clip(parent + sigma * rng.standard_normal(dim))
                    f_val = eval_point(x_candidate)
                    new_candidates.append(x_candidate)
                    new_f.append(f_val)

                if len(new_candidates) == 0:
                    break

                # Combine population with new candidates
                combined_pop = population + new_candidates
                combined_f = pop_f + new_f

                # Sort and keep best pop_size
                order = np.argsort(combined_f)
                order = order[:pop_size]
                population = [combined_pop[i] for i in order]
                pop_f = [combined_f[i] for i in order]

                # Update best
                if pop_f[0] < best_f:
                    best_f = pop_f[0]
                    best_x = population[0].copy()
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1

                # Adapt sigma based on stagnation
                if stagnation_counter >= window_size:
                    sigma = max(sigma * 0.7, sigma_min)
                    stagnation_counter = 0
                elif stagnation_counter == 0:
                    sigma = min(sigma * 1.1, sigma_max)

                # If sigma is very small, do local refinement
                if sigma < 1e-3 and evaluated < budget - 20:
                    # Run Nelder-Mead from best point
                    run_nelder_mead(best_x, 0.1, max_iter=50)
                    return

        # Main optimization loop
        # Phase 1: Population-based ES for exploration
        es_budget = int(budget * 0.7)
        start_point = rng.uniform(-3.0, 3.0, size=dim)
        run_population_es(start_point, es_budget, initial_sigma=2.0)

        # Phase 2: Nelder-Mead refinement from best point
        if best_x is not None and evaluated < budget - 10 and elapsed_ok():
            run_nelder_mead(best_x, 0.5, max_iter=100)

        # Phase 3: Additional ES refinement if budget remains
        if evaluated < budget - 10 and elapsed_ok():
            remaining_budget = budget - evaluated
            if remaining_budget > 20:
                run_population_es(best_x, remaining_budget, initial_sigma=0.5)

        # Final NM polish
        if best_x is not None and evaluated < budget - 5 and elapsed_ok():
            run_nelder_mead(best_x, 0.1, max_iter=50)

        return best_x, best_f
# EVOLVE-BLOCK-END
