"""Optimizer for BBOB-style f4 (Büche-Rastrigin) function."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid optimizer: Differential Evolution for global search,
    Nelder-Mead simplex for local refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Track global best
        global_best_x = None
        global_best_fx = np.inf
        calls_made = 0

        def try_update(x, fx):
            nonlocal global_best_x, global_best_fx
            if fx < global_best_fx:
                global_best_fx = fx
                global_best_x = x.copy()
                return True
            return False

        def evaluate(x):
            nonlocal calls_made
            if calls_made >= budget:
                return None
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            calls_made += 1
            try_update(x, fx)
            return fx

        # ============================================================
        # Phase 1: Differential Evolution for global exploration
        # ============================================================
        pop_size = 20
        # Initialize population uniformly in the search space
        population = np.array([rng.uniform(-4.5, 4.5, size=dim) for _ in range(pop_size)])
        pop_f = np.full(pop_size, np.inf)

        for i in range(pop_size):
            if calls_made >= budget or time.time() > deadline:
                break
            fx = evaluate(population[i])
            if fx is not None:
                pop_f[i] = fx

        # DE parameters
        F = 0.8  # mutation factor
        CR = 0.7  # crossover rate
        best_idx = np.argmin(pop_f)
        de_stagnation = 0
        max_de_stagnation = 15

        while calls_made < budget and time.time() < deadline:
            # Find current best in population
            best_idx = np.argmin(pop_f)
            best_x = population[best_idx].copy()
            best_f = pop_f[best_idx]

            # Generate trial vectors for each member
            for i in range(pop_size):
                if calls_made >= budget or time.time() > deadline:
                    break

                # Select two distinct random indices different from i and best_idx
                candidates = [j for j in range(pop_size) if j != i and j != best_idx]
                if len(candidates) < 2:
                    candidates = [j for j in range(pop_size) if j != i]
                if len(candidates) < 2:
                    break

                a, b = rng.choice(candidates, size=2, replace=False)

                # Mutation: v = x_best + F * (x_a - x_b)
                v = best_x + F * (population[a] - population[b])

                # Crossover: trial vector
                trial = population[i].copy()
                j_rand = rng.integers(dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]

                # Clip to bounds
                trial = np.clip(trial, -5.0, 5.0)

                # Evaluate trial
                fx_trial = evaluate(trial)
                if fx_trial is None:
                    break

                # Selection
                if fx_trial <= pop_f[i]:
                    population[i] = trial
                    pop_f[i] = fx_trial

            # Check for improvement
            new_best_idx = np.argmin(pop_f)
            new_best_f = pop_f[new_best_idx]
            if new_best_f < best_f - 1e-10:
                de_stagnation = 0
            else:
                de_stagnation += 1

            # If DE is stagnating, restart with a fresh random population around best
            if de_stagnation >= max_de_stagnation:
                # Keep best, replace rest with new random points
                for i in range(pop_size):
                    if i == new_best_idx:
                        continue
                    if calls_made >= budget or time.time() > deadline:
                        break
                    # Mix: some random, some near best
                    if rng.random() < 0.5:
                        population[i] = rng.uniform(-4.5, 4.5, size=dim)
                    else:
                        population[i] = best_x + rng.standard_normal(dim) * 1.0
                        population[i] = np.clip(population[i], -5.0, 5.0)
                    fx = evaluate(population[i])
                    if fx is not None:
                        pop_f[i] = fx

                de_stagnation = 0
                max_de_stagnation = min(max_de_stagnation + 5, 30)

        # ============================================================
        # Phase 2: Nelder-Mead local refinement from best DE point
        # ============================================================
        if global_best_x is not None and calls_made < budget and time.time() < deadline:
            # Initialize simplex around global best
            x0 = global_best_x.copy()
            simplex = [x0.copy()]
            for j in range(dim):
                x_init = x0.copy()
                step = 0.5 if abs(x0[j]) < 4.0 else 0.2
                x_init[j] += step
                x_init = np.clip(x_init, -5.0, 5.0)
                simplex.append(x_init)

            # Evaluate simplex vertices
            simplex_f = [np.inf] * (dim + 1)
            simplex_f[0] = global_best_fx
            for i in range(1, dim + 1):
                if calls_made >= budget or time.time() > deadline:
                    break
                fx = evaluate(simplex[i])
                if fx is not None:
                    simplex_f[i] = fx

            # Nelder-Mead parameters
            alpha = 1.0  # reflection
            gamma = 2.0  # expansion
            rho = 0.5    # contraction
            sigma = 0.5  # shrink
            max_nm_iter = 200

            for nm_iter in range(max_nm_iter):
                if calls_made >= budget or time.time() > deadline:
                    break

                # Sort simplex by function value
                order = np.argsort(simplex_f)
                simplex = [simplex[i].copy() for i in order]
                simplex_f = [simplex_f[i] for i in order]

                # Check convergence
                if abs(simplex_f[0] - simplex_f[-1]) < 1e-8:
                    break

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5.0, 5.0)
                fx_r = evaluate(x_r)
                if fx_r is None:
                    break

                if fx_r < simplex_f[0]:
                    # Try expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5.0, 5.0)
                    fx_e = evaluate(x_e)
                    if fx_e is None:
                        break
                    if fx_e < fx_r:
                        simplex[-1] = x_e
                        simplex_f[-1] = fx_e
                    else:
                        simplex[-1] = x_r
                        simplex_f[-1] = fx_r
                elif fx_r < simplex_f[-2]:
                    simplex[-1] = x_r
                    simplex_f[-1] = fx_r
                else:
                    # Contraction
                    if fx_r < simplex_f[-1]:
                        # Outside contraction
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        # Inside contraction
                        x_c = centroid - rho * (x_r - centroid)
                    x_c = np.clip(x_c, -5.0, 5.0)
                    fx_c = evaluate(x_c)
                    if fx_c is None:
                        break

                    if fx_c <= min(fx_r, simplex_f[-1]):
                        simplex[-1] = x_c
                        simplex_f[-1] = fx_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            if calls_made >= budget or time.time() > deadline:
                                break
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                            fx_s = evaluate(simplex[i])
                            if fx_s is None:
                                break
                            simplex_f[i] = fx_s

            # After Nelder-Mead, do a few more local steps if budget remains
            if calls_made < budget and time.time() < deadline:
                x_best = global_best_x.copy()
                f_best = global_best_fx
                step_sizes = [0.3, 0.15, 0.075, 0.04, 0.02, 0.01, 0.005, 0.002]
                for step in step_sizes:
                    if calls_made >= budget or time.time() > deadline:
                        break
                    improved = True
                    while improved and calls_made < budget and time.time() < deadline:
                        improved = False
                        for j in range(dim):
                            for direction in [-1, 1]:
                                if calls_made >= budget or time.time() > deadline:
                                    break
                                x_trial = x_best.copy()
                                x_trial[j] += direction * step
                                x_trial = np.clip(x_trial, -5.0, 5.0)
                                fx_trial = evaluate(x_trial)
                                if fx_trial is not None and fx_trial < f_best:
                                    x_best = x_trial
                                    f_best = fx_trial
                                    improved = True
                                    break

        return global_best_x, global_best_fx
# EVOLVE-BLOCK-END
