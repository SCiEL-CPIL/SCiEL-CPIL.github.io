"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid (mu+lambda)-ES with Nelder-Mead polishing for Lunacek bi-Rastrigin."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        start_time = time.time()
        max_time = 290.0  # Safety margin before 300s limit

        rng = np.random.default_rng(42)

        best_x = np.zeros(dim)
        best_f = float('inf')
        calls = 0

        # Phase 1: (mu+lambda)-ES for exploration
        mu = 4
        lam = 12
        sigma = 1.8
        sigma_min = 1e-8
        sigma_max = 4.0

        # Initial population
        population = []
        for i in range(lam):
            if calls >= budget or time.time() - start_time > max_time:
                break
            if i == 0:
                x = np.zeros(dim)
            else:
                x = rng.uniform(-4.5, 4.5, dim)
            fx = func(x)
            calls += 1
            population.append((fx, x.copy()))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        population.sort(key=lambda t: t[0])

        # ES main loop
        iters_no_improve = 0
        stagnation_limit = 40
        es_phase_budget = int(budget * 0.7)  # Use 70% of budget for ES

        while calls < es_phase_budget and time.time() - start_time < max_time:
            # Select mu best
            parents = [x for _, x in population[:mu]]
            mean_x = np.mean(parents, axis=0)

            # Generate offspring
            offspring = []
            for j in range(lam):
                if calls >= es_phase_budget or time.time() - start_time > max_time:
                    break
                direction = rng.standard_normal(dim)
                candidate = mean_x + sigma * direction
                candidate = np.clip(candidate, -5, 5)
                fc = func(candidate)
                calls += 1
                offspring.append((fc, candidate.copy()))
                if fc < best_f:
                    best_f = fc
                    best_x = candidate.copy()

            if not offspring:
                break

            # Combine and select
            combined = population + offspring
            combined.sort(key=lambda t: t[0])
            population = combined[:lam]

            # Track improvement
            if population[0][0] < best_f - 1e-12:
                iters_no_improve = 0
            else:
                iters_no_improve += 1

            # Adapt sigma using success rate
            parent_best = parents[0][0] if parents else best_f
            successes = sum(1 for f, _ in offspring if f < parent_best)
            success_rate = successes / len(offspring) if offspring else 0.0

            if success_rate > 0.25:
                sigma = min(sigma * 1.06, sigma_max)
            elif success_rate < 0.15:
                sigma = max(sigma * 0.94, sigma_min)

            # Stagnation restart
            if iters_no_improve >= stagnation_limit:
                population = []
                for i in range(lam):
                    if calls >= es_phase_budget or time.time() - start_time > max_time:
                        break
                    x = best_x + rng.normal(0, 1.5, dim)
                    x = np.clip(x, -5, 5)
                    fx = func(x)
                    calls += 1
                    population.append((fx, x.copy()))
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                population.sort(key=lambda t: t[0])
                sigma = 1.2
                iters_no_improve = 0

        # Phase 2: Nelder-Mead polishing around best solution
        if calls < budget and time.time() - start_time < max_time:
            nm_start = time.time()
            nm_max_time = max(0, max_time - (time.time() - start_time))
            nm_budget = budget - calls

            # Build initial simplex
            simplex = [best_x.copy()]
            base_sigma = 0.5
            for i in range(dim):
                pt = best_x.copy()
                pt[i] += base_sigma
                pt = np.clip(pt, -5, 5)
                simplex.append(pt)

            # Evaluate simplex
            simplex_f = []
            for pt in simplex:
                if calls >= budget or time.time() - start_time > max_time:
                    break
                f = func(pt)
                calls += 1
                simplex_f.append(f)

            # Nelder-Mead parameters
            alpha = 1.0   # reflection
            gamma = 2.0   # expansion
            rho = 0.5     # contraction
            sigma_shrink = 0.5

            nm_iter = 0
            max_nm_iter = min(nm_budget, 200)

            while nm_iter < max_nm_iter and calls < budget and time.time() - start_time < max_time:
                nm_iter += 1

                # Sort simplex by fitness
                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]

                # Check convergence
                if simplex_f[-1] - simplex_f[0] < 1e-10:
                    break

                # Centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                worst_pt = simplex[-1]
                worst_f = simplex_f[-1]
                x_ref = centroid + alpha * (centroid - worst_pt)
                x_ref = np.clip(x_ref, -5, 5)

                if calls >= budget or time.time() - start_time > max_time:
                    break
                f_ref = func(x_ref)
                calls += 1

                if f_ref < simplex_f[0]:
                    # Expansion
                    x_exp = centroid + gamma * (x_ref - centroid)
                    x_exp = np.clip(x_exp, -5, 5)
                    if calls >= budget or time.time() - start_time > max_time:
                        break
                    f_exp = func(x_exp)
                    calls += 1
                    if f_exp < f_ref:
                        x_new, f_new = x_exp, f_exp
                    else:
                        x_new, f_new = x_ref, f_ref
                    simplex[-1] = x_new
                    simplex_f[-1] = f_new
                    if f_new < best_f:
                        best_f = f_new
                        best_x = x_new.copy()
                elif f_ref < simplex_f[-2]:
                    # Accept reflection
                    simplex[-1] = x_ref
                    simplex_f[-1] = f_ref
                    if f_ref < best_f:
                        best_f = f_ref
                        best_x = x_ref.copy()
                else:
                    # Contraction
                    x_con = centroid + rho * (worst_pt - centroid)
                    x_con = np.clip(x_con, -5, 5)
                    if calls >= budget or time.time() - start_time > max_time:
                        break
                    f_con = func(x_con)
                    calls += 1
                    if f_con < worst_f:
                        simplex[-1] = x_con
                        simplex_f[-1] = f_con
                        if f_con < best_f:
                            best_f = f_con
                            best_x = x_con.copy()
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma_shrink * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            if calls >= budget or time.time() - start_time > max_time:
                                break
                            f = func(simplex[i])
                            calls += 1
                            simplex_f[i] = f
                            if f < best_f:
                                best_f = f
                                best_x = simplex[i].copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
