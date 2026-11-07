import numpy as np
import time
import math
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = float('inf')
        calls_made = 0

        def clip(x):
            return np.clip(x, -5, 5)

        def remaining_time():
            return deadline - time.time()

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            x = clip(np.asarray(x, dtype=float))
            fx = float(func(x))
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        rng = np.random.default_rng(42)

        # Phase 1: Stratified initialization for diverse coverage
        # Use stratified sampling in [-4, 4]^5 (where minimum lies) plus some in [-5, 5]^5
        n_stratified = 30
        n_random = 20
        initial_points = []

        # Stratified sampling: divide each dim into 2 strata, sample 1 per stratum combo
        # 2^5 = 32 strata, sample from each
        strata_bounds = [(-4, -1), (-1, 4)]
        for s0 in range(2):
            for s1 in range(2):
                for s2 in range(2):
                    for s3 in range(2):
                        for s4 in range(2):
                            x = np.array([
                                rng.uniform(strata_bounds[s0][0], strata_bounds[s0][1]),
                                rng.uniform(strata_bounds[s1][0], strata_bounds[s1][1]),
                                rng.uniform(strata_bounds[s2][0], strata_bounds[s2][1]),
                                rng.uniform(strata_bounds[s3][0], strata_bounds[s3][1]),
                                rng.uniform(strata_bounds[s4][0], strata_bounds[s4][1]),
                            ])
                            initial_points.append(x)

        # Add random points in full range
        for _ in range(n_random):
            initial_points.append(rng.uniform(-5, 5, dim))

        # Add center and near-center
        initial_points.append(np.zeros(dim))
        initial_points.append(rng.uniform(-0.5, 0.5, dim))

        # Evaluate initial points
        for x0 in initial_points:
            if calls_made >= budget or remaining_time() < 15.0:
                break
            evaluate(clip(x0))

        if best_x is None:
            best_x = np.zeros(dim)
            evaluate(best_x)

        # Phase 2: Multi-start DE with population of 50
        pop_size = 50
        remaining_calls = budget - calls_made

        if remaining_calls > pop_size and remaining_time() > 20.0:
            # Initialize population with diverse candidates
            population = []
            fitnesses = []

            # Seed with best found
            population.append(best_x.copy())
            fitnesses.append(best_f)

            # Add stratified points near best
            for i in range(10):
                x = clip(best_x + 0.5 * rng.standard_normal(dim))
                population.append(x)
                fitnesses.append(evaluate(x))

            # Add mid-range diverse points
            for i in range(20):
                x = clip(rng.uniform(-4, 4, dim))
                population.append(x)
                fitnesses.append(evaluate(x))

            # Add full-range points
            for i in range(pop_size - len(population)):
                if calls_made >= budget or remaining_time() < 10.0:
                    break
                x = clip(rng.uniform(-5, 5, dim))
                population.append(x)
                fitnesses.append(evaluate(x))

            # Pad if we ran out of budget during init
            while len(population) < pop_size:
                population.append(best_x.copy())
                fitnesses.append(best_f)

            # Sort by fitness
            order = np.argsort(fitnesses)
            population = [population[i] for i in order]
            fitnesses = [fitnesses[i] for i in order]

            if fitnesses[0] < best_f:
                best_f = fitnesses[0]
                best_x = population[0].copy()

            # DE/rand/1 loop with adaptive parameters
            F = 0.8  # scaling factor
            CR = 0.85  # crossover rate
            stagnation_count = 0
            prev_best = best_f

            max_generations = max(1, (budget - calls_made - pop_size) // pop_size)

            for gen in range(max_generations):
                if calls_made >= budget or remaining_time() < 5.0:
                    break

                # Update best from population
                if fitnesses[0] < best_f:
                    best_f = fitnesses[0]
                    best_x = population[0].copy()

                # Detect stagnation
                if best_f < prev_best - 1e-10:
                    stagnation_count = 0
                    prev_best = best_f
                else:
                    stagnation_count += 1

                # Adapt F and CR based on stagnation
                if stagnation_count > 3:
                    F = max(0.3, F * 0.9)
                    CR = min(0.95, CR * 1.05)
                elif stagnation_count == 0:
                    F = min(0.9, F * 1.02)
                    CR = max(0.7, CR * 0.98)

                new_pop = []
                new_fit = []

                # Process population in parallel (vectorized where possible)
                for i in range(pop_size):
                    if calls_made >= budget or remaining_time() < 3.0:
                        new_pop.append(population[i])
                        new_fit.append(fitnesses[i])
                        continue

                    # Select three distinct random indices different from i
                    candidates = list(range(pop_size))
                    candidates.remove(i)
                    r1_idx, r2_idx, r3_idx = rng.choice(candidates, size=3, replace=False)
                    r1, r2, r3 = int(r1_idx), int(r2_idx), int(r3_idx)

                    # Mutation: v = x[r1] + F * (x[r2] - x[r3])
                    v = population[r1] + F * (population[r2] - population[r3])
                    v = clip(v)

                    # Crossover: binary crossover with CR
                    trial = np.empty(dim)
                    j_rand = rng.integers(0, dim)
                    for j in range(dim):
                        if rng.random() < CR or j == j_rand:
                            trial[j] = v[j]
                        else:
                            trial[j] = population[i][j]

                    trial = clip(trial)
                    f_trial = evaluate(trial)

                    # Selection
                    if f_trial <= fitnesses[i]:
                        new_pop.append(trial)
                        new_fit.append(f_trial)
                    else:
                        new_pop.append(population[i])
                        new_fit.append(fitnesses[i])

                # Sort new population
                order = np.argsort(new_fit)
                population = [new_pop[i] for i in order]
                fitnesses = [new_fit[i] for i in order]

                # Inject best into population to prevent loss
                if fitnesses[0] > best_f:
                    population[0] = best_x.copy()
                    fitnesses[0] = best_f
                    order = np.argsort(fitnesses)
                    population = [population[i] for i in order]
                    fitnesses = [fitnesses[i] for i in order]

                # Periodically inject diverse random points to maintain exploration
                if gen % 10 == 5 and gen > 0:
                    for k in range(3):
                        idx = rng.integers(1, pop_size)
                        population[idx] = clip(rng.uniform(-5, 5, dim))
                        fitnesses[idx] = evaluate(population[idx])
                    order = np.argsort(fitnesses)
                    population = [population[i] for i in order]
                    fitnesses = [fitnesses[i] for i in order]

        # Phase 3: Final polish with adaptive sigma (1+1)-ES
        if calls_made < budget and remaining_time() > 5.0:
            local_budget = min(budget - calls_made, 80)
            local_calls = [0]
            sigma = 0.3
            success_rate = 0.5
            x_curr = best_x.copy()
            f_curr = best_f

            for step in range(local_budget):
                if calls_made >= budget or remaining_time() < 3.0:
                    break

                # Generate candidate
                direction = rng.standard_normal(dim)
                x_trial = clip(x_curr + sigma * direction)
                f_trial = evaluate(x_trial)

                # Update success rate (exponential moving average)
                success = 1.0 if f_trial < f_curr else 0.0
                success_rate = 0.9 * success_rate + 0.1 * success

                # Adapt sigma based on success rate (target ~1/4 success for optimal exploration)
                if success_rate > 0.25:
                    sigma = min(1.0, sigma * 1.1)
                elif success_rate < 0.15:
                    sigma = max(1e-6, sigma * 0.9)

                if f_trial < f_curr:
                    x_curr = x_trial.copy()
                    f_curr = f_trial
                    if f_curr < best_f:
                        best_f = f_curr
                        best_x = x_curr.copy()

        # Phase 4: Tight Nelder-Mead final polish if time allows
        if calls_made < budget and remaining_time() > 3.0:
            local_budget = min(budget - calls_made, 40)
            if local_budget > 5:
                local_calls = [0]

                def local_func(x):
                    local_calls[0] += 1
                    if local_calls[0] > local_budget:
                        return best_f + 10.0
                    return evaluate(x)

                try:
                    res = minimize(local_func, best_x, method='Nelder-Mead',
                                 options={'maxiter': local_budget, 'xatol': 1e-10, 'fatol': 1e-12})
                    if res.fun < best_f:
                        best_f = float(res.fun)
                        best_x = clip(res.x)
                except Exception:
                    pass

        return best_x, best_f
