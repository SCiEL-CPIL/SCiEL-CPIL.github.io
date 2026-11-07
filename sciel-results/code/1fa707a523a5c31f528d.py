"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Differential Evolution + Nelder-Mead polish for BBOB-style Katsuura function."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 270.0  # Safety margin

        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_made = 0

        def counted_func(x):
            nonlocal calls_made, best_f, best_x
            if calls_made >= self.budget:
                raise RuntimeError("Budget exceeded")
            if time.time() > deadline - 3:
                raise RuntimeError("Time limit")
            calls_made += 1
            val = float(func(x))
            if val < best_f:
                best_f = val
                best_x = np.array(x, copy=True)
            return val

        dim = self.dim
        budget = self.budget
        lb = np.full(dim, -5.0)
        ub = np.full(dim, 5.0)

        def clip(x):
            return np.clip(x, lb, ub)

        def nelder_mead(start_point, max_iter=300):
            """Nelder-Mead simplex optimization with adaptive step."""
            nonlocal best_f, best_x, calls_made

            # Build initial simplex with larger steps for multimodal landscape
            simplex = [start_point.copy()]
            for i in range(dim):
                new_point = start_point.copy()
                new_point[i] += 0.5
                simplex.append(clip(new_point))

            # Evaluate initial simplex
            values = []
            for x in simplex:
                try:
                    val = counted_func(x)
                    values.append(val)
                except RuntimeError:
                    return

            # Nelder-Mead parameters
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma_nm = 0.5

            for _ in range(max_iter):
                if calls_made >= budget or time.time() > deadline - 3:
                    return

                order = np.argsort(values)
                simplex = [simplex[i] for i in order]
                values = [values[i] for i in order]

                if abs(values[-1] - values[0]) < 1e-14:
                    break

                centroid = np.mean(simplex[:-1], axis=0)
                worst = simplex[-1]
                worst_val = values[-1]

                x_r = clip(centroid + alpha * (centroid - worst))
                try:
                    f_r = counted_func(x_r)
                except RuntimeError:
                    return

                if f_r < values[0]:
                    x_e = clip(centroid + gamma * (centroid - worst))
                    try:
                        f_e = counted_func(x_e)
                    except RuntimeError:
                        return
                    if f_e < f_r:
                        simplex[-1] = x_e
                        values[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        values[-1] = f_r
                elif f_r < values[-2]:
                    simplex[-1] = x_r
                    values[-1] = f_r
                else:
                    if f_r < worst_val:
                        x_c = clip(centroid + rho * (x_r - centroid))
                    else:
                        x_c = clip(centroid - rho * (centroid - worst))

                    try:
                        f_c = counted_func(x_c)
                    except RuntimeError:
                        return

                    if f_c < min(f_r, worst_val):
                        simplex[-1] = x_c
                        values[-1] = f_c
                    else:
                        for i in range(1, dim + 1):
                            simplex[i] = clip(simplex[0] + sigma_nm * (simplex[i] - simplex[0]))
                            try:
                                values[i] = counted_func(simplex[i])
                            except RuntimeError:
                                return

                min_idx = np.argmin(values)
                if values[min_idx] < best_f:
                    best_f = values[min_idx]
                    best_x = np.array(simplex[min_idx], copy=True)

        # Phase 1: Differential Evolution
        pop_size = 30
        F = 0.8  # scale factor
        CR = 0.9  # crossover rate

        # Initialize population
        population = [rng.uniform(-4, 4, dim) for _ in range(pop_size)]
        pop_values = []
        for x in population:
            try:
                val = counted_func(x)
                pop_values.append(val)
            except RuntimeError:
                break

        # DE loop
        max_de_iters = 150
        for gen in range(max_de_iters):
            if calls_made >= budget or time.time() > deadline - 10:
                break

            # Sort population
            order = np.argsort(pop_values)
            population = [population[i] for i in order]
            pop_values = [pop_values[i] for i in order]

            # Early stopping if converged
            if pop_values[-1] - pop_values[0] < 1e-12:
                break

            # Generate new candidates
            new_candidates = []
            for i in range(pop_size):
                if calls_made >= budget or time.time() > deadline - 3:
                    break

                # Select 3 distinct random individuals
                r1, r2, r3 = rng.choice([j for j in range(pop_size) if j != i], 3, replace=False)

                # Mutation
                mutant = clip(population[r1] + F * (population[r2] - population[r3]))

                # Crossover
                j_rand = rng.integers(dim)
                trial = np.zeros(dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = mutant[j]
                    else:
                        trial[j] = population[i][j]

                try:
                    val = counted_func(trial)
                except RuntimeError:
                    break

                # Selection
                if val <= pop_values[i]:
                    population[i] = trial
                    pop_values[i] = val

        # Phase 2: NM polish from top diverse solutions
        if best_x is not None:
            # Sort by fitness
            order = np.argsort(pop_values)
            top_k = min(5, len(population))

            # Pick diverse points from top
            selected = []
            for idx in order[:top_k]:
                p = population[idx]
                # Check diversity
                diverse = True
                for sp in selected:
                    if np.linalg.norm(p - sp) < 0.5:
                        diverse = False
                        break
                if diverse:
                    selected.append(p)

            for p in selected:
                if calls_made >= budget or time.time() > deadline - 10:
                    break
                nelder_mead(p.copy(), max_iter=200)

        # Phase 3: Final polish from global best
        if best_x is not None and calls_made < budget and time.time() < deadline - 10:
            nelder_mead(best_x.copy(), max_iter=300)

        # Phase 4: Use remaining budget for random exploration near best
        while calls_made < budget - 10 and time.time() < deadline - 10:
            if best_x is None:
                p = rng.uniform(-4, 4, dim)
            else:
                p = clip(best_x + rng.normal(0, 0.3, dim))
            try:
                counted_func(p)
            except RuntimeError:
                break

            if calls_made < budget - 20 and time.time() < deadline - 10:
                nelder_mead(p.copy(), max_iter=50)

        return best_x, best_f
# EVOLVE-BLOCK-END
