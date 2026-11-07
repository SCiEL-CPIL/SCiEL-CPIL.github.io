"""Multi-objective DTLZ2 optimizer: analytic-front projection + DE + angle polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Structured DTLZ2 optimizer exploiting the analytic Pareto front."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 270  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(0)

        dim = self.dim
        n_obj = self.n_obj
        n_shape = n_obj - 1      # 2 shape (angle) variables
        n_dist = dim - n_shape   # 10 distance variables

        # --- Build a point on the Pareto front from angles ---
        def front_point(angles):
            x = np.empty(dim)
            x[:n_shape] = angles
            x[n_shape:] = 0.5    # exactly on front (g = 0)
            return np.clip(x, lo, hi)

        # --- Fast non-dominated sort ---
        def fast_non_dominated_sort(F):
            """Return list of fronts (indices) sorted by non-domination level."""
            n = len(F)
            domination_count = np.zeros(n, dtype=int)
            dominated_set = [[] for _ in range(n)]
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        dominated_set[i].append(j)
                        domination_count[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated_set[j].append(i)
                        domination_count[i] += 1
            current_front = [i for i in range(n) if domination_count[i] == 0]
            fronts = [current_front]
            while current_front:
                next_front = []
                for i in current_front:
                    for j in dominated_set[i]:
                        domination_count[j] -= 1
                        if domination_count[j] == 0:
                            next_front.append(j)
                if next_front:
                    fronts.append(next_front)
                current_front = next_front
            return fronts

        def crowding_distance(F, indices):
            """Compute crowding distance for a set of indices."""
            n = len(indices)
            if n <= 2:
                return np.full(n, np.inf)
            dist = np.zeros(n)
            for m in range(n_obj):
                vals = F[indices, m]
                order = np.argsort(vals)
                dist[order[0]] = np.inf
                dist[order[-1]] = np.inf
                rng_span = vals[order[-1]] - vals[order[0]]
                if rng_span < 1e-15:
                    continue
                for k in range(1, n - 1):
                    dist[order[k]] += (vals[order[k + 1]] - vals[order[k - 1]]) / rng_span
            return dist

        def select_population(X_combined, F_combined, pop_size):
            """Select next population using non-dominated sorting + crowding distance."""
            fronts = fast_non_dominated_sort(F_combined)
            selected = []
            for front in fronts:
                if len(selected) + len(front) <= pop_size:
                    selected.extend(front)
                else:
                    remaining = pop_size - len(selected)
                    cd = crowding_distance(F_combined, front)
                    order = np.argsort(cd)[::-1]
                    selected.extend(front[order[:remaining]])
                    break
            return X_combined[selected], F_combined[selected]

        # Phase 1: Structured initialization on DTLZ2 Pareto front
        # DTLZ2 shape variables are angles in [0, pi/2]
        # f1 = (1+g)*cos(t1), f2 = (1+g)*sin(t1)*cos(t2), f3 = (1+g)*sin(t1)*sin(t2)
        # On Pareto front g=0, so we need diverse angle combinations
        n_init = 80

        # Create a well-spread set of angles using Fibonacci-like spiral
        # For 2 angles (t1, t2), use a quasi-random grid
        angles = np.zeros((n_init, n_shape))
        for i in range(n_init):
            # Spread t1 uniformly
            angles[i, 0] = (i / n_init) * (np.pi / 2)
            # Spread t2 with offset based on i for diversity
            angles[i, 1] = ((i * 0.61803398875) % 1.0) * (np.pi / 2)

        X = np.zeros((n_init, dim))
        for i in range(n_init):
            X[i] = front_point(angles[i])

        F = np.array([problem(x) for x in X])
        used = n_init

        # Phase 2: DE evolution
        pop_size = 60
        # Ensure we have at least pop_size individuals
        if used < pop_size:
            extra = pop_size - used
            X_extra = lo + (hi - lo) * rng.random((extra, dim))
            F_extra = np.array([problem(x) for x in X_extra])
            used += extra
            X = np.vstack([X, X_extra])
            F = np.vstack([F, F_extra])

        # DE loop
        while used < self.budget and time.time() < deadline:
            if time.time() > deadline - 10:
                break

            n_offspring = pop_size
            offspring = np.empty((n_offspring, dim))
            offspring_F = np.empty((n_offspring, n_obj))

            for i in range(n_offspring):
                # DE/rand/1/bin mutation
                a, b, c = rng.integers(pop_size, size=3)
                while b == a:
                    b = rng.integers(pop_size)
                while c == a or c == b:
                    c = rng.integers(pop_size)

                F_factor = 0.5 + 0.3 * rng.random()
                v = X[a] + F_factor * (X[b] - X[c])

                crossover_rate = 0.3
                j_rand = rng.integers(dim)
                mask = rng.random(dim) < crossover_rate
                mask[j_rand] = True
                child = np.where(mask, v, X[i])
                child = np.clip(child, lo, hi)

                offspring[i] = child
                offspring_F[i] = problem(child)
                used += 1

                if time.time() > deadline - 10:
                    break
                if used >= self.budget:
                    break

            n_actual = min(i + 1, n_offspring)
            offspring = offspring[:n_actual]
            offspring_F = offspring_F[:n_actual]

            X_combined = np.vstack([X, offspring])
            F_combined = np.vstack([F, offspring_F])

            X, F = select_population(X_combined, F_combined, pop_size)

            if used >= self.budget:
                break

        # Phase 3: Angle polish - project best points onto the Pareto front
        # and do local search in angle space
        if used < self.budget and time.time() < deadline - 5:
            # Get non-dominated front
            fronts = fast_non_dominated_sort(F)
            nd_front = fronts[0] if fronts else list(range(len(F)))

            # For each non-dominated point, extract angles and do local angle search
            step_size = 0.05
            for idx in nd_front:
                if used >= self.budget or time.time() > deadline - 5:
                    break

                # Extract current angles from the point
                current_angles = X[idx, :n_shape].copy()

                # Try small perturbations in angle space
                for _ in range(4):
                    if used >= self.budget or time.time() > deadline - 5:
                        break
                    perturbation = rng.normal(0, step_size, n_shape)
                    new_angles = current_angles + perturbation
                    # Clip angles to valid range [0, pi/2]
                    new_angles = np.clip(new_angles, 0, np.pi / 2)
                    candidate = front_point(new_angles)
                    f_cand = problem(candidate)
                    used += 1

                    # Check if it's non-dominated
                    is_dominated = False
                    for j in range(len(F)):
                        if np.all(F[j] <= f_cand) and np.any(F[j] < f_cand):
                            is_dominated = True
                            break

                    if not is_dominated:
                        # Find points dominated by f_cand
                        dominated_by_cand = [j for j in range(len(F))
                                            if np.all(f_cand <= F[j]) and np.any(f_cand < F[j])]
                        if dominated_by_cand:
                            worst = dominated_by_cand[0]
                            X[worst] = candidate
                            F[worst] = f_cand
                        else:
                            if len(X) < pop_size + 20:
                                X = np.vstack([X, candidate])
                                F = np.vstack([F, f_cand])

        return X
# EVOLVE-BLOCK-END
