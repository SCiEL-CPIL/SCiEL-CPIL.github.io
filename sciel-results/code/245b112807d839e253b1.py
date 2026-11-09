"""Multi-objective optimizer for RE34 using NSGA-II style approach."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using NSGA-II style with local refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = 320.0  # Safety margin before 330s kill

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(42)

        # Initial population: structured + random
        pop_size = 60
        init_points = []

        # Corners
        for i in range(2 ** dim):
            point = np.array([hi[j] if (i >> j) & 1 else lo[j] for j in range(dim)])
            init_points.append(point)

        # Midpoint
        init_points.append((lo + hi) / 2)

        # Face centers (midpoint of each pair of opposite faces)
        for j in range(dim):
            for k in range(j + 1, dim):
                for combo in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                    point = np.array([(lo[t] + hi[t]) / 2 if t not in (j, k)
                                      else (hi[t] if combo[t - j if t == j else t - k] else lo[t])
                                      for t in range(dim)])
                    # Simpler: just add some structured midpoints
                    break
            break

        # Add structured points along each axis
        for j in range(dim):
            for frac in [0.25, 0.75]:
                point = (lo + hi) / 2
                point[j] = lo[j] + frac * (hi[j] - lo[j])
                init_points.append(point)

        # Fill with random
        while len(init_points) < pop_size:
            init_points.append(lo + (hi - lo) * rng.random(dim))

        X = np.array(init_points[:pop_size])
        F = np.array([problem(x) for x in X])
        used = pop_size

        def dominates(a, b):
            """Return True if a dominates b (a is better in all objectives, strictly in at least one)."""
            return np.all(a <= b) and np.any(a < b)

        def non_dominated_rank(F):
            """Return rank array and fronts (list of lists of indices)."""
            n = len(F)
            rank = np.zeros(n, dtype=int)
            domination_count = np.zeros(n, dtype=int)
            dominated_set = [set() for _ in range(n)]

            for i in range(n):
                for j in range(i + 1, n):
                    if dominates(F[i], F[j]):
                        dominated_set[i].add(j)
                        domination_count[j] += 1
                    elif dominates(F[j], F[i]):
                        dominated_set[j].add(i)
                        domination_count[i] += 1

            fronts = []
            current_front = [i for i in range(n) if domination_count[i] == 0]
            current_rank = 0
            for i in current_front:
                rank[i] = current_rank

            while current_front:
                fronts.append(current_front)
                next_front = []
                for i in current_front:
                    for j in dominated_set[i]:
                        domination_count[j] -= 1
                        if domination_count[j] == 0:
                            rank[j] = current_rank + 1
                            next_front.append(j)
                current_front = next_front
                current_rank += 1

            # Assign remaining unassigned ranks
            for i in range(n):
                if rank[i] == 0 and i not in fronts[0] if fronts else True:
                    rank[i] = len(fronts) if fronts else 0

            return rank, fronts

        def crowding_distance(F, indices):
            """Compute crowding distance for given indices."""
            n = len(indices)
            if n <= 2:
                return np.full(n, np.inf)

            cd = np.zeros(n)
            m = F[indices].shape[1]

            for k in range(m):
                obj_vals = F[indices, k]
                min_val = np.min(obj_vals)
                max_val = np.max(obj_vals)
                if max_val - min_val < 1e-12:
                    continue
                sorted_idx = np.argsort(obj_vals)
                cd[sorted_idx[0]] = np.inf
                cd[sorted_idx[-1]] = np.inf
                for i in range(1, n - 1):
                    cd[sorted_idx[i]] += (obj_vals[sorted_idx[i + 1]] - obj_vals[sorted_idx[i - 1]]) / (max_val - min_val)

            return cd

        def select_parents(pop_F, pop_X, n_parents):
            """Select parents using tournament selection based on rank and crowding distance."""
            n = len(pop_F)
            rank, fronts = non_dominated_rank(pop_F)

            # Compute crowding distance for each front
            crowding = np.zeros(n)
            for front in fronts:
                if len(front) > 0:
                    cd = crowding_distance(pop_F, front)
                    for i, idx in enumerate(front):
                        crowding[idx] = cd[i]

            # Tournament selection
            selected = []
            for _ in range(n_parents):
                a, b = rng.integers(0, n, size=2)
                if rank[a] < rank[b]:
                    selected.append(a)
                elif rank[b] < rank[a]:
                    selected.append(b)
                else:
                    if crowding[a] > crowding[b]:
                        selected.append(a)
                    else:
                        selected.append(b)
            return selected

        # Main loop
        gen = 0
        while used < budget:
            if time.time() - start_time > deadline:
                break

            # Adaptive parameters: shift from exploration to exploitation
            progress = used / budget
            if progress < 0.3:
                # Early: more exploration
                n_offspring = 15
                F_factor_base = 0.8
                F_factor_range = 0.4
                CR_base = 0.8
                CR_range = 0.2
            elif progress < 0.7:
                # Mid: balanced
                n_offspring = 12
                F_factor_base = 0.6
                F_factor_range = 0.3
                CR_base = 0.7
                CR_range = 0.2
            else:
                # Late: more exploitation
                n_offspring = 10
                F_factor_base = 0.4
                F_factor_range = 0.3
                CR_base = 0.6
                CR_range = 0.2

            offspring_X = []
            offspring_F = []

            for _ in range(n_offspring):
                if used >= budget:
                    break

                # Select 3 parents for DE mutation
                p1, p2, p3 = rng.integers(0, len(X), size=3)
                while p2 == p3:
                    p3 = rng.integers(len(X))

                F_factor = F_factor_base + rng.random() * F_factor_range
                CR = CR_base + rng.random() * CR_range

                trial = X[p1].copy()
                r = rng.random(dim)
                for j in range(dim):
                    if r[j] < CR:
                        trial[j] = X[p1][j] + F_factor * (X[p2][j] - X[p3][j])
                    else:
                        trial[j] = X[p1][j]
                trial = np.clip(trial, lo, hi)

                ft = problem(trial)
                used += 1
                offspring_X.append(trial)
                offspring_F.append(ft)

            if not offspring_X:
                break

            offspring_X = np.array(offspring_X)
            offspring_F = np.array(offspring_F)

            # Combine parent and offspring
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])

            # Environmental selection
            n_keep = len(X)
            rank, fronts = non_dominated_rank(combined_F)

            selected_indices = []
            for front in fronts:
                if len(selected_indices) + len(front) <= n_keep:
                    selected_indices.extend(front)
                else:
                    remaining = n_keep - len(selected_indices)
                    cd = crowding_distance(combined_F, front)
                    sorted_by_cd = np.argsort(-cd)
                    for i in range(remaining):
                        selected_indices.append(front[sorted_by_cd[i]])
                    break

            X = combined_X[selected_indices]
            F = combined_F[selected_indices]
            gen += 1

        # Final local refinement: Nelder-Mead on top non-dominated points
        if used < budget and time.time() - start_time < deadline - 5.0:
            # Get non-dominated points from final population
            rank, fronts = non_dominated_rank(F)
            if fronts and len(fronts[0]) > 0:
                nd_indices = fronts[0]
                # Polish top few non-dominated points
                n_polish = min(len(nd_indices), max(1, (budget - used) // 5))
                for idx in nd_indices[:n_polish]:
                    if used >= budget or time.time() - start_time > deadline - 2.0:
                        break
                    x0 = X[idx].copy()
                    # Simple coordinate-wise local search
                    step = 0.05 * (hi - lo)
                    for coord in range(dim):
                        for direction in [-1, 1]:
                            x_trial = x0.copy()
                            x_trial[coord] += direction * step
                            x_trial = np.clip(x_trial, lo, hi)
                            f_trial = problem(x_trial)
                            used += 1
                            if used >= budget or time.time() - start_time > deadline - 2.0:
                                break
                            # Check if improvement in at least one objective without worsening others too much
                            f_curr = F[idx]
                            if np.all(f_trial <= f_curr + 0.01 * np.abs(f_curr) + 1e-6):
                                if np.any(f_trial < f_curr):
                                    x0 = x_trial
                                    F[idx] = f_trial
                    X[idx] = x0

        return X
# EVOLVE-BLOCK-END
