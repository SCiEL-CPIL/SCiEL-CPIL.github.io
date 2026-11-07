"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer with deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 50s safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Fixed reference point from task
        ref_point = np.array([1705.0, 12.0, 0.3])

        # Population size
        pop_size = min(40, budget // 2)
        if pop_size < 10:
            pop_size = 10

        # Initialize population
        X = lo + (hi - lo) * np.random.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        def dominates(a, b):
            """Return True if a dominates b (a <= b in all, a < b in at least one)."""
            return np.all(a <= b) and np.any(a < b)

        def non_dominated_sort(F_arr):
            """Return list of fronts, each front is a list of indices."""
            n = len(F_arr)
            S = [[] for _ in range(n)]  # domination count
            n_dom = [0] * n  # number of points dominated by i
            fronts = []
            front0 = []
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if dominates(F_arr[i], F_arr[j]):
                        S[i].append(j)
                        n_dom[i] += 1
                    elif dominates(F_arr[j], F_arr[i]):
                        n_dom[i] += 1  # wait, this is wrong
                if n_dom[i] == 0:
                    front0.append(i)
            # Actually let me redo this properly
            fronts = []
            S = [[] for _ in range(n)]
            n_dom = [0] * n
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if dominates(F_arr[i], F_arr[j]):
                        S[i].append(j)
                    elif dominates(F_arr[j], F_arr[i]):
                        n_dom[i] += 1
                if n_dom[i] == 0:
                    fronts.append([])
                    fronts[0].append(i)

            k = 1
            while fronts[k - 1]:
                next_front = []
                for i in fronts[k - 1]:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            next_front.append(j)
                if next_front:
                    fronts.append(next_front)
                    k += 1
                else:
                    break
            return fronts

        def crowding_distance(F_arr, indices):
            """Compute crowding distance for given indices in F_arr."""
            if len(indices) <= 2:
                return [float('inf')] * len(indices)
            n = len(indices)
            dist = [0.0] * n
            for m in range(n_obj):
                vals = [F_arr[idx, m] for idx in indices]
                order = np.argsort(vals)
                dist[order[0]] = float('inf')
                dist[order[-1]] = float('inf')
                val_range = vals[order[-1]] - vals[order[0]]
                if val_range == 0:
                    continue
                for i in range(1, n - 1):
                    dist[order[i]] += (vals[order[i + 1]] - vals[order[i - 1]]) / val_range
            return dist

        # Main loop
        while used < budget:
            # Check deadline
            if time.time() > deadline - 5:
                break

            # Non-dominated sort
            fronts = non_dominated_sort(F)

            # Build next generation
            next_X = []
            next_F = []
            used_in_gen = 0

            for front in fronts:
                if used_in_gen + len(front) <= pop_size:
                    for idx in front:
                        next_X.append(X[idx])
                        next_F.append(F[idx])
                    used_in_gen += len(front)
                else:
                    # Need to select from this front using crowding distance
                    remaining = pop_size - used_in_gen
                    cd = crowding_distance(F, front)
                    # Sort by crowding distance descending
                    order = np.argsort(cd)[::-1]
                    for idx in order[:remaining]:
                        next_X.append(X[front[idx]])
                        next_F.append(F[front[idx]])
                    used_in_gen += remaining
                    break

            # Convert to arrays
            next_X = np.array(next_X)
            next_F = np.array(next_F)

            # Generate offspring
            # Use binary tournament selection on the current population
            def tournament_select(pop_F, pop_X, n_selected):
                selected = []
                for _ in range(n_selected):
                    i = np.random.randint(len(pop_F))
                    j = np.random.randint(len(pop_F))
                    # Determine which is better using front and crowding
                    # Simple: just pick randomly for now, or use rank
                    # For simplicity, pick the one with lower sum of objectives
                    if np.sum(pop_F[i]) < np.sum(pop_F[j]):
                        selected.append(pop_X[i])
                    else:
                        selected.append(pop_X[j])
                return np.array(selected)

            # Crossover and mutation
            offspring_X = []
            for _ in range(pop_size // 2):
                # Select parents
                p1 = tournament_select(F, X, 1)[0]
                p2 = tournament_select(F, X, 1)[0]

                # Simulated binary crossover (SBX)
                child1 = np.zeros(dim)
                child2 = np.zeros(dim)
                for d in range(dim):
                    if np.random.random() < 0.5:
                        # Crossover
                        beta = np.random.random()
                        if beta <= 0.5:
                            beta_prime = (beta * 2) ** (1 / 21)
                        else:
                            beta_prime = ((1 - beta) * 2) ** (1 / 21)
                        c1 = 0.5 * ((1 + beta_prime) * p1[d] + (1 - beta_prime) * p2[d])
                        c2 = 0.5 * ((1 - beta_prime) * p1[d] + (1 + beta_prime) * p2[d])
                        child1[d] = c1
                        child2[d] = c2
                    else:
                        child1[d] = p1[d]
                        child2[d] = p2[d]

                # Mutation
                for d in range(dim):
                    if np.random.random() < 0.2:
                        child1[d] += 0.1 * (hi[d] - lo[d]) * np.random.standard_normal()
                    if np.random.random() < 0.2:
                        child2[d] += 0.1 * (hi[d] - lo[d]) * np.random.standard_normal()

                child1 = np.clip(child1, lo, hi)
                child2 = np.clip(child2, lo, hi)
                offspring_X.append(child1)
                offspring_X.append(child2)

            offspring_X = np.array(offspring_X)

            # Evaluate offspring
            offspring_F = []
            for x in offspring_X:
                if time.time() > deadline - 5:
                    break
                f = problem(x)
                offspring_F.append(f)
                used += 1

            if len(offspring_F) == 0:
                break

            offspring_F = np.array(offspring_F)

            # Combine and select
            combined_X = np.vstack([next_X, offspring_X])
            combined_F = np.vstack([next_F, offspring_F])

            # Non-dominated sort on combined
            combined_fronts = non_dominated_sort(combined_F)

            # Select best pop_size from combined
            X = []
            F = []
            for front in combined_fronts:
                if len(X) + len(front) <= pop_size:
                    for idx in front:
                        X.append(combined_X[idx])
                        F.append(combined_F[idx])
                else:
                    remaining = pop_size - len(X)
                    cd = crowding_distance(combined_F, front)
                    order = np.argsort(cd)[::-1]
                    for idx in order[:remaining]:
                        X.append(combined_X[front[idx]])
                        F.append(combined_F[front[idx]])
                    break

            X = np.array(X)
            F = np.array(F)

        return X
# EVOLVE-BLOCK-END
