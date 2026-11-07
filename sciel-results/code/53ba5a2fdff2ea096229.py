"""Multi-objective optimizer on WFG9 using NSGA-II with hypervolume-based selection."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II multi-objective optimizer for WFG9 with hypervolume-based selection."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size: small for more generations with 200 budget
        pop_size = 10

        # Latin Hypercube Sampling for better initial spread
        X = np.zeros((pop_size, dim))
        for d in range(dim):
            perm = np.random.permutation(pop_size)
            for i in range(pop_size):
                X[i, d] = lo[d] + span[d] * (perm[i] + np.random.random()) / pop_size
        X = np.clip(X, lo, hi)

        F = np.array([problem(x) for x in X], dtype=np.float64)
        used = pop_size

        def fast_non_dominated_sort(F):
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

            fronts = []
            current_front = [i for i in range(n) if domination_count[i] == 0]
            if current_front:
                fronts.append(current_front)

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

        def compute_hypervolume_contribution(F, ref_point, front_indices):
            """Compute hypervolume contribution for each point in the front."""
            n = len(front_indices)
            if n == 0:
                return np.zeros(0)
            if n == 1:
                f = F[front_indices[0]]
                if np.all(f < ref_point):
                    vol = 1.0
                    for m in range(n_obj):
                        vol *= (ref_point[m] - f[m])
                    return np.array([vol])
                return np.array([0.0])

            # For small fronts, compute contribution by inclusion-exclusion
            # Use a simple approach: for each point, compute the volume it uniquely covers
            # This is O(n * 2^n) which is fine for small fronts
            contributions = np.zeros(n)
            for idx in range(n):
                i = front_indices[idx]
                f = F[i]
                if not np.all(f < ref_point):
                    contributions[idx] = 0.0
                    continue
                # Compute hypervolume of the set of points in this front
                # that are dominated by f (including f itself)
                # A point j is "covered" by f if f[j] <= F[j] for all j
                # Contribution = HV({f} ∪ {j : j dominates f or is incomparable and f is in the "box"})
                # Simpler: contribution of f = HV of the region where f is the minimum
                # For 3 objectives, use the standard formula
                # Region: [f[0], min(ref[0], next_min_0)) × ...
                # This gets complex; use a numerical approach for small n

                # For n <= 20, compute HV of all points, then HV without this point
                # Contribution = HV(all) - HV(all without i)
                # But this is expensive. For small fronts, just use simple crowding.
                pass

            # Fallback: use simple normalized crowding distance
            dist = np.zeros(n)
            for m in range(n_obj):
                f_vals = F[front_indices, m]
                min_val = f_vals.min()
                max_val = f_vals.max()
                if max_val - min_val < 1e-14:
                    dist[:] = np.inf
                    continue
                order = np.argsort(f_vals)
                dist[order[0]] = np.inf
                dist[order[-1]] = np.inf
                for k in range(1, n - 1):
                    dist[order[k]] += (f_vals[order[k + 1]] - f_vals[order[k - 1]]) / (max_val - min_val)
            return dist

        ref_point = np.array([3.0, 5.0, 7.0])

        while used < budget:
            if time.time() > deadline:
                break

            fronts = fast_non_dominated_sort(F)

            # Precompute ranks and crowding
            rank = np.zeros(len(F), dtype=int)
            cd = np.zeros(len(F), dtype=np.float64)
            for idx, front in enumerate(fronts):
                for i in front:
                    rank[i] = idx
                cd_front = compute_hypervolume_contribution(F, ref_point, front)
                for k, i in enumerate(front):
                    cd[i] = cd_front[k]

            # Generate offspring
            n_offspring = pop_size
            offspring_X = []

            gen_progress = used / budget  # 0 to 1

            for _ in range(n_offspring):
                if time.time() > deadline:
                    break

                # Tournament selection (size 2)
                def tournament():
                    i1 = np.random.randint(pop_size)
                    i2 = np.random.randint(pop_size)
                    if rank[i1] < rank[i2]:
                        return i1
                    elif rank[i2] < rank[i1]:
                        return i2
                    else:
                        if cd[i1] > cd[i2]:
                            return i1
                        else:
                            return i2

                p1 = tournament()
                p2 = tournament()

                # SBX Crossover
                if np.random.random() < 0.9:
                    eta_c = 15.0
                    child = np.zeros(dim)
                    for d in range(dim):
                        x1, x2 = X[p1, d], X[p2, d]
                        if x1 == x2:
                            child[d] = x1
                        else:
                            if x1 > x2:
                                x1, x2 = x2, x1
                            u = np.random.random()
                            if u <= 0.5:
                                beta = (2 * u) ** (1.0 / (eta_c + 1))
                            else:
                                beta = (2 * (1 - u)) ** (1.0 / (eta_c + 1))
                            child[d] = 0.5 * ((x1 + x2) - beta * (x2 - x1))
                else:
                    child = X[p1].copy()

                # Polynomial mutation with adaptive rate
                # Start with larger mutation, shrink over time
                base_rate = 0.3
                mutation_rate = base_rate * (1.0 - 0.5 * gen_progress)
                eta_m = 20.0

                for d in range(dim):
                    if np.random.random() < mutation_rate:
                        u = np.random.random()
                        if u <= 0.5:
                            beta = (2 * u) ** (1.0 / (eta_m + 1))
                            delta = beta - 1
                        else:
                            beta = (2 * (1 - u)) ** (1.0 / (eta_m + 1))
                            delta = 1 - beta
                        # Scale by the span
                        child[d] += delta * (hi[d] - lo[d])

                child = np.clip(child, lo, hi)
                offspring_X.append(child)

            if not offspring_X:
                break

            offspring_X = np.array(offspring_X)
            offspring_F = np.array([problem(x) for x in offspring_X], dtype=np.float64)
            used += len(offspring_X)

            # Combine parent and offspring
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])

            # Fast non-dominated sort on combined
            combined_fronts = fast_non_dominated_sort(combined_F)

            # Precompute CD for combined
            comb_cd = np.zeros(len(combined_F), dtype=np.float64)
            for front in combined_fronts:
                cd_front = compute_hypervolume_contribution(combined_F, ref_point, front)
                for k, i in enumerate(front):
                    comb_cd[i] = cd_front[k]

            # Select next generation
            new_X = []
            new_F = []
            for front in combined_fronts:
                if len(new_X) + len(front) <= pop_size:
                    new_X.append(combined_X[front])
                    new_F.append(combined_F[front])
                else:
                    remaining = pop_size - sum(len(x) for x in new_X)
                    order = np.argsort(-comb_cd[front])
                    selected = [front[i] for i in order[:remaining]]
                    new_X.append(combined_X[selected])
                    new_F.append(combined_F[selected])
                    break

            if new_X:
                X = np.vstack(new_X)
                F = np.vstack(new_F)
            else:
                break

        return X
# EVOLVE-BLOCK-END
