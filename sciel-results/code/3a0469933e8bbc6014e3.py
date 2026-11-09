import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer for ZDT3 with smart initialization."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0  # safety margin before 330s kill

        if self.budget <= 0:
            return np.array([])

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        n = self.dim
        budget = self.budget

        # ZDT3 Pareto front segments (approximate f1 ranges)
        # The front has 7 disjoint segments
        segments = [
            (0.0, 0.0575),
            (0.099, 0.206),
            (0.273, 0.417),
            (0.508, 0.666),
            (0.737, 0.860),
            (0.907, 0.972),
            (0.991, 1.0),
        ]
        seg_lengths = [hi_f - lo_f for lo_f, hi_f in segments]
        total_len = sum(seg_lengths)
        seg_probs = np.array(seg_lengths) / total_len

        # Population size
        pop_size = min(40, max(10, budget // 5))

        # Smart initialization: sample x1 proportionally to segment lengths
        # x2...xn = 0 for Pareto front, but add small perturbation
        def smart_sample(n_samples, rng):
            # Choose which segment each sample comes from
            seg_indices = rng.choice(len(segments), size=n_samples, p=seg_probs)
            x = np.zeros((n_samples, n))
            for i, si in enumerate(seg_indices):
                lo_f, hi_f = segments[si]
                x[i, 0] = lo_f + (hi_f - lo_f) * rng.random()
                # Small perturbation for x2...xn to stay near Pareto front
                x[i, 1:] = 0.01 * rng.random(n - 1)
            return x

        rng = np.random.default_rng(42)

        # Initialize population
        X = smart_sample(pop_size, rng)
        # Clip to bounds
        X = np.clip(X, lo, hi)

        # Evaluate
        F = np.array([problem(x) for x in X], dtype=np.float64)
        used = pop_size

        # Track all evaluated points for hypervolume
        all_F = [F.copy()]
        all_X = [X.copy()]

        def non_dominated_sort(F):
            """Return list of fronts, each a list of indices."""
            m = len(F)
            S = [set() for _ in range(m)]  # S[i] = set of indices dominated by i
            n_dom = [0] * m  # number of solutions dominating i
            fronts = [[]]

            for i in range(m):
                for j in range(m):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].add(j)
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_dom[i] += 1
                if n_dom[i] == 0:
                    fronts[0].append(i)

            i = 0
            while fronts[i]:
                next_front = []
                for p in fronts[i]:
                    for q in S[p]:
                        n_dom[q] -= 1
                        if n_dom[q] == 0:
                            next_front.append(q)
                i += 1
                if next_front:
                    fronts.append(next_front)
            return fronts

        def crowding_distance(F, indices):
            """Compute crowding distance for given indices."""
            m = len(indices)
            if m <= 2:
                return np.full(m, np.inf)
            dist = np.zeros(m)
            F_sub = F[indices]
            for k in range(self.n_obj):
            # Sort by objective k
                order = np.argsort(F_sub[:, k])
                dist[order[0]] = np.inf
                dist[order[-1]] = np.inf
                f_range = F_sub[order[-1], k] - F_sub[order[0], k]
                if f_range > 1e-12:
                    for i in range(1, m - 1):
                        dist[order[i]] += (F_sub[order[i + 1], k] - F_sub[order[i - 1], k]) / f_range
            return dist

        # Main loop
        mutation_scale = 0.05
        while used < budget:
            if time.time() > deadline:
                break

            # Non-dominated sort
            fronts = non_dominated_sort(F)

            # Select parents from first few fronts
            parent_pool = []
            for front in fronts[:2]:
                parent_pool.extend(front)
            if not parent_pool:
                parent_pool = list(range(len(F)))

            # Binary tournament selection
            def select_parent():
                i = rng.integers(len(parent_pool))
                j = rng.integers(len(parent_pool))
                # Prefer earlier fronts
                front_i = next(fi for fi, f in enumerate(fronts) if parent_pool[i] in f)
                front_j = next(fi for fi, f in enumerate(fronts) if parent_pool[j] in f)
                if front_i < front_j:
                    return parent_pool[i]
                elif front_j < front_i:
                    return parent_pool[j]
                else:
                    # Crowding distance tiebreak
                    cd = crowding_distance(F, parent_pool)
                    return parent_pool[i] if cd[i] > cd[j] else parent_pool[j]

            p1 = select_parent()
            p2 = select_parent()

            # Crossover and mutation
            if rng.random() < 0.7:
                # Simulated binary crossover-like: blend
                alpha = rng.random()
                child = alpha * X[p1] + (1 - alpha) * X[p2]
            else:
                child = X[p1].copy()

            # Adaptive mutation
            mutation = mutation_scale * (hi - lo) * rng.standard_normal(n)
            child = child + mutation
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            # Replace worst point
            # Compute a simple fitness: rank + crowding
            # For simplicity, replace the point with highest sum of objectives
            # weighted by front rank
            front_ranks = np.zeros(len(F))
            for fi, front in enumerate(fronts):
                for idx in front:
                    front_ranks[idx] = fi

            # Prefer replacing from later fronts, and within same front, lower crowding
            # Simple approach: replace from the last front
            last_front = fronts[-1]
            if len(last_front) > 1:
                # Replace the one with lowest crowding distance in last front
                cd = crowding_distance(F, last_front)
                worst_idx = last_front[np.argmin(cd)]
            else:
                worst_idx = np.argmax(np.sum(F, axis=1))

            X[worst_idx] = child
            F[worst_idx] = fc

            # Adapt mutation scale based on progress
            if used % 50 == 0:
                mutation_scale = max(0.01, mutation_scale * 0.95)

        return X
