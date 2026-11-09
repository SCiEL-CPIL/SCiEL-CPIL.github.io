"""Multi-objective optimizer for WFG7 with NSGA-II and hypervolume-based selection."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer with hypervolume-based parent selection."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        ref_point = np.array([3.0, 5.0, 7.0])

        # Population size
        pop_size = min(40, self.budget // 5)
        pop_size = max(pop_size, 10)

        # Initialize population
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        def fast_non_dominated_sort(F):
            """Fast non-dominated sorting. Returns list of fronts (list of indices)."""
            n = len(F)
            S = [[] for _ in range(n)]  # S[i] = set of indices dominated by i
            n_d = np.zeros(n, dtype=int)  # number of points that dominate i
            rank = np.zeros(n, dtype=int)
            fronts = []

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    # Check if i dominates j
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_d[i] += 1

            # First front
            front = [i for i in range(n) if n_d[i] == 0]
            for i in front:
                rank[i] = 0

            if front:
                fronts.append(front)

            k = 0
            while front:
                next_front = []
                for i in front:
                    for j in S[i]:
                        n_d[j] -= 1
                        if n_d[j] == 0:
                            rank[j] = k + 1
                            next_front.append(j)
                k += 1
                if next_front:
                    fronts.append(next_front)
                front = next_front

            return fronts

        def crowding_distance(F, indices):
            """Compute crowding distance for a set of indices."""
            if len(indices) <= 2:
                return np.full(len(indices), np.inf)

            m = self.n_obj
            n = len(indices)
            dist = np.zeros(n)

            for k in range(m):
                # Sort by objective k
                obj_vals = F[indices, k]
                order = np.argsort(obj_vals)
                sorted_indices = indices[order]
                sorted_vals = obj_vals[order]

                if sorted_vals[0] == sorted_vals[-1]:
                    continue

                dist[order[0]] = np.inf
                dist[order[-1]] = np.inf

                for i in range(1, n - 1):
                    dist[order[i]] += (sorted_vals[i + 1] - sorted_vals[i - 1]) / (
                        sorted_vals[-1] - sorted_vals[0]
                    )

            return dist

        def hypervolume_contribution(points, ref):
            """Compute exact hypervolume contribution for each point (for small n)."""
            n = len(points)
            if n == 0:
                return np.array([])
            if n == 1:
                vol = np.prod(np.maximum(0, ref - points[0]))
                return np.array([vol])

            # For 3 objectives, we can compute contributions more efficiently
            # Use a simple approach: for each point, compute the volume of the region
            # where it is the unique best point

            contribs = np.zeros(n)

            # First, filter to only non-dominated points
            non_dom_mask = np.ones(n, dtype=bool)
            for i in range(n):
                for j in range(n):
                    if i != j:
                        if np.all(points[j] <= points[i]) and np.any(points[j] < points[i]):
                            non_dom_mask[i] = False
                            break

            # For non-dominated points, compute approximate contribution
            # We'll use a grid-based approximation
            nd_points = points[non_dom_mask]
            if len(nd_points) == 0:
                return contribs

            # Use a simple Monte Carlo-like estimate
            # For each non-dominated point, estimate the volume it uniquely contributes
            for i in range(n):
                if not non_dom_mask[i]:
                    continue

                # The box from point i to ref
                box = np.maximum(0, ref - points[i])
                box_vol = np.prod(box)
                if box_vol <= 0:
                    continue

                # Estimate how much of this box is uniquely dominated by point i
                # Sample points in the box
                n_samples = 50
                sample_lo = points[i]
                sample_hi = ref
                samples = sample_lo + (sample_hi - sample_lo) * rng.random((n_samples, self.n_obj))

                # Count how many samples are NOT dominated by any other point
                unique_count = 0
                for s in samples:
                    unique = True
                    for j in range(n):
                        if i != j and non_dom_mask[j]:
                            if np.all(points[j] <= s) and np.any(points[j] < s):
                                unique = False
                                break
                    if unique:
                        unique_count += 1

                contribs[i] = box_vol * unique_count / n_samples

            return contribs

        def select_parents(F, X):
            """Select parents using hypervolume contribution on non-dominated front."""
            fronts = fast_non_dominated_sort(F)
            if not fronts:
                return len(F) // 2

            # Use the first front
            front = fronts[0]
            if len(front) == 1:
                return front[0]

            # Compute hypervolume contributions for the front
            front_F = F[front]
            contribs = hypervolume_contribution(front_F, ref_point)

            # Normalize
            total = contribs.sum()
            if total > 0:
                contribs = contribs / total
            else:
                contribs = np.ones(len(front)) / len(front)

            # Select one parent
            idx = rng.choice(len(front), p=contribs)
            return front[idx]

        # Main evolutionary loop
        while used < self.budget:
            if time.time() > deadline:
                break

            # Select parent
            parent_idx = select_parents(F, X)

            # SBX crossover (with a virtual second parent for diversity)
            # Use mutation primarily for exploration
            crossover_prob = 0.8
            if rng.random() < crossover_prob and len(X) > 1:
                # Select second parent
                parent2_idx = select_parents(F, X)
                if parent2_idx == parent_idx:
                    # Pick a different one
                    others = [i for i in range(len(X)) if i != parent_idx]
                    if others:
                        parent2_idx = others[rng.integers(len(others))]

                # SBX crossover
                eta = 20.0
                child = np.zeros(self.dim)
                for d in range(self.dim):
                    x1 = X[parent_idx, d]
                    x2 = X[parent2_idx, d]
                    if abs(x1 - x2) < 1e-10:
                        child[d] = x1
                    else:
                        u = rng.random()
                        if u <= 0.5:
                            beta = (2 * u) ** (1 / (eta + 1))
                        else:
                            beta = (2 * (1 - u)) ** (1 / (eta + 1))
                        child[d] = 0.5 * ((1 + beta) * x1 + (1 - beta) * x2)
            else:
                child = X[parent_idx].copy()

            # Polynomial mutation
            eta_mut = 20.0
            for d in range(self.dim):
                if rng.random() < 1.0 / self.dim:  # mutate each dimension with prob 1/dim
                    u = rng.random()
                    if u <= 0.5:
                        delta_q = (2 * u) ** (1 / (eta_mut + 1)) - 1
                    else:
                        delta_q = 1 - (2 * (1 - u)) ** (1 / (eta_mut + 1))
                    child[d] += delta_q * (hi[d] - lo[d])

            child = np.clip(child, lo, hi)

            # Evaluate
            fc = problem(child)
            used += 1

            # Add to population and sort
            X = np.vstack([X, child])
            F = np.vstack([F, fc.reshape(1, -1)])

            # Truncate to population size using NSGA-II selection
            if len(X) > pop_size:
                # Fast non-dominated sort
                fronts = fast_non_dominated_sort(F)

                # Build new population
                new_indices = []
                for front in fronts:
                    if len(new_indices) + len(front) <= pop_size:
                        new_indices.extend(front)
                    else:
                        # Need to select from this front using crowding distance
                        remaining = pop_size - len(new_indices)
                        if remaining <= 0:
                            break
                        cd = crowding_distance(F, front)
                        # Sort by crowding distance descending
                        order = np.argsort(-cd)
                        new_indices.extend(front[order[:remaining]])
                        break

                if len(new_indices) < pop_size:
                    # Fill with best remaining
                    used_indices = set(new_indices)
                    for i in range(len(F)):
                        if len(new_indices) >= pop_size:
                            break
                        if i not in used_indices:
                            new_indices.append(i)

                X = X[new_indices[:pop_size]]
                F = F[new_indices[:pop_size]]

        return X
# EVOLVE-BLOCK-END
