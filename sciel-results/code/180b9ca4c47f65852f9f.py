"""Multi-objective optimizer for WFG9: fast NSGA-II + local refinement."""
import numpy as np
import time


class Optimizer:
    """Compact, fast NSGA-II with vectorised crossover and a local polish phase."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 290.0  # 40 s safety margin before 330 s kill

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(42)

        # Population size: small pop, many generations for good coverage
        pop_size = 40

        # Initial population
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Vectorised dominance check: returns (n, n) boolean matrix where
        # dom[i,j] is True if i dominates j
        def dominance_matrix(F):
            n = len(F)
            # F[i] <= F[j] element-wise
            le = F[:, np.newaxis, :] <= F[np.newaxis, :, :]
            # F[i] < F[j] element-wise
            lt = F[:, np.newaxis, :] < F[np.newaxis, :, :]
            # i dominates j: all objectives <= and at least one <
            return np.all(le, axis=2) & np.any(lt, axis=2)

        def fast_non_dominated_sort(F):
            """Return list of fronts, each a list of indices. Vectorised."""
            n = len(F)
            dom = dominance_matrix(F)
            n_dominating = dom.sum(axis=0)  # number of individuals dominating each
            fronts = []
            current_front = np.where(n_dominating == 0)[0].tolist()
            fronts.append(current_front)

            remaining = set(range(n)) - set(current_front)
            while remaining:
                next_front = []
                for i in current_front:
                    for j in remaining:
                        if dom[i, j]:
                            n_dominating[j] -= 1
                            if n_dominating[j] == 0:
                                next_front.append(j)
                if not next_front:
                    break
                fronts.append(next_front)
                remaining -= set(next_front)
                current_front = next_front

            return fronts

        def crowding_distance(F, front_indices):
            """Compute crowding distance for individuals in a front."""
            n = len(front_indices)
            if n <= 2:
                return np.full(n, np.inf)

            dist = np.zeros(n)
            for m in range(n_obj):
                vals = F[front_indices, m]
                min_val, max_val = vals.min(), vals.max()
                if max_val - min_val < 1e-15:
                    continue
                sorted_idx = np.argsort(vals)
                dist[sorted_idx[0]] = np.inf
                dist[sorted_idx[-1]] = np.inf
                for i in range(1, n - 1):
                    dist[sorted_idx[i]] += (vals[sorted_idx[i + 1]] - vals[sorted_idx[i - 1]]) / (max_val - min_val)
            return dist

        def sort_by_dominance_and_crowding(F):
            """Return indices sorted by non-dominated rank then crowding distance."""
            fronts = fast_non_dominated_sort(F)
            order = []
            for front in fronts:
                if not front:
                    continue
                cd = crowding_distance(F, front)
                front_sorted = sorted(front, key=lambda i: cd[front.index(i)], reverse=True)
                order.extend(front_sorted)
            return order

        # Precompute ranks and crowding for tournament selection
        def compute_ranks_and_crowding(F):
            fronts = fast_non_dominated_sort(F)
            rank_map = {}
            cd_map = {}
            for rank, front in enumerate(fronts):
                cd = crowding_distance(F, front)
                for idx in front:
                    rank_map[idx] = rank
                    cd_map[idx] = cd[front.index(idx)]
            return rank_map, cd_map

        # Main evolution loop
        while used < budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Compute ranks and crowding once per generation
            rank_map, cd_map = compute_ranks_and_crowding(F)

            # Select parents using binary tournament
            parents = []
            for _ in range(pop_size):
                i1 = rng.integers(pop_size)
                i2 = rng.integers(pop_size)
                r1 = rank_map.get(i1, 999)
                r2 = rank_map.get(i2, 999)
                if r1 < r2:
                    parents.append(i1)
                elif r2 < r1:
                    parents.append(i2)
                else:
                    parents.append(i1 if cd_map.get(i1, 0) >= cd_map.get(i2, 0) else i2)

            # Create offspring with vectorised SBX crossover
            max_children = min(pop_size, budget - used)
            children_X = np.zeros((max_children, dim))
            children_F = np.zeros((max_children, n_obj))

            # Adaptive mutation step: start large, shrink over time
            progress = used / max(1, budget)
            base_step = 0.25 * (1 - progress * 0.6)  # shrinks from 0.25 to 0.10

            for i in range(max_children):
                p1 = parents[i]
                p2 = parents[(i + 1) % pop_size]

                # Simulated binary crossover (SBX) - vectorised
                if rng.random() < 0.9:
                    x1, x2 = X[p1], X[p2]
                    u = rng.random(dim)
                    # Compute beta for each dimension
                    beta = np.where(
                        u <= 0.5,
                        (2 * u) ** (1 / 3.0),
                        (1 / (2 * (1 - u))) ** (1 / 3.0)
                    )
                    # For each dimension, randomly choose which parent is "first"
                    swap = rng.random(dim) < 0.5
                    x1_s = np.where(swap, x2, x1)
                    x2_s = np.where(swap, x1, x2)
                    child = 0.5 * ((1 + beta) * x1_s + (1 - beta) * x2_s)
                else:
                    child = X[p1].copy()

                # Mutation
                if rng.random() < 0.9:
                    step = base_step * rng.standard_normal(dim)
                    child = child + step * (hi - lo) / 2

                child = np.clip(child, lo, hi)
                children_X[i] = child

            # Evaluate children (respect budget)
            for i in range(max_children):
                children_F[i] = problem(children_X[i])
                used += 1
                if time.time() > deadline:
                    break

            # Combine parent and child populations
            X_comb = np.vstack([X, children_X])
            F_comb = np.vstack([F, children_F])

            # Select next generation
            order_comb = sort_by_dominance_and_crowding(F_comb)
            X = X_comb[order_comb[:pop_size]]
            F = F_comb[order_comb[:pop_size]]

        # Local refinement phase: use remaining budget to polish the Pareto front
        remaining = budget - used
        if remaining > 0 and time.time() < deadline - 10:
            # Get the current non-dominated front
            fronts = fast_non_dominated_sort(F)
            if fronts:
                nd_front = fronts[0]
                # For each point in the non-dominated front, do local refinement
                per_point = max(1, remaining // max(1, len(nd_front)))
                for idx in nd_front:
                    if used >= budget or time.time() > deadline - 5:
                        break
                    # Try small perturbations around this point
                    for trial in range(per_point):
                        if used >= budget or time.time() > deadline - 5:
                            break
                        # Random direction, small step
                        direction = rng.standard_normal(dim)
                        direction /= (np.linalg.norm(direction) + 1e-15)
                        step_size = 0.05 * rng.random()
                        new_x = X[idx] + direction * step_size * (hi - lo) / 2
                        new_x = np.clip(new_x, lo, hi)
                        new_f = problem(new_x)
                        used += 1
                        # If new point is better or non-dominated, replace
                        dominated_by_any = False
                        for j in range(len(F)):
                            if np.all(F[j] <= new_f) and np.any(F[j] < new_f):
                                dominated_by_any = True
                                break
                        if not dominated_by_any:
                            # Check if it dominates any existing point
                            for j in range(len(F)):
                                if np.all(new_f <= F[j]) and np.any(new_f < F[j]):
                                    X[idx] = new_x
                                    F[idx] = new_f
                                    break
                            else:
                                # It's non-dominated but doesn't dominate anything
                                # Only replace if it's better in at least one objective
                                # and not worse in others (simple improvement check)
                                if np.any(new_f < F[idx]) and np.all(new_f <= F[idx] + 1e-10):
                                    X[idx] = new_x
                                    F[idx] = new_f

        # Return the best non-dominated set found
        return X
