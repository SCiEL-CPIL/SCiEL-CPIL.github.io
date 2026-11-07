"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer for DTLZ4."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = 285.0  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        budget = self.budget

        # Population size
        pop_size = 60

        rng = np.random.default_rng(42)

        # Initialize population with random points
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X], dtype=np.float64)
        used = pop_size

        # Maintain an archive of non-dominated points for hypervolume
        archive_X = [X[i].copy() for i in range(pop_size)]
        archive_F = [F[i].copy() for i in range(pop_size)]

        def dominates(f1, f2):
            return np.all(f1 <= f2) and np.any(f1 < f2)

        def update_archive(new_x, new_f):
            nonlocal archive_X, archive_F
            # Remove dominated points from archive
            new_archive_X = []
            new_archive_F = []
            for ax, af in zip(archive_X, archive_F):
                if not dominates(new_f, af):
                    new_archive_X.append(ax)
                    new_archive_F.append(af)
            # Check if new point dominates any archive points
            for ax, af in zip(new_archive_X, new_archive_F):
                if dominates(new_f, af):
                    pass  # already removed above
            # Add new point if not dominated by any remaining archive point
            is_dominated = False
            for af in new_archive_F:
                if dominates(af, new_f):
                    is_dominated = True
                    break
            if not is_dominated:
                new_archive_X.append(new_x.copy())
                new_archive_F.append(new_f.copy())
            archive_X = new_archive_X
            archive_F = new_archive_F

        # Update archive with initial population
        for i in range(pop_size):
            update_archive(X[i], F[i])

        def fast_non_dominated_sort(F):
            """Fast non-dominated sort. Returns list of fronts (each is list of indices)."""
            n = len(F)
            S = [[] for _ in range(n)]  # domination sets
            n_dom = np.zeros(n, dtype=int)  # number of dominated
            fronts = []
            current_front = []
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_dom[i] += 1
                if n_dom[i] == 0:
                    current_front.append(i)
            if current_front:
                fronts.append(current_front)
            idx = 0
            while idx < len(fronts) and fronts[idx]:
                next_front = []
                for i in fronts[idx]:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            next_front.append(j)
                if next_front:
                    fronts.append(next_front)
                idx += 1
            return fronts

        def crowding_distance(F, indices):
            """Compute crowding distance for a set of indices."""
            n = len(indices)
            if n <= 2:
                return np.full(n, np.inf)
            cd = np.zeros(n, dtype=np.float64)
            F_sub = F[indices]
            for m in range(F_sub.shape[1]):
                min_val = F_sub[:, m].min()
                max_val = F_sub[:, m].max()
                if max_val - min_val < 1e-14:
                    continue
                sorted_idx = np.argsort(F_sub[:, m])
                cd[sorted_idx[0]] = np.inf
                cd[sorted_idx[-1]] = np.inf
                for i in range(1, n - 1):
                    cd[sorted_idx[i]] += (F_sub[sorted_idx[i + 1], m] - F_sub[sorted_idx[i - 1], m]) / (max_val - min_val)
            return cd

        def sbx_crossover(p1, p2, lo, hi):
            """Simulated binary crossover."""
            eta = 15.0
            child = np.zeros(dim)
            for d in range(dim):
                if rng.random() < 0.5:
                    u = rng.random()
                    if u <= 0.5:
                        beta = (2 * u) ** (1 / (eta + 1))
                    else:
                        beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))
                    child[d] = 0.5 * ((1 + beta) * p1[d] + (1 - beta) * p2[d])
                else:
                    u = rng.random()
                    if u <= 0.5:
                        beta = (2 * u) ** (1 / (eta + 1))
                    else:
                        beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))
                    child[d] = 0.5 * ((1 + beta) * p2[d] + (1 - beta) * p1[d])
            return np.clip(child, lo, hi)

        def polynomial_mutation(x, lo, hi):
            """Polynomial mutation."""
            eta = 20.0
            x = x.copy()
            for d in range(dim):
                if rng.random() < 1.0 / dim:
                    u = rng.random()
                    if u < 0.5:
                        delta = (2 * u) ** (1 / (eta + 1)) - 1
                    else:
                        delta = 1 - (2 * (1 - u)) ** (1 / (eta + 1))
                    x[d] += delta * (hi[d] - lo[d])
            return np.clip(x, lo, hi)

        while used < budget:
            # Check time
            if time.time() - t0 > deadline - 5:
                break

            # Fast non-dominated sort
            fronts = fast_non_dominated_sort(F)
            
            # Compute front ranks for each individual
            front_ranks = np.zeros(len(F), dtype=int)
            for fi, front in enumerate(fronts):
                for idx in front:
                    front_ranks[idx] = fi

            # Precompute crowding distances for each front
            front_cds = {}
            for fi, front in enumerate(fronts):
                if len(front) > 2:
                    front_cds[fi] = crowding_distance(F, front)

            # Select parents using tournament selection on fronts
            parents = []
            for _ in range(pop_size):
                # Pick two random individuals, select the one in a better front
                i1 = rng.integers(pop_size)
                i2 = rng.integers(pop_size)
                rank1 = front_ranks[i1]
                rank2 = front_ranks[i2]
                if rank1 < rank2:
                    parents.append(i1)
                elif rank2 < rank1:
                    parents.append(i2)
                else:
                    # Same front, use crowding distance
                    cd = front_cds.get(rank1, np.full(len(fronts[rank1]), np.inf))
                    front = fronts[rank1]
                    pos1 = front.index(i1)
                    pos2 = front.index(i2)
                    if cd[pos1] > cd[pos2]:
                        parents.append(i1)
                    else:
                        parents.append(i2)

            # Generate offspring
            children = []
            child_F = []
            n_children = min(pop_size, budget - used)
            for _ in range(n_children):
                p1 = parents[rng.integers(pop_size)]
                p2 = parents[rng.integers(pop_size)]
                
                # Simulated binary crossover
                if rng.random() < 0.9:
                    child = sbx_crossover(X[p1], X[p2], lo, hi)
                else:
                    child = X[p1].copy()

                # Polynomial mutation
                child = polynomial_mutation(child, lo, hi)

                children.append(child)
                child_F.append(problem(child))
                used += 1
                if used >= budget:
                    break
                if time.time() - t0 > deadline - 5:
                    break

            if used >= budget or time.time() - t0 > deadline - 5:
                break

            child_F = np.array(child_F, dtype=np.float64)
            children = np.array(children, dtype=np.float64)

            # Update archive with children
            for i in range(len(children)):
                update_archive(children[i], child_F[i])

            # Combine parents and children
            X_all = np.vstack([X, children])
            F_all = np.vstack([F, child_F])

            # Non-dominated sort on combined population
            fronts = fast_non_dominated_sort(F_all)

            # Precompute crowding distances for selection
            front_cds_sel = {}
            for fi, front in enumerate(fronts):
                if len(front) > 2:
                    front_cds_sel[fi] = crowding_distance(F_all, front)

            # Select next generation
            new_X = []
            new_F = []
            for fi, front in enumerate(fronts):
                if len(new_X) + len(front) <= pop_size:
                    for idx in front:
                        new_X.append(X_all[idx])
                        new_F.append(F_all[idx])
                else:
                    # Need to select from this front based on crowding distance
                    remaining = pop_size - len(new_X)
                    cd = front_cds_sel.get(fi, np.full(len(front), np.inf))
                    sorted_idx = np.argsort(-cd)
                    for i in sorted_idx[:remaining]:
                        idx = front[i]
                        new_X.append(X_all[idx])
                        new_F.append(F_all[idx])
                    break

            X = np.array(new_X, dtype=np.float64)
            F = np.array(new_F, dtype=np.float64)

        return X
# EVOLVE-BLOCK-END
