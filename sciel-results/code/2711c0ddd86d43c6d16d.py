import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        start_time = time.time()
        deadline = start_time + 280

        # NSGA-II parameters
        pop_size = 40
        archive = []  # list of (x, f)
        ref_point = np.array([3.0, 5.0, 7.0])

        # Initialize population
        X = lo + (hi - lo) * np.random.random((pop_size, dim))
        F = np.array([problem(x) for x in X], dtype=np.float64)
        used = pop_size

        # Initialize archive
        for i in range(pop_size):
            archive.append((X[i].copy(), F[i].copy()))
        archive = self._trim_archive(archive, ref_point)

        while used < budget:
            if time.time() > deadline:
                break

            # Combine population and archive
            all_X = np.vstack([X, np.array([a[0] for a in archive])])
            all_F = np.vstack([F, np.array([a[1] for a in archive])])
            all_size = len(all_F)

            # Non-dominated sorting
            fronts = self._non_dominated_sort(all_F)

            # Select next generation
            next_X = []
            next_F = []
            for front in fronts:
                if len(next_F) + len(front) <= pop_size:
                    for idx in front:
                        next_X.append(all_X[idx])
                        next_F.append(all_F[idx])
                else:
                    # Need to select from this front using crowding distance
                    remaining = pop_size - len(next_F)
                    if remaining > 0:
                        front_F = all_F[front]
                        distances = self._crowding_distance(front_F)
                        sorted_indices = np.argsort(-distances)
                        for i in range(min(remaining, len(sorted_indices))):
                            idx = front[sorted_indices[i]]
                            next_X.append(all_X[idx])
                            next_F.append(all_F[idx])
                    break

            next_X = np.array(next_X)
            next_F = np.array(next_F)

            # Tournament selection for parents
            new_X = []
            new_F = []
            for _ in range(pop_size // 2):
                # Select two parents via tournament
                p1 = self._tournament_select(next_F, next_X, 2)
                p2 = self._tournament_select(next_F, next_X, 2)
                
                # SBX crossover
                child1, child2 = self._sbx_crossover(p1, p2, lo, hi, 20)
                new_X.append(child1)
                new_X.append(child2)

            new_X = np.array(new_X)

            # Polynomial mutation
            for i in range(len(new_X)):
                if np.random.random() < 0.9:
                    new_X[i] = self._polynomial_mutate(new_X[i], lo, hi, 12)

            # Evaluate new population
            new_F = np.array([problem(x) for x in new_X], dtype=np.float64)
            used += len(new_F)

            if used > budget:
                break

            # Update population
            X = new_X
            F = new_F

            # Update archive
            for i in range(len(X)):
                archive.append((X[i].copy(), F[i].copy()))
            archive = self._trim_archive(archive, ref_point)

        return X

    def _non_dominated_sort(self, F):
        """
        Vectorized non-dominated sorting.
        Returns list of fronts, each front is a list of indices.
        """
        n = len(F)
        domination_count = np.zeros(n, dtype=int)
        dominated_set = [[] for _ in range(n)]
        fronts = []

        for i in range(n):
            for j in range(i + 1, n):
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    domination_count[j] += 1
                    dominated_set[i].append(j)
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    domination_count[i] += 1
                    dominated_set[j].append(i)

        current_front = [i for i in range(n) if domination_count[i] == 0]
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

    def _crowding_distance(self, F):
        """
        Compute crowding distance for a set of points.
        """
        n = len(F)
        if n <= 2:
            return np.full(n, np.inf)

        distances = np.zeros(n)
        m = F.shape[1]

        for k in range(m):
            sorted_idx = np.argsort(F[:, k])
            distances[sorted_idx[0]] = np.inf
            distances[sorted_idx[-1]] = np.inf
            for i in range(1, n - 1):
                if F[sorted_idx[i + 1], k] > F[sorted_idx[i - 1], k]:
                    distances[sorted_idx[i]] += (F[sorted_idx[i + 1], k] - F[sorted_idx[i - 1], k]) / (F[sorted_idx[-1], k] - F[sorted_idx[0], k] + 1e-10)

        return distances

    def _tournament_select(self, F, X, k=2):
        """
        Tournament selection.
        """
        n = len(F)
        indices = np.random.choice(n, k, replace=False)
        # Find the non-dominated one
        best = indices[0]
        for i in indices[1:]:
            if np.all(F[i] <= F[best]) and np.any(F[i] < F[best]):
                best = i
        return X[best].copy()

    def _sbx_crossover(self, p1, p2, lo, hi, eta=20):
        """
        Simulated Binary Crossover.
        """
        dim = len(p1)
        c1 = np.zeros(dim)
        c2 = np.zeros(dim)
        for i in range(dim):
            if np.random.random() < 0.5:
                if abs(p1[i] - p2[i]) > 1e-14:
                    if np.random.random() < 0.5:
                        beta = (2 * np.random.random()) ** (1 / (eta + 1))
                    else:
                        beta = (1 / (2 * np.random.random())) ** (1 / (eta + 1))
                    c1[i] = 0.5 * (1 + beta) * p1[i] + 0.5 * (1 - beta) * p2[i]
                    c2[i] = 0.5 * (1 - beta) * p1[i] + 0.5 * (1 + beta) * p2[i]
                else:
                    c1[i] = p1[i]
                    c2[i] = p2[i]
            else:
                c1[i] = p1[i]
                c2[i] = p2[i]
        c1 = np.clip(c1, lo, hi)
        c2 = np.clip(c2, lo, hi)
        return c1, c2

    def _polynomial_mutate(self, x, lo, hi, eta=20):
        """
        Polynomial mutation.
        """
        dim = len(x)
        y = x.copy()
        for i in range(dim):
            if np.random.random() < 0.5:
                y[i] = np.random.random() * (hi[i] - lo[i]) + lo[i]
        return y

    def _trim_archive(self, archive, ref_point):
        """
        Trim archive to maintain non-dominance and diversity.
        """
        if not archive:
            return archive

        # Remove dominated solutions
        new_archive = []
        for i, (xi, fi) in enumerate(archive):
            dominated = False
            for j, (xj, fj) in enumerate(archive):
                if i == j:
                    continue
                if np.all(fj <= fi) and np.any(fj < fi):
                    dominated = True
                    break
            if not dominated:
                new_archive.append((xi, fi))

        # If archive is too large, remove points with smallest hypervolume contribution
        max_size = 50
        while len(new_archive) > max_size:
            if len(new_archive) <= 1:
                break

            # Compute hypervolume contribution (simplified)
            contributions = []
            for i, (xi, fi) in enumerate(new_archive):
                if np.any(fi >= ref_point):
                    contributions.append(0.0)
                    continue
                vol = 1.0
                for k in range(len(fi)):
                    vol *= (ref_point[k] - fi[k])
                contributions.append(vol)

            min_idx = np.argmin(contributions)
            new_archive.pop(min_idx)

        return new_archive

    def dominates(self, a, b):
        """
        Returns True if a dominates b (a is better than b in all objectives).
        """
        return np.all(a <= b) and np.any(a < b)
