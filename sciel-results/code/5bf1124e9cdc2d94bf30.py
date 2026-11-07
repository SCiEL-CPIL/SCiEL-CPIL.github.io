"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """NSGA-II style multi-objective optimizer with time guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _fast_non_dominated(self, F):
        """Return list of fronts (each a list of indices) using non-dominated sorting."""
        n = len(F)
        S = [[] for _ in range(n)]
        nd = [0] * n
        fronts = []
        front_idx = [0] * n

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    nd[i] += 1
            if nd[i] == 0:
                front_idx[i] = 0
                fronts.append([])

        # Re-initialize fronts as proper list
        fronts = []
        current_front = []
        for i in range(n):
            if nd[i] == 0:
                front_idx[i] = 0
                current_front.append(i)
        if current_front:
            fronts.append(current_front)

        k = 0
        while k < len(fronts):
            next_front = []
            for i in fronts[k]:
                for j in S[i]:
                    nd[j] -= 1
                    if nd[j] == 0:
                        front_idx[j] = k + 1
                        next_front.append(j)
            if next_front:
                fronts.append(next_front)
            k += 1

        return fronts

    def _crowding_distance(self, F, indices):
        """Compute crowding distance for a set of indices."""
        n = len(indices)
        if n <= 2:
            return [float('inf')] * n

        cd = [0.0] * n
        m = self.n_obj

        for k in range(m):
            # Sort by objective k
            sorted_idx = sorted(range(n), key=lambda i: F[indices[i]][k])
            cd[sorted_idx[0]] = float('inf')
            cd[sorted_idx[-1]] = float('inf')

            f_min = F[indices[sorted_idx[0]]][k]
            f_max = F[indices[sorted_idx[-1]]][k]
            f_range = f_max - f_min
            if f_range == 0:
                continue

            for i in range(1, n - 1):
                cd[sorted_idx[i]] += (F[indices[sorted_idx[i + 1]]][k] - F[indices[sorted_idx[i - 1]]][k]) / f_range

        return cd

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size
        pop_size = min(100, max(20, budget // 4))

        # Initial random population
        X = lo + (hi - lo) * np.random.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Reference point for hypervolume (from task)
        ref_point = np.array([1705.0, 12.0, 0.3])

        # Track best Pareto front found
        all_F = F.copy()
        all_X = X.copy()

        while used < budget:
            # Check time
            if time.time() > deadline:
                break

            # Non-dominated sorting
            fronts = self._fast_non_dominated(F)

            # Build next generation
            next_X = []
            next_F = []
            next_used = 0

            for front in fronts:
                if len(next_X) + len(front) <= pop_size:
                    # Add all from this front
                    for idx in front:
                        next_X.append(X[idx])
                        next_F.append(F[idx])
                    next_used += len(front)
                else:
                    # Need to select from this front using crowding distance
                    remaining = pop_size - len(next_X)
                    cd = self._crowding_distance(F, front)
                    # Sort by crowding distance (descending)
                    sorted_front = sorted(front, key=lambda i: cd[front.index(i)], reverse=True)
                    for idx in sorted_front[:remaining]:
                        next_X.append(X[idx])
                        next_F.append(F[idx])
                    next_used += remaining
                    break

            # Create children using binary tournament selection and SBX crossover
            children_X = []
            children_F = []

            while len(children_X) < pop_size - next_used:
                # Check time
                if time.time() > deadline:
                    break

                # Tournament selection
                def tournament():
                    i1 = np.random.randint(len(fronts[0]))
                    i2 = np.random.randint(len(fronts[0]))
                    # Select from first front for diversity, or use rank
                    rank1 = front_idx[i1] if i1 < len(front_idx) else 0
                    rank2 = front_idx[i2] if i2 < len(front_idx) else 0
                    if rank1 < rank2:
                        return i1
                    elif rank2 < rank1:
                        return i2
                    else:
                        # Use crowding distance
                        return i1 if np.random.random() < 0.5 else i2

                # Use simpler selection: pick from non-dominated front
                non_dom_indices = fronts[0]
                p1 = non_dom_indices[np.random.randint(len(non_dom_indices))]
                p2 = non_dom_indices[np.random.randint(len(non_dom_indices))]

                # SBX crossover
                if np.random.random() < 0.9:
                    eta = 20.0
                    child1 = np.zeros(dim)
                    child2 = np.zeros(dim)
                    for j in range(dim):
                        u = np.random.random()
                        if u <= 0.5:
                            beta = (2 * u) ** (1 / (eta + 1))
                        else:
                            beta = (2 * (1 - u)) ** (1 / (eta + 1))
                        child1[j] = 0.5 * ((1 + beta) * X[p1, j] + (1 - beta) * X[p2, j])
                        child2[j] = 0.5 * ((1 - beta) * X[p1, j] + (1 + beta) * X[p2, j])

                    # Polynomial mutation
                    if np.random.random() < 0.5:
                        eta_m = 20.0
                        for j in range(dim):
                            if np.random.random() < 0.5:
                                u = np.random.random()
                                if u <= 0.5:
                                    delta_q = (2 * u) ** (1 / (eta_m + 1)) - 1
                                else:
                                    delta_q = 1 - (2 * (1 - u)) ** (1 / (eta_m + 1))
                                child1[j] += delta_q * (hi[j] - lo[j])
                        for j in range(dim):
                            if np.random.random() < 0.5:
                                u = np.random.random()
                                if u <= 0.5:
                                    delta_q = (2 * u) ** (1 / (eta_m + 1)) - 1
                                else:
                                    delta_q = 1 - (2 * (1 - u)) ** (1 / (eta_m + 1))
                                child2[j] += delta_q * (hi[j] - lo[j])

                    child1 = np.clip(child1, lo, hi)
                    child2 = np.clip(child2, lo, hi)

                    if len(children_X) < pop_size - next_used:
                        fc1 = problem(child1)
                        children_X.append(child1)
                        children_F.append(fc1)
                        used += 1

                    if len(children_X) < pop_size - next_used and used < budget:
                        fc2 = problem(child2)
                        children_X.append(child2)
                        children_F.append(fc2)
                        used += 1
                else:
                    # Just copy
                    child = X[p1].copy()
                    fc = problem(child)
                    children_X.append(child)
                    children_F.append(fc)
                    used += 1

            if not children_X:
                break

            # Combine and select next generation
            combined_X = next_X + children_X
            combined_F = np.array(next_F + children_F)

            # Non-dominated sort combined
            combined_fronts = self._fast_non_dominated(combined_F)

            new_X = []
            new_F = []
            for front in combined_fronts:
                if len(new_X) + len(front) <= pop_size:
                    for idx in front:
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                else:
                    remaining = pop_size - len(new_X)
                    cd = self._crowding_distance(combined_F, front)
                    sorted_front = sorted(front, key=lambda i: cd[front.index(i)], reverse=True)
                    for idx in sorted_front[:remaining]:
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                    break

            X = np.array(new_X)
            F = np.array(new_F)

            # Update all recorded points
            all_F = np.vstack([all_F, F])
            all_X = np.vstack([all_X, X])

            # Early stop if no improvement in last few generations
            if used > budget * 0.8:
                break

        return X
# EVOLVE-BLOCK-END
