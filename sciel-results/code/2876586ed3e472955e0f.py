"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for WFG4 using NSGA-II with adaptive mutation."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 290.0  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        mid = (lo + hi) / 2.0

        # Track all evaluated points and their objectives
        all_X = []
        all_F = []
        used = 0

        def is_deadline():
            return time.time() > deadline

        def try_eval(x):
            """Evaluate x, increment used counter. Returns f or None if over budget."""
            nonlocal used
            if used >= self.budget or time.time() > deadline:
                return None
            x_clipped = np.clip(x, lo, hi)
            f = problem(x_clipped)
            used += 1
            return f

        def non_dominated_sort(F):
            """Perform non-dominated sorting. Returns list of fronts."""
            n = len(F)
            if n == 0:
                return []

            # S[i] = set of solutions dominated by i
            S = [[] for _ in range(n)]
            # n_dom[i] = number of solutions that dominate i
            n_dom = np.zeros(n, dtype=int)
            # rank[i] = front index
            rank = np.full(n, -1, dtype=int)

            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_dom[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].append(i)
                        n_dom[i] += 1

            fronts = []
            current_front = [i for i in range(n) if n_dom[i] == 0]
            front_idx = 0

            while current_front:
                fronts.append(current_front)
                next_front = []
                for i in current_front:
                    rank[i] = front_idx
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            next_front.append(j)
                current_front = next_front
                front_idx += 1

            return fronts

        def crowding_distance(F, front_indices):
            """Compute crowding distance for a front."""
            n = len(front_indices)
            if n <= 2:
                return np.full(n, np.inf)

            distances = np.zeros(n)
            F_front = F[front_indices]

            for m in range(F.shape[1]):
                obj_vals = F_front[:, m]
                min_val = np.min(obj_vals)
                max_val = np.max(obj_vals)
                if max_val - min_val < 1e-12:
                    continue

                sorted_idx = np.argsort(obj_vals)
                distances[sorted_idx[0]] = np.inf
                distances[sorted_idx[-1]] = np.inf

                for i in range(1, n - 1):
                    distances[sorted_idx[i]] += (obj_vals[sorted_idx[i + 1]] - obj_vals[sorted_idx[i - 1]]) / (max_val - min_val)

            return distances

        def tournament_select(pop_F, pop_rank, pop_cd, k=2):
            """Tournament selection based on rank and crowding distance."""
            n = len(pop_F)
            if n == 0:
                return None
            if n == 1:
                return 0

            idx1 = np.random.randint(n)
            idx2 = np.random.randint(n)

            if pop_rank[idx1] != pop_rank[idx2]:
                return idx1 if pop_rank[idx1] < pop_rank[idx2] else idx2
            else:
                return idx1 if pop_cd[idx1] > pop_cd[idx2] else idx2

        def crossover(x1, x2, rate=0.9):
            """Simulated binary crossover."""
            if np.random.rand() > rate:
                return x1.copy(), x2.copy()

            c1 = x1.copy()
            c2 = x2.copy()
            for d in range(self.dim):
                if np.random.rand() < 0.5:
                    beta = np.random.rand()
                    if beta <= 0.5:
                        alpha = 2 * beta
                        beta_q = (1 / (1 + alpha)) ** (1 / 2)
                    else:
                        alpha = 2 * (1 - beta)
                        beta_q = (1 / (1 + alpha)) ** (1 / 2)

                    x1_val = x1[d]
                    x2_val = x2[d]
                    c1[d] = 0.5 * ((1 + beta_q) * x1_val + (1 - beta_q) * x2_val)
                    c2[d] = 0.5 * ((1 - beta_q) * x1_val + (1 + beta_q) * x2_val)

            return c1, c2

        def mutate(x, rate=0.1, sigma=0.1):
            """Polynomial mutation."""
            x_mut = x.copy()
            for d in range(self.dim):
                if np.random.rand() < rate:
                    u = np.random.rand()
                    if u < 0.5:
                        delta = (2 * u) ** (1 / (2 + 2)) - 1
                    else:
                        delta = 1 - (2 * (1 - u)) ** (1 / (2 + 2))
                    x_mut[d] += delta * sigma * span[d]
            return np.clip(x_mut, lo, hi)

        # Initial population: diverse sampling
        pop_size = min(40, self.budget)
        pop_size = max(pop_size, 10)

        # Initialize with a mix of random and structured points
        init_X = np.random.uniform(lo, hi, size=(pop_size, self.dim))
        # Add some structured points for better coverage
        n_structured = min(pop_size // 4, 10)
        for i in range(n_structured):
            x_struct = np.random.uniform(lo, hi)
            # Randomly set some dimensions to bounds
            for d in range(self.dim):
                if np.random.rand() < 0.3:
                    x_struct[d] = lo[d] if np.random.rand() < 0.5 else hi[d]
            init_X[i] = x_struct

        pop_F = []
        for i in range(pop_size):
            f = try_eval(init_X[i])
            if f is None:
                break
            pop_F.append(f)

        if not pop_F:
            return np.array([])

        pop_F = np.array(pop_F)
        pop_X = init_X[:len(pop_F)]

        # Record initial points
        all_X.extend([x.copy() for x in pop_X])
        all_F.extend([f.copy() for f in pop_F])

        # Main NSGA-II loop
        gen = 0
        while used < self.budget and not is_deadline():
            # Evaluate offspring
            offspring_X = []
            offspring_F = []

            # Create offspring population same size as parent
            while len(offspring_X) < pop_size and used < self.budget and not is_deadline():
                # Selection
                idx1 = tournament_select(pop_F, pop_rank, pop_cd)
                idx2 = tournament_select(pop_F, pop_rank, pop_cd)

                # Crossover
                c1, c2 = crossover(pop_X[idx1], pop_X[idx2])

                # Mutation
                c1 = mutate(c1)
                c2 = mutate(c2)

                # Evaluate
                f1 = try_eval(c1)
                if f1 is None:
                    break
                offspring_X.append(c1)
                offspring_F.append(f1)

                if len(offspring_X) >= pop_size:
                    break

                f2 = try_eval(c2)
                if f2 is None:
                    break
                offspring_X.append(c2)
                offspring_F.append(f2)

            if not offspring_X:
                break

            offspring_F = np.array(offspring_F)
            offspring_X = np.array(offspring_X)

            # Record offspring
            all_X.extend([x.copy() for x in offspring_X])
            all_F.extend([f.copy() for f in offspring_F])

            # Combine parent and offspring
            combined_X = np.vstack([pop_X, offspring_X])
            combined_F = np.vstack([pop_F, offspring_F])

            # Non-dominated sorting
            fronts = non_dominated_sort(combined_F)

            # Build new population
            new_pop_idx = []
            for front in fronts:
                if len(new_pop_idx) + len(front) <= pop_size:
                    new_pop_idx.extend(front)
                else:
                    # Need to select from this front using crowding distance
                    remaining = pop_size - len(new_pop_idx)
                    F_front = combined_F[front]
                    cd = crowding_distance(combined_F, front)
                    sorted_by_cd = np.argsort(cd)[::-1]
                    new_pop_idx.extend(front[sorted_by_cd[:remaining]])
                    break

            # Update population
            pop_X = combined_X[new_pop_idx]
            pop_F = combined_F[new_pop_idx]

            # Compute ranks and crowding distances for next generation
            fronts = non_dominated_sort(pop_F)
            pop_rank = np.full(len(pop_F), -1, dtype=int)
            for idx, front in enumerate(fronts):
                for i in front:
                    pop_rank[i] = idx

            # Compute crowding distance for each front
            pop_cd = np.zeros(len(pop_F))
            for front in fronts:
                cd = crowding_distance(pop_F, front)
                for i, j in enumerate(front):
                    pop_cd[j] = cd[i]

            gen += 1

        return np.array(all_X)
# EVOLVE-BLOCK-END
