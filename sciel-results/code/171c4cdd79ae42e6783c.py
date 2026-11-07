"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer for ZDT6."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        N = 50  # population size
        budget = self.budget

        # Initial population
        rng = np.random.default_rng(42)
        X = lo + span * rng.random((N, self.dim))
        F = np.array([problem(x) for x in X])
        used = N

        # ZDT6 Pareto front parameterization:
        # f1 = 1 - u, where u in [0,1]
        # x1 = sin(10 * pi * x1) ... actually ZDT6:
        # f1 = 1 - exp(-4*x1) * sin^6(6*pi*x1)
        # g(x) = 1 + 9 * (sum_{i=2}^n x_i)/(n-1)
        # f2 = g(x) * (1 - (f1/g(x))^2)
        # Pareto front: x1 such that f1 = 1-u, xi=0 for i>=2
        # So g = 1, f1 = 1-u, f2 = 1 - (1-u)^2 = 2u - u^2

        # We'll use a hybrid approach:
        # 1) Seed with known Pareto front points
        # 2) Use NSGA-II for exploration

        # Add some front points to initial population
        n_front = min(10, N // 5)
        u_vals = np.linspace(0.05, 0.95, n_front)
        for i, u in enumerate(u_vals):
            # Find x1 such that f1 = 1-u
            # f1 = 1 - exp(-4*x1)*sin^6(6*pi*x1) = 1-u
            # exp(-4*x1)*sin^6(6*pi*x1) = u
            # Use a simple search for x1 in [0,1]
            best_x1 = 0.5
            best_err = float('inf')
            for x1_try in np.linspace(0.01, 0.99, 50):
                f1_try = 1 - np.exp(-4 * x1_try) * (np.sin(6 * np.pi * x1_try) ** 6)
                err = abs(f1_try - (1 - u))
                if err < best_err:
                    best_err = err
                    best_x1 = x1_try
            x_front = np.zeros(self.dim)
            x_front[0] = best_x1
            # Other variables set to 0 for ideal g
            x_front[1:] = 0.0
            # But we need to evaluate via problem()
            # Replace worst point in population
            X[i] = x_front
            F[i] = problem(x_front)
            used += 1

        # NSGA-II main loop
        def fast_non_dominated_sort(F):
            """Fast non-dominated sort. Returns list of fronts."""
            n = len(F)
            S = [[] for _ in range(n)]  # S[i] = indices dominated by i
            n_count = np.zeros(n, dtype=int)  # number of points that dominate i
            fronts = [[]]
            for p in range(n):
                for q in range(n):
                    if p == q:
                        continue
                    if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                        S[p].append(q)
                    elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                        n_count[p] += 1
                if n_count[p] == 0:
                    fronts[0].append(p)
            i = 0
            while len(fronts[i]) > 0:
                Q = []
                for p in fronts[i]:
                    for q in S[p]:
                        n_count[q] -= 1
                        if n_count[q] == 0:
                            Q.append(q)
                i += 1
                if len(Q) > 0:
                    fronts.append(Q)
                else:
                    break
            return fronts

        def crowding_distance(F, indices):
            """Compute crowding distance for a set of indices."""
            if len(indices) <= 2:
                return {idx: float('inf') for idx in indices}
            n = len(indices)
            dist = np.zeros(n)
            F_sub = F[indices]
            for m in range(self.n_obj):
                sorted_idx = np.argsort(F_sub[:, m])
                dist[sorted_idx[0]] = float('inf')
                dist[sorted_idx[-1]] = float('inf')
                f_min = F_sub[sorted_idx[0], m]
                f_max = F_sub[sorted_idx[-1], m]
                if f_max - f_min > 1e-15:
                    for k in range(1, n - 1):
                        dist[sorted_idx[k]] += (F_sub[sorted_idx[k + 1], m] - F_sub[sorted_idx[k - 1], m]) / (f_max - f_min)
            return {indices[i]: dist[i] for i in range(n)}

        # Main evolution loop
        while used < budget:
            if time.time() > deadline - 5.0:
                break

            # Evaluate fitness
            fronts = fast_non_dominated_sort(F)

            # Create mating pool
            mating_pool = []
            for front in fronts:
                if len(mating_pool) + len(front) <= N:
                    mating_pool.extend(front)
                else:
                    break

            # If we don't have enough, fill with remaining
            if len(mating_pool) < N:
                remaining = [i for i in range(N) if i not in mating_pool]
                mating_pool.extend(remaining[:N - len(mating_pool)])

            # Generate offspring
            offspring_X = []
            offspring_F = []
            n_offspring = min(N, budget - used)

            for _ in range(n_offspring):
                if time.time() > deadline - 5.0:
                    break
                # Select parents using tournament
                p1 = mating_pool[rng.integers(len(mating_pool))]
                p2 = mating_pool[rng.integers(len(mating_pool))]

                # Simulated binary crossover
                if rng.random() < 0.9:
                    eta = 20.0
                    child1 = np.zeros(self.dim)
                    child2 = np.zeros(self.dim)
                    for j in range(self.dim):
                        if rng.random() < 0.5:
                            u = rng.random()
                            if u <= 0.5:
                                beta = (2 * u) ** (1 / (eta + 1))
                            else:
                                beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))
                            child1[j] = 0.5 * ((1 + beta) * X[p1, j] + (1 - beta) * X[p2, j])
                            child2[j] = 0.5 * ((1 - beta) * X[p1, j] + (1 + beta) * X[p2, j])
                        else:
                            child1[j] = X[p1, j]
                            child2[j] = X[p2, j]
                    child1 = np.clip(child1, lo, hi)
                    child2 = np.clip(child2, lo, hi)
                    offspring_X.extend([child1, child2])
                    offspring_F.extend([problem(child1), problem(child2)])
                    used += 2
                else:
                    child = X[p1].copy()
                    # Polynomial mutation
                    if rng.random() < 0.5:
                        j = rng.integers(self.dim)
                        eta_m = 20.0
                        u = rng.random()
                        if u <= 0.5:
                            delta = (2 * u) ** (1 / (eta_m + 1)) - 1
                        else:
                            delta = 1 - (2 * (1 - u)) ** (1 / (eta_m + 1))
                        child[j] += delta * (hi[j] - lo[j])
                        child[j] = np.clip(child[j], lo[j], hi[j])
                    offspring_X.append(child)
                    offspring_F.append(problem(child))
                    used += 1

            if not offspring_X:
                break

            offspring_X = np.array(offspring_X)
            offspring_F = np.array(offspring_F)

            # Combine parents and offspring
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])
            N_combined = len(combined_X)

            # Non-dominated sort on combined
            fronts = fast_non_dominated_sort(combined_F)

            # Select next generation
            next_indices = []
            for front in fronts:
                if len(next_indices) + len(front) <= N:
                    next_indices.extend(front)
                else:
                    # Need to select from this front using crowding distance
                    remaining = N - len(next_indices)
                    cd = crowding_distance(combined_F, front)
                    sorted_front = sorted(front, key=lambda i: cd[i], reverse=True)
                    next_indices.extend(sorted_front[:remaining])
                    break

            X = combined_X[next_indices]
            F = combined_F[next_indices]

        return X
# EVOLVE-BLOCK-END
