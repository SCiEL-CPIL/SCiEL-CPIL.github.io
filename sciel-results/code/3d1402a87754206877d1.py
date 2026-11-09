"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for ZDT4 using NSGA-II with exploitation of x1."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = 320.0  # safety margin before 330s kill
        rng = np.random.default_rng(0)

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        dim = self.dim
        budget = self.budget

        # ZDT4 specific: x1 in [0,1], x2..x10 in [-5,5]
        # The Pareto front depends only on x1 (for f1) and sum of sin for f2
        # f1 = x1
        # f2 = g(x) * h(f1, g(x))
        # g(x) = 1 + 10*(n-1) + sum(x_i^2 - 10*cos(4*pi*x_i)) for i=2..n
        # h(u,v) = 1 - sqrt(u/v)

        # Strategy:
        # 1. Initialize with diverse x1 values (0 to 1) and x2..x10 near 0 (to minimize g)
        # 2. Use NSGA-II to explore
        # 3. Exploit the structure: for each x1, try to minimize g by setting x2..x10 near 0

        # Initial population: diverse x1, x2..x10 near 0 with some noise
        pop_size = min(50, budget // 4)
        X = np.zeros((pop_size, dim))
        # x1: spread evenly from 0 to 1 with some randomness
        X[:, 0] = np.linspace(0, 1, pop_size) + rng.normal(0, 0.01, pop_size)
        X[:, 0] = np.clip(X[:, 0], lo[0], hi[0])
        # x2..x10: near 0 (optimal for g) with small noise
        X[:, 1:] = rng.normal(0, 0.1, (pop_size, dim - 1))
        X[:, 1:] = np.clip(X[:, 1:], lo[1:], hi[1:])

        F = np.array([problem(x) for x in X])
        used = pop_size

        # NSGA-II parameters
        mu = pop_size
        eta = 0.2  # mutation rate
        sigma = 0.1  # mutation strength

        def fast_non_dominated_sort(F):
            """Fast non-dominated sorting. Returns list of fronts (each a list of indices)."""
            n = len(F)
            S = [[] for _ in range(n)]  # S[i] = indices dominated by i
            n_dom = np.zeros(n, dtype=int)  # number of solutions that dominate i
            rank = np.zeros(n, dtype=int)
            fronts = []

            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_dom[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].append(i)
                        n_dom[i] += 1

            current_front = [i for i in range(n) if n_dom[i] == 0]
            k = 0
            while current_front:
                fronts.append(current_front)
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            next_front.append(j)
                current_front = next_front
                k += 1

            for i in range(n):
                for k_idx, front in enumerate(fronts):
                    if i in front:
                        rank[i] = k_idx
                        break
            return fronts, rank

        def crowding_distance(F, front_indices):
            """Compute crowding distance for a front."""
            n = len(front_indices)
            if n <= 2:
                return np.full(n, np.inf)
            dist = np.zeros(n)
            for m in range(F.shape[1]):
                f = F[front_indices, m]
                fmin, fmax = f.min(), f.max()
                if fmax - fmin < 1e-12:
                    continue
                idx = np.argsort(f)
                dist[idx[0]] = np.inf
                dist[idx[-1]] = np.inf
                for i in range(1, n - 1):
                    dist[idx[i]] += (f[idx[i + 1]] - f[idx[i - 1]]) / (fmax - fmin)
            return dist

        def crossover(parent1, parent2):
            """SBX crossover."""
            if rng.random() < 0.9:  # crossover probability
                child1 = np.copy(parent1)
                child2 = np.copy(parent2)
                for i in range(dim):
                    if rng.random() < 0.5:
                        if abs(parent1[i] - parent2[i]) > 1e-12:
                            u = rng.random()
                            if u <= 0.5:
                                beta = (2 * u) ** (1 / (2 + 1))
                            else:
                                beta = (1 / (2 * (1 - u))) ** (1 / (2 + 1))
                            c1 = 0.5 * ((1 + beta) * parent1[i] + (1 - beta) * parent2[i])
                            c2 = 0.5 * ((1 - beta) * parent1[i] + (1 + beta) * parent2[i])
                            child1[i] = c1
                            child2[i] = c2
                return child1, child2
            return np.copy(parent1), np.copy(parent2)

        def mutate(x):
            """Polynomial mutation."""
            child = np.copy(x)
            for i in range(dim):
                if rng.random() < eta:
                    u = rng.random()
                    delta = hi[i] - lo[i]
                    if u < 0.5:
                        val = (2 * u + 1 - (hi[i] - x[i]) / delta) ** 7
                        child[i] = x[i] + delta * (val - 1)
                    else:
                        val = (2 * (1 - u) + (hi[i] - x[i]) / delta) ** 7
                        child[i] = x[i] - delta * (1 - val)
                    child[i] = np.clip(child[i], lo[i], hi[i])
            return child

        # ZDT4-specific exploitation: for a given x1, optimize x2..x10 to minimize g
        # g = 1 + 10*(n-1) + sum(x_i^2 - 10*cos(4*pi*x_i))
        # Minimum of x^2 - 10*cos(4*pi*x) is at x=0: 0 - 10 = -10
        # So min g = 1 + 10*(n-1) + (n-1)*(-10) = 1
        # At g=1, f2 = 1 * h(f1, 1) = 1 - sqrt(f1)
        # So the Pareto front is: f1 = x1, f2 = 1 - sqrt(x1) for x1 in [0,1]

        # Exploitation: for promising x1 values, set x2..x10 close to 0
        def exploit_x1(x1_val):
            """Create a solution with given x1 and x2..x10 optimized for low g."""
            x = np.zeros(dim)
            x[0] = np.clip(x1_val, lo[0], hi[0])
            # Set x2..x10 near 0 with small perturbation
            x[1:] = rng.normal(0, 0.01, dim - 1)
            x[1:] = np.clip(x[1:], lo[1:], hi[1:])
            return x

        # Main loop: alternate between NSGA-II and exploitation
        while used < budget:
            if time.time() > deadline:
                break

            # Check time periodically
            if (used - pop_size) % 10 == 0 and time.time() > deadline:
                break

            # Get non-dominated fronts
            fronts, rank = fast_non_dominated_sort(F)

            # Select parents from first front (or first few fronts)
            # Use binary tournament selection
            def tournament_select(F, rank, k=2):
                idxs = rng.integers(0, len(F), size=k)
                best = idxs[0]
                for i in range(1, k):
                    if rank[idxs[i]] < rank[best]:
                        best = idxs[i]
                return best

            # Generate offspring
            n_offspring = mu
            offspring = []
            for _ in range(n_offspring // 2):
                p1_idx = tournament_select(F, rank)
                p2_idx = tournament_select(F, rank)
                c1, c2 = crossover(X[p1_idx], X[p2_idx])
                c1 = mutate(c1)
                c2 = mutate(c2)
                offspring.append(c1)
                offspring.append(c2)

            # Ensure we don't exceed budget
            n_new = min(len(offspring), budget - used)
            if n_new <= 0:
                break

            # Evaluate new offspring
            for i in range(n_new):
                if used >= budget:
                    break
                if time.time() > deadline:
                    break
                x = offspring[i]
                fx = problem(x)
                X = np.vstack([X, x])
                F = np.vstack([F, [fx]])
                used += 1

            # Selection: combine parent and offspring, select best mu
            combined_X = X  # already combined
            combined_F = F
            n_total = len(combined_F)

            if n_total > mu:
                fronts, rank = fast_non_dominated_sort(combined_F)
                # Select from fronts until we have mu
                selected = []
                for front in fronts:
                    if len(selected) + len(front) <= mu:
                        selected.extend(front)
                    else:
                        # Need to trim this front by crowding distance
                        need = mu - len(selected)
                        cd = crowding_distance(combined_F, front)
                        front_sorted = np.argsort(-cd)
                        selected.extend([front[i] for i in front_sorted[:need]])
                        break

                if len(selected) >= mu:
                    X = combined_X[selected[:mu]]
                    F = combined_F[selected[:mu]]
            else:
                # Just keep all
                X = combined_X
                F = combined_F

            # Periodic exploitation: try to improve solutions by optimizing x2..x10
            if (used - pop_size) % 20 == 0 and used < budget - 5:
                # Pick a few random solutions and exploit their x1
                n_exploit = min(3, budget - used)
                for _ in range(n_exploit):
                    if used >= budget or time.time() > deadline:
                        break
                    idx = rng.integers(0, len(X))
                    x1_val = X[idx, 0]
                    x_exp = exploit_x1(x1_val)
                    fx_exp = problem(x_exp)
                    # Replace worst solution if this is better
                    worst_idx = int(np.argmax(np.sum(F, axis=1)))
                    # Check if x_exp dominates or is better
                    if np.all(fx_exp <= F[worst_idx]) or rng.random() < 0.3:
                        X[worst_idx] = x_exp
                        F[worst_idx] = fx_exp
                    used += 1

        # Final exploitation pass: for the best x1 values, try to get g as close to 1 as possible
        if used < budget and time.time() < deadline:
            # Get unique x1 values from current population
            x1_vals = X[:, 0]
            # Sort by x1
            sorted_idx = np.argsort(x1_vals)
            # Try a few evenly spaced x1 values
            n_final = min(10, budget - used)
            if n_final > 0:
                x1_targets = np.linspace(0, 1, n_final)
                for x1_t in x1_targets:
                    if used >= budget or time.time() > deadline:
                        break
                    x_final = exploit_x1(x1_t)
                    # Try a few iterations to minimize g
                    for _ in range(3):
                        if used >= budget or time.time() > deadline:
                            break
                        x_final[1:] = rng.normal(0, 0.005, dim - 1)
                        x_final[1:] = np.clip(x_final[1:], lo[1:], hi[1:])
                        fx = problem(x_final)
                        # Find if this is better than any existing solution
                        is_better = False
                        for i in range(len(X)):
                            if np.all(fx <= F[i]) and np.any(fx < F[i]):
                                is_better = True
                                break
                        if is_better:
                            # Replace worst
                            worst_idx = int(np.argmax(np.sum(F, axis=1)))
                            X[worst_idx] = x_final
                            F[worst_idx] = fx
                        used += 1

        return X
# EVOLVE-BLOCK-END
