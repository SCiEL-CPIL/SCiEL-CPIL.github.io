"""Multi-objective optimizer on WFG4 using NSGA-II with deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer with deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(42)

        # Population size
        pop_size = min(50, self.budget // 2)
        if pop_size < 10:
            pop_size = 10

        # Initialize population
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Track all evaluated points for hypervolume
        all_points = [F[i].copy() for i in range(pop_size)]

        # NSGA-II parameters
        crossover_rate = 0.9
        mutation_rate = 0.1  # per gene
        sigma = 0.2  # mutation scale

        def dominates(a, b):
            """Check if a dominates b (a is better or equal in all, better in at least one)."""
            return np.all(a <= b) and np.any(a < b)

        def non_dominated_sort(F):
            """Non-dominated sort. Returns list of fronts, each a list of indices."""
            n = len(F)
            fronts = []
            S = [[] for _ in range(n)]  # S[i] = set of indices dominated by i
            n_doms = [0] * n  # n_doms[i] = number of points dominating i
            front = []

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if dominates(F[i], F[j]):
                        S[i].append(j)
                    elif dominates(F[j], F[i]):
                        n_doms[i] += 1
                if n_doms[i] == 0:
                    front.append(i)

            if front:
                fronts.append(front)

            while front:
                next_front = []
                for i in front:
                    for j in S[i]:
                        n_doms[j] -= 1
                        if n_doms[j] == 0:
                            next_front.append(j)
                if next_front:
                    fronts.append(next_front)
                front = next_front

            return fronts

        def crowding_distance(F, front_indices):
            """Compute crowding distance for a front."""
            n = len(front_indices)
            if n <= 2:
                return [float('inf')] * n

            cd = [0.0] * n
            for m in range(self.n_obj):
                # Sort by objective m
                sorted_idx = sorted(front_indices, key=lambda i: F[i, m])
                cd[sorted_idx[0]] = float('inf')
                cd[sorted_idx[-1]] = float('inf')
                fmin = F[sorted_idx[0], m]
                fmax = F[sorted_idx[-1], m]
                if fmax - fmin < 1e-15:
                    continue
                for k in range(1, n - 1):
                    cd[sorted_idx[k]] += (F[sorted_idx[k + 1], m] - F[sorted_idx[k - 1], m]) / (fmax - fmin)
            return cd

        def tournament_select(fronts, cd_dict):
            """Tournament selection based on rank and crowding distance."""
            # Pick two random individuals
            i = rng.integers(len(F))
            j = rng.integers(len(F))

            # Find their ranks
            rank_i = None
            rank_j = None
            for r, front in enumerate(fronts):
                if i in front:
                    rank_i = r
                    break
                if j in front:
                    rank_j = r
                    break

            # If one has better rank, pick it
            if rank_i is not None and rank_j is not None:
                if rank_i < rank_j:
                    return i
                elif rank_j < rank_i:
                    return j
                else:
                    # Same rank, use crowding distance
                    cd_i = cd_dict.get(i, 0)
                    cd_j = cd_dict.get(j, 0)
                    if cd_i > cd_j:
                        return i
                    elif cd_j > cd_i:
                        return j
                    else:
                        return i if rng.random() < 0.5 else j
            elif rank_i is not None:
                return i
            elif rank_j is not None:
                return j
            else:
                return i if rng.random() < 0.5 else j

        def blend_crossover(p1, p2):
            """BLX-alpha crossover."""
            alpha = 0.5
            child = np.zeros(self.dim)
            for d in range(self.dim):
                lo_c = min(p1[d], p2[d]) - alpha * abs(p1[d] - p2[d])
                hi_c = max(p1[d], p2[d]) + alpha * abs(p1[d] - p2[d])
                child[d] = rng.uniform(lo_c, hi_c)
            return np.clip(child, lo, hi)

        def mutate(x):
            """Polynomial-style mutation with Gaussian perturbation."""
            x_mut = x.copy()
            for d in range(self.dim):
                if rng.random() < mutation_rate:
                    x_mut[d] += sigma * (hi[d] - lo[d]) * rng.standard_normal()
            return np.clip(x_mut, lo, hi)

        # Main NSGA-II loop
        while used < self.budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Non-dominated sort current population
            fronts = non_dominated_sort(F)

            # Compute crowding distance for each front
            cd_dict = {}
            for front in fronts:
                cd = crowding_distance(F, front)
                for idx, cd_val in zip(front, cd):
                    cd_dict[idx] = cd_val

            # Generate offspring
            offspring_X = []
            offspring_F = []

            # Number of offspring to generate (try to maintain population size)
            n_offspring = min(pop_size, self.budget - used)
            if n_offspring <= 0:
                break

            for _ in range(n_offspring):
                if time.time() > deadline:
                    break

                # Select parents via tournament
                p1 = tournament_select(fronts, cd_dict)
                p2 = tournament_select(fronts, cd_dict)

                # Crossover
                if rng.random() < crossover_rate:
                    child = blend_crossover(X[p1], X[p2])
                else:
                    child = X[p1].copy()

                # Mutation
                child = mutate(child)

                # Evaluate
                fc = problem(child)
                used += 1
                offspring_X.append(child)
                offspring_F.append(fc)
                all_points.append(fc.copy())

            if not offspring_X:
                break

            # Combine parents and offspring
            combined_X = np.vstack([X, np.array(offspring_X)])
            combined_F = np.vstack([F, np.array(offspring_F)])

            # Non-dominated sort combined population
            combined_fronts = non_dominated_sort(combined_F)

            # Compute crowding distance for each front in combined population
            combined_cd = {}
            for front in combined_fronts:
                cd = crowding_distance(combined_F, front)
                for idx, cd_val in zip(front, cd):
                    combined_cd[idx] = cd_val

            # Select next population
            new_X = []
            new_F = []
            for front in combined_fronts:
                if len(new_X) + len(front) <= pop_size:
                    for idx in front:
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                else:
                    # Need to select from this front based on crowding distance
                    remaining = pop_size - len(new_X)
                    if remaining <= 0:
                        break
                    # Sort front by crowding distance (descending)
                    sorted_front = sorted(front, key=lambda i: combined_cd.get(i, 0), reverse=True)
                    for idx in sorted_front[:remaining]:
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                    break

            X = np.array(new_X)
            F = np.array(new_F)

        # Return (return value is ignored, but we return the final population)
        return X

# EVOLVE-BLOCK-END
