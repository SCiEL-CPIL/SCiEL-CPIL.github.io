"""Multi-objective optimizer on WFG4 using a hybrid LHS + NSGA-II approach."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid multi-objective optimizer for WFG4: LHS initialization + NSGA-II refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 330s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(42)

        # Population size for NSGA-II phase
        pop_size = min(40, self.budget // 3)
        if pop_size < 10:
            pop_size = 10

        # Phase 1: Latin Hypercube Sampling for broad initial coverage
        n_lhs = min(60, self.budget // 3)
        if n_lhs < 10:
            n_lhs = 10

        # Generate proper LHS points
        lhs_points = []
        for j in range(self.dim):
            strata = np.linspace(lo[j], hi[j], n_lhs + 1)
            samples = np.array([strata[i] + (strata[i + 1] - strata[i]) * rng.random() for i in range(n_lhs)])
            if j == 0:
                lhs_points = [np.array([s]) for s in samples]
            else:
                for i in range(n_lhs):
                    lhs_points[i] = np.append(lhs_points[i], samples[i])

        # Evaluate LHS points and build initial population
        X = []
        F = []
        evaluations = 0

        for x in lhs_points:
            if evaluations >= self.budget:
                break
            if time.time() > deadline - 5.0:
                break
            f = problem(x)
            evaluations += 1
            X.append(x)
            F.append(f)

        # Ensure we have pop_size points
        while len(X) < pop_size and evaluations < self.budget:
            if time.time() > deadline - 5.0:
                break
            x = lo + (hi - lo) * rng.random(self.dim)
            f = problem(x)
            evaluations += 1
            X.append(x)
            F.append(f)

        # Truncate to pop_size if we have more
        X = X[:pop_size]
        F = F[:pop_size]

        def dominates(f1, f2):
            """Check if f1 dominates f2."""
            return np.all(f1 <= f2) and np.any(f1 < f2)

        def fast_non_dominated_sort(F_arr):
            """Fast non-dominated sort. Returns list of fronts (list of indices)."""
            n = len(F_arr)
            S = [[] for _ in range(n)]
            nd = np.zeros(n, dtype=int)
            fronts = [[]]

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F_arr[i] <= F_arr[j]) and np.any(F_arr[i] < F_arr[j]):
                        S[i].append(j)
                        nd[j] += 1
                    elif np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        nd[i] += 1

            for i in range(n):
                if nd[i] == 0:
                    fronts[0].append(i)

            i = 0
            while len(fronts[i]) > 0:
                Q = []
                for p in fronts[i]:
                    for q in S[p]:
                        nd[q] -= 1
                        if nd[q] == 0:
                            Q.append(q)
                i += 1
                if len(Q) > 0:
                    fronts.append(Q)
                else:
                    break

            return fronts[:-1] if len(fronts) > 1 else fronts

        def crowding_distance(F_arr, indices):
            """Compute crowding distance for a set of indices."""
            if len(indices) <= 2:
                return {i: float('inf') for i in indices}

            distances = {i: 0.0 for i in indices}
            for m in range(self.n_obj):
                values = [F_arr[i][m] for i in indices]
                min_val = min(values)
                max_val = max(values)
                if max_val - min_val < 1e-12:
                    continue
                sorted_idx = sorted(indices, key=lambda i: F_arr[i][m])
                distances[sorted_idx[0]] = float('inf')
                distances[sorted_idx[-1]] = float('inf')
                for k in range(1, len(sorted_idx) - 1):
                    distances[sorted_idx[k]] += (
                        F_arr[sorted_idx[k + 1]][m] - F_arr[sorted_idx[k - 1]][m]
                    ) / (max_val - min_val)
            return distances

        def sbc_crossover(x1, x2):
            """Simulated binary crossover."""
            eta = 20.0
            child1 = np.zeros_like(x1)
            child2 = np.zeros_like(x2)
            for d in range(self.dim):
                u = rng.random()
                if u <= 0.5:
                    beta = (2 * u) ** (1.0 / (eta + 1))
                else:
                    beta = (1 / (2 * (1 - u))) ** (1.0 / (eta + 1))
                child1[d] = 0.5 * ((1 + beta) * x1[d] + (1 - beta) * x2[d])
                child2[d] = 0.5 * ((1 - beta) * x1[d] + (1 + beta) * x2[d])
            return child1, child2

        def polynomial_mutation(x, sigma=0.1):
            """Polynomial mutation."""
            eta_m = 20.0
            result = x.copy()
            for d in range(self.dim):
                if rng.random() < 0.1:  # Mutation probability per gene
                    u = rng.random()
                    if u < 0.5:
                        delta = (2 * u) ** (1.0 / (eta_m + 1)) - 1
                    else:
                        delta = 1 - (2 * (1 - u)) ** (1.0 / (eta_m + 1))
                    result[d] += sigma * (hi[d] - lo[d]) * delta
            return np.clip(result, lo, hi)

        def tournament_select(pop_F, k=2):
            """Tournament selection based on non-dominated rank and crowding distance."""
            n = len(pop_F)
            fronts = fast_non_dominated_sort(np.array(pop_F))

            # Assign ranks
            rank = np.zeros(n, dtype=int)
            for r, front in enumerate(fronts):
                for idx in front:
                    rank[idx] = r

            # Select k individuals randomly, return the best one
            candidates = rng.choice(n, k, replace=False)
            best = candidates[0]
            for c in candidates[1:]:
                if rank[c] < rank[best]:
                    best = c
            return best

        # Phase 2: NSGA-II refinement
        while evaluations < self.budget:
            if time.time() > deadline - 5.0:
                break

            # Create offspring population
            offspring_X = []
            offspring_F = []
            n_offspring = min(pop_size, self.budget - evaluations)

            for _ in range(n_offspring):
                # Select parents using tournament selection
                p1 = tournament_select(F)
                p2 = tournament_select(F)

                # Crossover
                child1, child2 = sbc_crossover(X[p1], X[p2])

                # Mutation
                child1 = polynomial_mutation(child1)
                child2 = polynomial_mutation(child2)

                # Evaluate child1
                if evaluations < self.budget:
                    f1 = problem(child1)
                    evaluations += 1
                    offspring_X.append(child1)
                    offspring_F.append(f1)

                # Evaluate child2
                if evaluations < self.budget:
                    f2 = problem(child2)
                    evaluations += 1
                    offspring_X.append(child2)
                    offspring_F.append(f2)

            if len(offspring_X) == 0:
                break

            # Combine parent and offspring populations
            all_X = X + offspring_X
            all_F = F + offspring_F

            # Select next generation using non-dominated sorting and crowding distance
            n_select = min(pop_size, len(all_X))
            fronts = fast_non_dominated_sort(np.array(all_F))
            selected = []
            for front in fronts:
                if len(selected) + len(front) <= n_select:
                    selected.extend(front)
                else:
                    cd = crowding_distance(np.array(all_F), front)
                    sorted_front = sorted(front, key=lambda i: cd[i], reverse=True)
                    selected.extend(sorted_front[:n_select - len(selected)])
                    break

            # Update population
            X = [all_X[i] for i in selected]
            F = [all_F[i] for i in selected]

        # Return all evaluated points (both LHS and NSGA-II)
        # The evaluator records all returned objective vectors
        # We need to return the archive of non-dominated points
        # But the contract says "Every returned objective vector is recorded"
        # and the return value of __call__ is ignored
        # So we should return something that makes sense

        # Actually, looking at the contract more carefully:
        # "Every returned objective vector is recorded and the return value of __call__ is ignored"
        # This means the evaluator calls problem(x) and records the result
        # The return value of __call__ itself is ignored
        # So we just need to make sure we don't exceed budget and don't throw exceptions

        # Return the final population (or empty list, since it's ignored)
        return X
# EVOLVE-BLOCK-END
